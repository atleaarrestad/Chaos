/**
 * GPU-accelerated Gray-Scott reaction-diffusion using WebGL2 ping-pong
 * render-to-texture.
 *
 * Architecture per frame:
 *   1. Step pass (×stepsPerFrame) – full-screen fragment shader executes one
 *      Gray-Scott iteration; reads from texture A (or B), writes to FBO B (or A).
 *   2. Render pass – maps the V-channel to a teal color gradient and draws
 *      the result to the class canvas so the React layer can drawImage it.
 *
 * Seeding – a dedicated shader pass reads the current texture, overwrites a
 * circular region with activator values, and writes to the ping-pong dest —
 * no CPU readback required.
 */

import { createWebGL2Context } from '@/lib/gpu/context';
import { createProgram } from '@/lib/gpu/shader';

export const SIM_W = 512;
export const SIM_H = 512;

// ─── Shaders ──────────────────────────────────────────────────────────────────

/**
 * Shared vertex shader for all full-screen passes.
 * Y is flipped so v_uv(0,0) = top-left, matching HTML canvas coordinates.
 */
const VERT_QUAD = /* glsl */`#version 300 es
layout(location=0) in vec2 a_position;
out vec2 v_uv;
void main() {
  v_uv = vec2(a_position.x * 0.5 + 0.5, 0.5 - a_position.y * 0.5);
  gl_Position = vec4(a_position, 0.0, 1.0);
}`;

/** One Gray-Scott time step. Reads RG state (U,V), outputs new (U,V). */
const FRAG_STEP = /* glsl */`#version 300 es
precision highp float;

uniform sampler2D u_state;
uniform vec2  u_texelSize;
uniform float u_f;
uniform float u_k;
uniform float u_Du;
uniform float u_Dv;

in  vec2 v_uv;
out vec4 fragColor;

void main() {
  vec2 ts = u_texelSize;
  vec2 c  = texture(u_state, v_uv).rg;
  vec2 nb = texture(u_state, v_uv + vec2( ts.x, 0.0)).rg
          + texture(u_state, v_uv + vec2(-ts.x, 0.0)).rg
          + texture(u_state, v_uv + vec2(0.0,  ts.y)).rg
          + texture(u_state, v_uv + vec2(0.0, -ts.y)).rg;

  vec2  lap = nb - 4.0 * c;
  float u   = c.r;
  float v   = c.g;
  float uvv = u * v * v;

  float newU = clamp(u + u_Du * lap.r - uvv + u_f * (1.0 - u), 0.0, 1.0);
  float newV = clamp(v + u_Dv * lap.g + uvv - (u_f + u_k) * v, 0.0, 1.0);

  fragColor = vec4(newU, newV, 0.0, 1.0);
}`;

/** Maps V-channel to the teal gradient and outputs to screen. */
const FRAG_RENDER = /* glsl */`#version 300 es
precision highp float;

uniform sampler2D u_state;

in  vec2 v_uv;
out vec4 fragColor;

void main() {
  float v = texture(u_state, v_uv).g;
  fragColor = vec4(
    (3.0  + v *  20.0) / 255.0,
    (13.0 + v * 195.0) / 255.0,
    (13.0 + v * 178.0) / 255.0,
    1.0
  );
}`;


// ─── Public types ─────────────────────────────────────────────────────────────

export interface RDParams {
  f:  number;
  k:  number;
  Du: number;
  Dv: number;
}

// ─── GPU class ────────────────────────────────────────────────────────────────

export class ReactionDiffusionGPU {
  readonly canvas: HTMLCanvasElement;
  private gl: WebGL2RenderingContext;

  private progStep:   WebGLProgram;
  private progRender: WebGLProgram;

  private texA: WebGLTexture;
  private texB: WebGLTexture;
  private fbA:  WebGLFramebuffer;
  private fbB:  WebGLFramebuffer;

  private quadVAO: WebGLVertexArrayObject;
  private quadBuf: WebGLBuffer;

  // Step uniforms
  private uStepState: WebGLUniformLocation;
  private uTexelSize: WebGLUniformLocation;
  private uF:         WebGLUniformLocation;
  private uK:         WebGLUniformLocation;
  private uDu:        WebGLUniformLocation;
  private uDv:        WebGLUniformLocation;

  // Render uniforms
  private uRenderState: WebGLUniformLocation;

  private seedScratch = new Float32Array((SIM_W * 2 + 1) * 2);

  /** false → texA is current read, texB is current write */
  private ping = false;

  constructor(initialU: Float32Array, initialV: Float32Array) {
    const canvas = document.createElement('canvas');
    canvas.width  = SIM_W;
    canvas.height = SIM_H;
    this.canvas = canvas;

    const gl = createWebGL2Context(canvas, {
      alpha: false, antialias: false, depth: false, stencil: false,
    });
    if (!gl) throw new Error('WebGL2 not available');
    this.gl = gl;

    if (!gl.getExtension('EXT_color_buffer_float')) {
      throw new Error('EXT_color_buffer_float not supported — GPU simulation requires float render targets');
    }

    this.progStep   = createProgram(gl, VERT_QUAD, FRAG_STEP);
    this.progRender = createProgram(gl, VERT_QUAD, FRAG_RENDER);

    const ps = this.progStep;
    this.uStepState = gl.getUniformLocation(ps, 'u_state')!;
    this.uTexelSize = gl.getUniformLocation(ps, 'u_texelSize')!;
    this.uF         = gl.getUniformLocation(ps, 'u_f')!;
    this.uK         = gl.getUniformLocation(ps, 'u_k')!;
    this.uDu        = gl.getUniformLocation(ps, 'u_Du')!;
    this.uDv        = gl.getUniformLocation(ps, 'u_Dv')!;

    this.uRenderState = gl.getUniformLocation(this.progRender, 'u_state')!;

    const data = interleaveRG(initialU, initialV);
    this.texA = this.makeTex(data);
    this.texB = this.makeTex(data);
    this.fbA  = this.makeFB(this.texA);
    this.fbB  = this.makeFB(this.texB);

    [this.quadVAO, this.quadBuf] = this.makeQuad();
  }

  private makeTex(data: Float32Array | null): WebGLTexture {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    // REPEAT gives toroidal (periodic) boundary conditions for free
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG32F, SIM_W, SIM_H, 0, gl.RG, gl.FLOAT, data);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return tex;
  }

  private makeFB(tex: WebGLTexture): WebGLFramebuffer {
    const gl = this.gl;
    const fb = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`Framebuffer incomplete (0x${status.toString(16)})`);
    }
    return fb;
  }

  private makeQuad(): [WebGLVertexArrayObject, WebGLBuffer] {
    const gl = this.gl;
    const positions = new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]);
    const buf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return [vao, buf];
  }

  /** Run `count` Gray-Scott steps via ping-pong framebuffers. */
  step(params: RDParams, count: number): void {
    const gl = this.gl;
    gl.viewport(0, 0, SIM_W, SIM_H);
    gl.useProgram(this.progStep);
    gl.uniform2f(this.uTexelSize, 1.0 / SIM_W, 1.0 / SIM_H);
    gl.uniform1f(this.uF,  params.f);
    gl.uniform1f(this.uK,  params.k);
    gl.uniform1f(this.uDu, params.Du);
    gl.uniform1f(this.uDv, params.Dv);
    gl.bindVertexArray(this.quadVAO);

    for (let i = 0; i < count; i++) {
      const srcTex = this.ping ? this.texB : this.texA;
      const dstFB  = this.ping ? this.fbA  : this.fbB;
      gl.bindFramebuffer(gl.FRAMEBUFFER, dstFB);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, srcTex);
      gl.uniform1i(this.uStepState, 0);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      this.ping = !this.ping;
    }

    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /** Color-map the current state and render to this.canvas. */
  render(): void {
    const gl = this.gl;
    const curTex = this.ping ? this.texB : this.texA;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, SIM_W, SIM_H);
    gl.useProgram(this.progRender);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, curTex);
    gl.uniform1i(this.uRenderState, 0);
    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindVertexArray(null);
  }

  /**
   * Seed activator at simulation pixel (sx, sy) with given pixel radius.
   * Uses CPU-side texSubImage2D row uploads — avoids any UV/flip ambiguity
   * that a full-screen GPU pass would have.  Both ping-pong textures are
   * updated so the seed is immediately visible regardless of current state.
   */
  seedAt(sx: number, sy: number, radius: number): void {
    const gl  = this.gl;
    const r   = Math.ceil(radius);
    const r2  = radius * radius;
    const buf = this.seedScratch;

    for (const tex of [this.texA, this.texB]) {
      gl.bindTexture(gl.TEXTURE_2D, tex);

      for (let dy = -r; dy <= r; dy++) {
        const rowY = sy + dy;
        if (rowY < 0 || rowY >= SIM_H) continue;
        const hw = Math.floor(Math.sqrt(Math.max(0, r2 - dy * dy)));
        const x0 = Math.max(0, sx - hw);
        const x1 = Math.min(SIM_W - 1, sx + hw);
        const w  = x1 - x0 + 1;
        if (w <= 0) continue;
        for (let i = 0; i < w; i++) {
          buf[i * 2]     = 0.5 + (Math.random() - 0.5) * 0.1;
          buf[i * 2 + 1] = 0.25 + Math.random() * 0.05;
        }
        gl.texSubImage2D(gl.TEXTURE_2D, 0, x0, rowY, w, 1, gl.RG, gl.FLOAT, buf);
      }

      gl.bindTexture(gl.TEXTURE_2D, null);
    }
    // ping unchanged — no ping-pong flip needed
  }

  /** Upload new initial state to both textures and reset ping-pong. */
  reset(initialU: Float32Array, initialV: Float32Array): void {
    const gl = this.gl;
    const data = interleaveRG(initialU, initialV);
    for (const tex of [this.texA, this.texB]) {
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, SIM_W, SIM_H, gl.RG, gl.FLOAT, data);
    }
    gl.bindTexture(gl.TEXTURE_2D, null);
    this.ping = false;
  }

  dispose(): void {
    const gl = this.gl;
    gl.deleteProgram(this.progStep);
    gl.deleteProgram(this.progRender);
    gl.deleteTexture(this.texA);
    gl.deleteTexture(this.texB);
    gl.deleteFramebuffer(this.fbA);
    gl.deleteFramebuffer(this.fbB);
    gl.deleteVertexArray(this.quadVAO);
    gl.deleteBuffer(this.quadBuf);
  }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function interleaveRG(u: Float32Array, v: Float32Array): Float32Array {
  const n = u.length;
  const data = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    data[i * 2]     = u[i];
    data[i * 2 + 1] = v[i];
  }
  return data;
}
