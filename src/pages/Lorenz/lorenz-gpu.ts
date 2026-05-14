/**
 * GPU-accelerated Lorenz particle system using WebGL2 transform feedback.
 *
 * Architecture (per frame):
 *   1. Integration pass  – vertex shader integrates N particles with RK4.
 *      Transform feedback captures:
 *        binding 0 → new state (x, y, z, prevPeak)
 *        binding 1 → section crossing (u, v, hasCrossed, 0)
 *        binding 2 → return-map peak pair (zPrev, zNew, hasPeak, 0)
 *      RASTERIZER_DISCARD on: no drawing happens.
 *   2. Section render pass – reads binding-1 output, renders crossing points
 *      to the section accumulation FBO (additive blend, RGBA8).
 *   3. Return-map render pass – reads binding-2 output, renders peak points
 *      to the return-map accumulation FBO.
 *   4. Display pass – tone-maps the selected accumulation texture to the
 *      WebGL canvas; caller uses ctx.drawImage() to blit it to the 2D canvas.
 *
 * The hidden WebGL canvas is 512×512; the two accumulation textures are also
 * 512×512. Neither is cleared between frames – density builds up naturally.
 * Calling reset() wipes the accumulation textures.
 */

import { createWebGL2Context } from '@/lib/gpu/context';
import { createProgram } from '@/lib/gpu/shader';

const N_PARTICLES = 65536;
const ACC_SIZE = 512;
const STEPS_PER_FRAME = 32;

type SectionAxis = 'x' | 'y' | 'z';

const SECTION_BOUNDS: Record<SectionAxis, readonly [number, number, number, number]> = {
  y: [-22, 22, -1, 52],
  z: [-22, 22, -28, 28],
  x: [-28, 28, -1, 52],
};
const RETURN_BOUNDS: readonly [number, number, number, number] = [30, 48, 30, 48];

const VERT_INTEGRATE = /* glsl */`#version 300 es
precision highp float;

in vec4 a_state;

uniform float u_dt;
uniform float u_sigma;
uniform float u_rho;
uniform float u_beta;
uniform int   u_steps;
uniform float u_sectionVal;
uniform int   u_sectionAxis;

out vec4 v_newState;
out vec4 v_sectionCross;
out vec4 v_returnPeak;

vec3 lorenz(vec3 p) {
  return vec3(
    u_sigma * (p.y - p.x),
    p.x * (u_rho - p.z) - p.y,
    p.x * p.y - u_beta * p.z
  );
}

vec3 rk4(vec3 p) {
  vec3 k1 = lorenz(p);
  vec3 k2 = lorenz(p + k1 * (u_dt * 0.5));
  vec3 k3 = lorenz(p + k2 * (u_dt * 0.5));
  vec3 k4 = lorenz(p + k3 * u_dt);
  return p + (k1 + 2.0*k2 + 2.0*k3 + k4) * (u_dt / 6.0);
}

#define MAX_STEPS 64

void main() {
  vec3 pos = a_state.xyz;
  float prevPk = a_state.w;

  vec2 sUV = vec2(0.0);
  bool sCrossed = false;
  vec2 rPair = vec2(0.0);
  bool rHasPeak = false;

  for (int i = 0; i < MAX_STEPS; i++) {
    if (i >= u_steps) break;

    vec3 op = pos;
    float dz_old = op.x * op.y - u_beta * op.z;
    vec3 np = rk4(op);
    float dz_new = np.x * np.y - u_beta * np.z;

    if (!sCrossed) {
      float ov = (u_sectionAxis == 0) ? op.x : (u_sectionAxis == 1) ? op.y : op.z;
      float nv = (u_sectionAxis == 0) ? np.x : (u_sectionAxis == 1) ? np.y : np.z;
      if (ov < u_sectionVal && nv >= u_sectionVal) {
        float f = (u_sectionVal - ov) / (nv - ov);
        vec3 ip = op + f * (np - op);
        sUV = (u_sectionAxis == 0) ? ip.yz
            : (u_sectionAxis == 1) ? ip.xz
            : ip.xy;
        sCrossed = true;
      }
    }

    if (!rHasPeak && dz_old > 0.0 && dz_new <= 0.0) {
      float f = dz_old / (dz_old - dz_new);
      float pkZ = op.z + f * (np.z - op.z);
      if (prevPk > 0.0) {
        rPair = vec2(prevPk, pkZ);
        rHasPeak = true;
      }
      prevPk = pkZ;
    }

    pos = np;
  }

  v_newState = vec4(pos, prevPk);
  v_sectionCross = vec4(sUV, sCrossed ? 1.0 : 0.0, 0.0);
  v_returnPeak = vec4(rPair, rHasPeak ? 1.0 : 0.0, 0.0);
  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}
`;

const FRAG_NOOP = /* glsl */`#version 300 es
precision mediump float;
out vec4 fragColor;
void main() { fragColor = vec4(0.0); }
`;

const VERT_RENDER = /* glsl */`#version 300 es
precision highp float;

in vec4 a_cross;
uniform vec4 u_bounds;

void main() {
  if (a_cross.z < 0.5) {
    gl_Position = vec4(-3.0, -3.0, 0.0, 1.0);
    gl_PointSize = 1.0;
    return;
  }
  float ndcX = 2.0 * (a_cross.x - u_bounds.x) / (u_bounds.y - u_bounds.x) - 1.0;
  float ndcY = 2.0 * (a_cross.y - u_bounds.z) / (u_bounds.w - u_bounds.z) - 1.0;
  gl_Position = vec4(ndcX, ndcY, 0.0, 1.0);
  gl_PointSize = 1.5;
}
`;

const FRAG_RENDER = /* glsl */`#version 300 es
precision mediump float;
uniform vec4 u_color;
out vec4 fragColor;
void main() { fragColor = u_color; }
`;

const VERT_QUAD = /* glsl */`#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}
`;

const FRAG_DISPLAY = /* glsl */`#version 300 es
precision highp float;
uniform sampler2D u_tex;
uniform int u_mode;
in vec2 v_uv;
out vec4 fragColor;

vec3 sectionColor(float t) {
  vec3 a = vec3(0.04, 0.03, 0.18);
  vec3 b = vec3(0.20, 0.55, 1.00);
  vec3 c = vec3(1.00, 1.00, 1.00);
  float s = clamp(t * 2.0, 0.0, 1.0);
  float r = clamp(t * 2.0 - 1.0, 0.0, 1.0);
  return mix(mix(a, b, s), c, r);
}

vec3 returnColor(float t) {
  vec3 a = vec3(0.12, 0.05, 0.01);
  vec3 b = vec3(1.00, 0.60, 0.05);
  vec3 c = vec3(1.00, 1.00, 0.90);
  float s = clamp(t * 2.0, 0.0, 1.0);
  float r = clamp(t * 2.0 - 1.0, 0.0, 1.0);
  return mix(mix(a, b, s), c, r);
}

void main() {
  vec4 acc = texture(u_tex, v_uv);
  float raw = (u_mode == 0) ? acc.r : acc.g;
  float t = pow(clamp(raw * 5.0, 0.0, 1.0), 0.38);
  vec3 col = (u_mode == 0) ? sectionColor(t) : returnColor(t);
  fragColor = vec4(col, 1.0);
}
`;

function cpuRk4(
  x: number,
  y: number,
  z: number,
  sigma: number,
  rho: number,
  beta: number,
  dt: number,
): [number, number, number] {
  const f = (x: number, y: number, z: number) => [
    sigma * (y - x), x * (rho - z) - y, x * y - beta * z,
  ] as const;
  const h = dt * 0.5;
  const [k1x, k1y, k1z] = f(x, y, z);
  const [k2x, k2y, k2z] = f(x + k1x * h, y + k1y * h, z + k1z * h);
  const [k3x, k3y, k3z] = f(x + k2x * h, y + k2y * h, z + k2z * h);
  const [k4x, k4y, k4z] = f(x + k3x * dt, y + k3y * dt, z + k3z * dt);
  const s = dt / 6;
  return [
    x + (k1x + 2 * k2x + 2 * k3x + k4x) * s,
    y + (k1y + 2 * k2y + 2 * k3y + k4y) * s,
    z + (k1z + 2 * k2z + 2 * k3z + k4z) * s,
  ];
}

function generateInitialState(
  sigma: number,
  rho: number,
  beta: number,
): Float32Array {
  const data = new Float32Array(N_PARTICLES * 4);
  let x = 0.1;
  let y = 0;
  let z = 0;
  const dt = 0.002;
  for (let i = 0; i < 8000; i++) {
    [x, y, z] = cpuRk4(x, y, z, sigma, rho, beta, dt);
  }
  for (let i = 0; i < N_PARTICLES; i++) {
    [x, y, z] = cpuRk4(x, y, z, sigma, rho, beta, dt);
    data[i * 4] = x;
    data[i * 4 + 1] = y;
    data[i * 4 + 2] = z;
    data[i * 4 + 3] = 0;
  }
  return data;
}

export interface LorenzGPUParams {
  dt: number;
  sigma: number;
  rho: number;
  beta: number;
  sectionAxis: SectionAxis;
  sectionVal: number;
}

export class LorenzGPU {
  readonly canvas: HTMLCanvasElement;
  private gl: WebGL2RenderingContext;
  private tf: WebGLTransformFeedback;

  private progIntegrate: WebGLProgram;
  private progRender: WebGLProgram;
  private progDisplay: WebGLProgram;

  private bufStateA: WebGLBuffer;
  private bufStateB: WebGLBuffer;
  private bufSection: WebGLBuffer;
  private bufReturn: WebGLBuffer;

  private vaoA: WebGLVertexArrayObject;
  private vaoB: WebGLVertexArrayObject;
  private vaoSection: WebGLVertexArrayObject;
  private vaoReturn: WebGLVertexArrayObject;

  private sectionFBO: WebGLFramebuffer;
  private sectionTex: WebGLTexture;
  private returnFBO: WebGLFramebuffer;
  private returnTex: WebGLTexture;

  private quadVBO: WebGLBuffer;
  private quadVAO: WebGLVertexArrayObject;

  private uDt: WebGLUniformLocation;
  private uSigma: WebGLUniformLocation;
  private uRho: WebGLUniformLocation;
  private uBeta: WebGLUniformLocation;
  private uSteps: WebGLUniformLocation;
  private uSectionVal: WebGLUniformLocation;
  private uSectionAxis: WebGLUniformLocation;

  private uBounds: WebGLUniformLocation;
  private uColor: WebGLUniformLocation;

  private uTex: WebGLUniformLocation;
  private uMode: WebGLUniformLocation;

  private ping = false;
  totalFrames = 0;

  constructor(sigma: number, rho: number, beta: number) {
    const canvas = document.createElement('canvas');
    canvas.width = ACC_SIZE;
    canvas.height = ACC_SIZE;
    this.canvas = canvas;

    const gl = createWebGL2Context(canvas, { preserveDrawingBuffer: true, alpha: false });
    if (!gl) throw new Error('WebGL2 not available');
    this.gl = gl;

    this.progIntegrate = createProgram(
      gl,
      VERT_INTEGRATE,
      FRAG_NOOP,
      ['v_newState', 'v_sectionCross', 'v_returnPeak'],
    );
    this.progRender = createProgram(gl, VERT_RENDER, FRAG_RENDER);
    this.progDisplay = createProgram(gl, VERT_QUAD, FRAG_DISPLAY);

    const pi = this.progIntegrate;
    this.uDt = gl.getUniformLocation(pi, 'u_dt')!;
    this.uSigma = gl.getUniformLocation(pi, 'u_sigma')!;
    this.uRho = gl.getUniformLocation(pi, 'u_rho')!;
    this.uBeta = gl.getUniformLocation(pi, 'u_beta')!;
    this.uSteps = gl.getUniformLocation(pi, 'u_steps')!;
    this.uSectionVal = gl.getUniformLocation(pi, 'u_sectionVal')!;
    this.uSectionAxis = gl.getUniformLocation(pi, 'u_sectionAxis')!;

    const pr = this.progRender;
    this.uBounds = gl.getUniformLocation(pr, 'u_bounds')!;
    this.uColor = gl.getUniformLocation(pr, 'u_color')!;

    const pd = this.progDisplay;
    this.uTex = gl.getUniformLocation(pd, 'u_tex')!;
    this.uMode = gl.getUniformLocation(pd, 'u_mode')!;

    const initialState = generateInitialState(sigma, rho, beta);
    const byteSize = N_PARTICLES * 4 * 4;

    this.bufStateA = this.makeBuffer(initialState);
    this.bufStateB = this.makeBuffer(null, byteSize);
    this.bufSection = this.makeBuffer(null, byteSize);
    this.bufReturn = this.makeBuffer(null, byteSize);

    this.tf = gl.createTransformFeedback()!;

    const aState = gl.getAttribLocation(this.progIntegrate, 'a_state');
    this.vaoA = this.makeStateVAO(this.bufStateA, aState);
    this.vaoB = this.makeStateVAO(this.bufStateB, aState);

    const aCross = gl.getAttribLocation(this.progRender, 'a_cross');
    this.vaoSection = this.makeCrossVAO(this.bufSection, aCross);
    this.vaoReturn = this.makeCrossVAO(this.bufReturn, aCross);

    [this.sectionFBO, this.sectionTex] = this.makeAccFBO();
    [this.returnFBO, this.returnTex] = this.makeAccFBO();

    this.quadVBO = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVBO);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1, 1, -1, -1, 1,
      -1, 1, 1, -1, 1, 1,
    ]), gl.STATIC_DRAW);
    this.quadVAO = gl.createVertexArray()!;
    gl.bindVertexArray(this.quadVAO);
    const aPos = gl.getAttribLocation(this.progDisplay, 'a_pos');
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    this.reset();
  }

  private makeBuffer(data: Float32Array | null, size?: number): WebGLBuffer {
    const gl = this.gl;
    const buf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    if (data) {
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_COPY);
    } else {
      gl.bufferData(gl.ARRAY_BUFFER, size!, gl.DYNAMIC_COPY);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return buf;
  }

  private makeStateVAO(buf: WebGLBuffer, attribLoc: number): WebGLVertexArrayObject {
    const gl = this.gl;
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(attribLoc);
    gl.vertexAttribPointer(attribLoc, 4, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    return vao;
  }

  private makeCrossVAO(buf: WebGLBuffer, attribLoc: number): WebGLVertexArrayObject {
    const gl = this.gl;
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(attribLoc);
    gl.vertexAttribPointer(attribLoc, 4, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    return vao;
  }

  private makeAccFBO(): [WebGLFramebuffer, WebGLTexture] {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, ACC_SIZE, ACC_SIZE, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);

    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return [fbo, tex];
  }

  step(p: LorenzGPUParams): void {
    const gl = this.gl;
    const axisInt = p.sectionAxis === 'x' ? 0 : p.sectionAxis === 'y' ? 1 : 2;

    const srcVAO = this.ping ? this.vaoB : this.vaoA;
    const dstBufState = this.ping ? this.bufStateA : this.bufStateB;

    gl.useProgram(this.progIntegrate);
    gl.uniform1f(this.uDt, p.dt);
    gl.uniform1f(this.uSigma, p.sigma);
    gl.uniform1f(this.uRho, p.rho);
    gl.uniform1f(this.uBeta, p.beta);
    gl.uniform1i(this.uSteps, STEPS_PER_FRAME);
    gl.uniform1f(this.uSectionVal, p.sectionVal);
    gl.uniform1i(this.uSectionAxis, axisInt);

    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.tf);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, dstBufState);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 1, this.bufSection);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 2, this.bufReturn);

    gl.enable(gl.RASTERIZER_DISCARD);
    gl.bindVertexArray(srcVAO);
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS, 0, N_PARTICLES);
    gl.endTransformFeedback();
    gl.disable(gl.RASTERIZER_DISCARD);

    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
    gl.bindVertexArray(null);

    this.ping = !this.ping;

    const sb = SECTION_BOUNDS[p.sectionAxis];
    this.accumulatePass(
      this.sectionFBO,
      this.vaoSection,
      [sb[0], sb[1], sb[2], sb[3]],
      [0.003, 0.0, 0.004, 0.003],
    );

    const rb = RETURN_BOUNDS;
    this.accumulatePass(
      this.returnFBO,
      this.vaoReturn,
      [rb[0], rb[1], rb[2], rb[3]],
      [0.0, 0.003, 0.0, 0.003],
    );

    this.totalFrames++;
  }

  private accumulatePass(
    fbo: WebGLFramebuffer,
    vao: WebGLVertexArrayObject,
    bounds: readonly [number, number, number, number],
    color: readonly [number, number, number, number],
  ): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.viewport(0, 0, ACC_SIZE, ACC_SIZE);
    gl.enable(gl.BLEND);
    gl.blendEquation(gl.FUNC_ADD);
    gl.blendFunc(gl.ONE, gl.ONE);

    gl.useProgram(this.progRender);
    gl.uniform4f(this.uBounds, bounds[0], bounds[1], bounds[2], bounds[3]);
    gl.uniform4f(this.uColor, color[0], color[1], color[2], color[3]);

    gl.bindVertexArray(vao);
    gl.drawArrays(gl.POINTS, 0, N_PARTICLES);
    gl.bindVertexArray(null);

    gl.disable(gl.BLEND);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  drawPanel(
    ctx: CanvasRenderingContext2D,
    mode: 0 | 1,
    destX: number,
    destY: number,
    destW: number,
    destH: number,
  ): void {
    const gl = this.gl;
    const tex = mode === 0 ? this.sectionTex : this.returnTex;

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, ACC_SIZE, ACC_SIZE);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.progDisplay);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(this.uTex, 0);
    gl.uniform1i(this.uMode, mode);

    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindVertexArray(null);

    ctx.drawImage(this.canvas, 0, 0, ACC_SIZE, ACC_SIZE, destX, destY, destW, destH);
  }

  reset(): void {
    const gl = this.gl;
    for (const fbo of [this.sectionFBO, this.returnFBO]) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.totalFrames = 0;
  }

  dispose(): void {
    const gl = this.gl;
    gl.deleteProgram(this.progIntegrate);
    gl.deleteProgram(this.progRender);
    gl.deleteProgram(this.progDisplay);
    for (const buf of [this.bufStateA, this.bufStateB, this.bufSection, this.bufReturn, this.quadVBO]) {
      gl.deleteBuffer(buf);
    }
    for (const vao of [this.vaoA, this.vaoB, this.vaoSection, this.vaoReturn, this.quadVAO]) {
      gl.deleteVertexArray(vao);
    }
    gl.deleteTransformFeedback(this.tf);
    gl.deleteFramebuffer(this.sectionFBO);
    gl.deleteFramebuffer(this.returnFBO);
    gl.deleteTexture(this.sectionTex);
    gl.deleteTexture(this.returnTex);
  }
}
