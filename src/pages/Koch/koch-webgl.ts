/**
 * WebGL2 renderer for the Koch snowflake fractal.
 *
 * Architecture:
 *   1. CPU generates Koch polygon vertices for the requested depth (cached).
 *   2. Two VBOs are uploaded to the GPU:
 *        fillBuf    — center + Koch polygon + close vertex  (TRIANGLE_FAN)
 *        outlineBuf — Koch polygon vertices only            (LINE_LOOP)
 *   3. render() performs up to four passes per call:
 *        a. Glow passes  — outline at 1.04× / 1.02× scale, low alpha, additive blend
 *        b. Fill pass    — TRIANGLE_FAN with semi-transparent alpha
 *        c. Outline pass — LINE_LOOP with high alpha
 *
 * The Koch polygon lives in unit-circle space (circumradius ≈ 1).
 * The vertex shader applies rotation (cos/sin uniforms), per-axis scale
 * (aspect-ratio + zoom), and NDC pan.
 */

import { createWebGL2Context } from '@/lib/gpu/context';
import { createProgram } from '@/lib/gpu/shader';

// ─── Geometry ─────────────────────────────────────────────────────────────────

export type ColorSchemeId = 'frost' | 'aurora' | 'fire' | 'mono';
export type FillModeId    = 'filled' | 'outline' | 'both';

const SQRT3   = Math.sqrt(3);
const SQRT3_6 = SQRT3 / 6;

type Pt = readonly [number, number];

function kochSubdivide(verts: Pt[], antiKoch: boolean): Pt[] {
  const result: Pt[] = [];
  const n    = verts.length;
  const sign = antiKoch ? -1 : 1;
  for (let i = 0; i < n; i++) {
    const A = verts[i];
    const B = verts[(i + 1) % n];
    const dx = B[0] - A[0];
    const dy = B[1] - A[1];
    const P1: Pt = [(2 * A[0] + B[0]) / 3, (2 * A[1] + B[1]) / 3];
    const P2: Pt = [(A[0] + 2 * B[0]) / 3, (A[1] + 2 * B[1]) / 3];
    const mx = (P1[0] + P2[0]) / 2;
    const my = (P1[1] + P2[1]) / 2;
    // Outward apex: right of CCW edge direction = CW 90° of (dx,dy) = (dy,−dx)
    const C: Pt = [mx + sign * SQRT3_6 * dy, my - sign * SQRT3_6 * dx];
    result.push(A, P1, C, P2);
  }
  return result;
}

/** Generate Koch snowflake polygon vertices (CCW, circumradius ≈ 1). */
export function generateKoch(depth: number, antiKoch = false): Pt[] {
  // Equilateral triangle, CCW, circumradius = 1, vertex pointing up.
  let verts: Pt[] = [
    [0, 1],
    [-SQRT3 / 2, -0.5],
    [SQRT3 / 2, -0.5],
  ];
  for (let d = 0; d < depth; d++) {
    verts = kochSubdivide(verts, antiKoch);
  }
  return verts;
}

// ─── Shaders ──────────────────────────────────────────────────────────────────

const VS = /* glsl */`#version 300 es
in  vec2  a_pos;
in  float a_t;

uniform float u_sx;     // x scale (aspect + zoom)
uniform float u_sy;     // y scale (aspect + zoom)
uniform float u_tx;     // x pan (NDC)
uniform float u_ty;     // y pan (NDC)
uniform float u_cr;     // cos(rotation)
uniform float u_sr;     // sin(rotation)
uniform float u_gscale; // extra scale for glow passes

out float v_t;

void main() {
  // Rotate
  vec2 r = vec2(a_pos.x * u_cr - a_pos.y * u_sr,
                a_pos.x * u_sr + a_pos.y * u_cr);
  gl_Position = vec4(r.x * u_gscale * u_sx + u_tx,
                     r.y * u_gscale * u_sy + u_ty,
                     0.0, 1.0);
  v_t = a_t;
}
`;

const FS = /* glsl */`#version 300 es
precision mediump float;

in  float v_t;

uniform int   u_cs;    // color scheme index
uniform float u_alpha;

out vec4 fragColor;

vec3 frost(float t) {
  return mix(vec3(0.18, 0.58, 0.97), vec3(0.88, 0.96, 1.00), t);
}

vec3 aurora(float t) {
  vec3 a = vec3(0.04, 0.88, 0.52);
  vec3 b = vec3(0.65, 0.08, 1.00);
  return t < 0.5 ? mix(a, b, t * 2.0) : mix(b, a, t * 2.0 - 1.0);
}

vec3 fire(float t) {
  vec3 a = vec3(0.95, 0.12, 0.02);
  vec3 b = vec3(1.00, 0.65, 0.00);
  vec3 c = vec3(1.00, 1.00, 0.80);
  return t < 0.5 ? mix(a, b, t * 2.0) : mix(b, c, t * 2.0 - 1.0);
}

void main() {
  float t = fract(v_t);
  vec3 col;
  if      (u_cs == 0) col = frost(t);
  else if (u_cs == 1) col = aurora(t);
  else if (u_cs == 2) col = fire(t);
  else                col = vec3(0.82, 0.92, 1.00);
  fragColor = vec4(col, u_alpha);
}
`;

// ─── Renderer ─────────────────────────────────────────────────────────────────

const CS_INDEX: Record<ColorSchemeId, number> = { frost: 0, aurora: 1, fire: 2, mono: 3 };

export interface KochRenderParams {
  depth:       number;
  antiKoch:    boolean;
  colorScheme: ColorSchemeId;
  fillMode:    FillModeId;
  glow:        boolean;
  zoom:        number;
  panX:        number;
  panY:        number;
  rotation:    number;
}

export interface KochWebGLRenderer {
  render(params: KochRenderParams): void;
  dispose(): void;
}

export function detectWebGL(): boolean {
  try {
    const c = document.createElement('canvas');
    const gl = c.getContext('webgl2');
    if (!gl) return false;
    // Immediately release the test context so it doesn't count against the
    // browser's active-context limit (~16 in Chrome).
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return true;
  } catch { return false; }
}

export function createKochRenderer(canvas: HTMLCanvasElement): KochWebGLRenderer | null {
  const gl = createWebGL2Context(canvas, {
    // alpha: false makes the canvas opaque so the CSS background (#070712)
    // never bleeds through. preserveDrawingBuffer must stay false (the
    // default) so the compositor only sees fully-composited frames — with
    // it true the GPU's async clear can be sampled before drawArrays
    // completes, producing a one-frame black flash.
    alpha: false, antialias: true, depth: false, stencil: false,
  });
  if (!gl) return null;

  // Alias with non-null assertion so inner closures don't need to re-check.
  const ctx = gl as WebGL2RenderingContext;

  let prog: WebGLProgram;
  try {
    prog = createProgram(gl, VS, FS);
  } catch (err) {
    console.error('[koch-webgl]', err);
    return null;
  }

  const aPos   = gl.getAttribLocation(prog, 'a_pos');
  const aT     = gl.getAttribLocation(prog, 'a_t');
  const uSx    = gl.getUniformLocation(prog, 'u_sx')!;
  const uSy    = gl.getUniformLocation(prog, 'u_sy')!;
  const uTx    = gl.getUniformLocation(prog, 'u_tx')!;
  const uTy    = gl.getUniformLocation(prog, 'u_ty')!;
  const uCr    = gl.getUniformLocation(prog, 'u_cr')!;
  const uSr    = gl.getUniformLocation(prog, 'u_sr')!;
  const uGs    = gl.getUniformLocation(prog, 'u_gscale')!;
  const uCs    = gl.getUniformLocation(prog, 'u_cs')!;
  const uAlpha = gl.getUniformLocation(prog, 'u_alpha')!;

  // Geometry cache
  let curKey       = '';
  let fillVAO:    WebGLVertexArrayObject | null = null;
  let outlineVAO: WebGLVertexArrayObject | null = null;
  let fillBuf:    WebGLBuffer | null = null;
  let outlineBuf: WebGLBuffer | null = null;
  let nVerts = 0;

  function makeVAO(buf: WebGLBuffer): WebGLVertexArrayObject {
    const vao = ctx.createVertexArray()!;
    ctx.bindVertexArray(vao);
    ctx.bindBuffer(ctx.ARRAY_BUFFER, buf);
    // Interleaved: [x f32, y f32, t f32]  stride = 12 bytes
    ctx.enableVertexAttribArray(aPos);
    ctx.vertexAttribPointer(aPos, 2, ctx.FLOAT, false, 12, 0);
    ctx.enableVertexAttribArray(aT);
    ctx.vertexAttribPointer(aT, 1, ctx.FLOAT, false, 12, 8);
    ctx.bindVertexArray(null);
    return vao;
  }

  function syncGeometry(depth: number, antiKoch: boolean): void {
    const key = `${depth}-${antiKoch}`;
    if (key === curKey) return;
    curKey = key;

    const pts = generateKoch(depth, antiKoch);
    nVerts = pts.length;

    // Fill buffer: [center, ...pts, pts[0]] for TRIANGLE_FAN
    const fillData = new Float32Array((2 + nVerts) * 3);
    fillData[0] = 0; fillData[1] = 0; fillData[2] = 0.5;
    for (let i = 0; i < nVerts; i++) {
      const o = (1 + i) * 3;
      fillData[o]     = pts[i][0];
      fillData[o + 1] = pts[i][1];
      fillData[o + 2] = i / nVerts;
    }
    const co = (1 + nVerts) * 3;
    fillData[co] = pts[0][0]; fillData[co + 1] = pts[0][1]; fillData[co + 2] = 0;

    // Outline buffer
    const outlineData = new Float32Array(nVerts * 3);
    for (let i = 0; i < nVerts; i++) {
      outlineData[i * 3]     = pts[i][0];
      outlineData[i * 3 + 1] = pts[i][1];
      outlineData[i * 3 + 2] = i / nVerts;
    }

    if (fillVAO)    ctx.deleteVertexArray(fillVAO);
    if (outlineVAO) ctx.deleteVertexArray(outlineVAO);
    if (fillBuf)    ctx.deleteBuffer(fillBuf);
    if (outlineBuf) ctx.deleteBuffer(outlineBuf);

    fillBuf = ctx.createBuffer()!;
    ctx.bindBuffer(ctx.ARRAY_BUFFER, fillBuf);
    ctx.bufferData(ctx.ARRAY_BUFFER, fillData, ctx.STATIC_DRAW);

    outlineBuf = ctx.createBuffer()!;
    ctx.bindBuffer(ctx.ARRAY_BUFFER, outlineBuf);
    ctx.bufferData(ctx.ARRAY_BUFFER, outlineData, ctx.STATIC_DRAW);

    fillVAO    = makeVAO(fillBuf);
    outlineVAO = makeVAO(outlineBuf);
  }

  return {
    render(p: KochRenderParams) {
      syncGeometry(p.depth, p.antiKoch);

      const W = canvas.width, H = canvas.height;
      if (W === 0 || H === 0) return;

      ctx.viewport(0, 0, W, H);
      // Clear to the container background colour (#070712) so any missed
      // draw frame shows the background rather than black.
      ctx.clearColor(7 / 255, 7 / 255, 18 / 255, 1);
      ctx.clear(ctx.COLOR_BUFFER_BIT);
      ctx.useProgram(prog);

      // Transform: Koch unit circle → NDC, preserving aspect ratio
      const minDim = Math.min(W, H);
      const base   = 0.88 * p.zoom;
      const sx = base * minDim / W;
      const sy = base * minDim / H;

      ctx.uniform1f(uSx, sx);
      ctx.uniform1f(uSy, sy);
      ctx.uniform1f(uTx, p.panX);
      ctx.uniform1f(uTy, p.panY);
      ctx.uniform1f(uCr, Math.cos(p.rotation));
      ctx.uniform1f(uSr, Math.sin(p.rotation));
      ctx.uniform1i(uCs, CS_INDEX[p.colorScheme] ?? 0);

      // Glow passes (additive blend)
      if (p.glow && p.fillMode !== 'filled') {
        ctx.enable(ctx.BLEND);
        ctx.blendFunc(ctx.SRC_ALPHA, ctx.ONE);
        ctx.bindVertexArray(outlineVAO!);
        ctx.uniform1f(uGs, 1.04);  ctx.uniform1f(uAlpha, 0.06);
        ctx.drawArrays(ctx.LINE_LOOP, 0, nVerts);
        ctx.uniform1f(uGs, 1.015); ctx.uniform1f(uAlpha, 0.12);
        ctx.drawArrays(ctx.LINE_LOOP, 0, nVerts);
        ctx.bindVertexArray(null);
        ctx.disable(ctx.BLEND);
      }

      ctx.enable(ctx.BLEND);
      ctx.blendFunc(ctx.SRC_ALPHA, ctx.ONE_MINUS_SRC_ALPHA);

      // Fill pass
      if (p.fillMode !== 'outline') {
        ctx.bindVertexArray(fillVAO!);
        ctx.uniform1f(uGs, 1.0);
        ctx.uniform1f(uAlpha, 0.22);
        ctx.drawArrays(ctx.TRIANGLE_FAN, 0, nVerts + 2);
        ctx.bindVertexArray(null);
      }

      // Outline pass
      if (p.fillMode !== 'filled') {
        ctx.bindVertexArray(outlineVAO!);
        ctx.uniform1f(uGs, 1.0);
        ctx.uniform1f(uAlpha, 0.92);
        ctx.drawArrays(ctx.LINE_LOOP, 0, nVerts);
        ctx.bindVertexArray(null);
      }

      ctx.disable(ctx.BLEND);
    },

    dispose() {
      if (fillVAO)    ctx.deleteVertexArray(fillVAO);
      if (outlineVAO) ctx.deleteVertexArray(outlineVAO);
      if (fillBuf)    ctx.deleteBuffer(fillBuf);
      if (outlineBuf) ctx.deleteBuffer(outlineBuf);
      ctx.deleteProgram(prog);
    },
  };
}
