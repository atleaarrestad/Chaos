/// WebGL2 Mandelbrot / Julia renderer for the main thread.
/// Uses double-double arithmetic (float32 hi+lo pairs) for ~48-bit precision,
/// supporting clean rendering up to zoom ~1e14 — matching the CPU worker range.

import type { PaletteId } from './mandelbrot.worker';

type RGB = readonly [number, number, number];

// Palette data mirrored from mandelbrot.worker.ts (kept in sync manually).
const PALETTES: Record<PaletteId, readonly RGB[]> = {
  classic: [
    [  9,   1,  47], [  4,   4,  73], [  0,   7, 100], [ 12,  44, 138],
    [ 24,  82, 177], [ 57, 125, 209], [134, 181, 229], [211, 236, 248],
    [241, 233, 191], [248, 201,  95], [255, 170,   0], [204, 128,   0],
    [153,  87,   0], [106,  52,   3], [ 66,  30,  15], [ 25,   7,  26],
  ],
  fire: [
    [  2,   0,   4], [ 15,   0,  10], [ 40,   0,   5], [ 80,   5,   0],
    [130,  10,   0], [180,  30,   0], [220,  70,   0], [255, 120,   0],
    [255, 170,   0], [255, 210,  20], [255, 240,  80], [255, 255, 160],
    [255, 255, 220], [255, 250, 240], [255, 255, 255], [200, 220, 255],
  ],
  ice: [
    [  0,   0,  12], [  0,   8,  35], [  0,  20,  70], [  0,  50, 110],
    [  5,  90, 150], [ 15, 130, 180], [ 40, 165, 205], [ 80, 195, 225],
    [120, 220, 238], [165, 237, 248], [205, 248, 253], [230, 252, 255],
    [245, 255, 255], [255, 255, 255], [210, 235, 255], [170, 210, 255],
  ],
  electric: [
    [  4,   0,  18], [ 18,   0,  50], [ 48,   0,  98], [ 80,   0, 160],
    [120,   0, 200], [158,  25, 222], [180,  85, 242], [155, 145, 255],
    [ 70, 200, 255], [ 10, 245, 228], [  0, 255, 175], [ 25, 255,  95],
    [100, 255,  75], [205, 255,  98], [255, 238, 148], [255, 255, 218],
  ],
  mono: [
    [  8,   8,   8], [ 18,  18,  18], [ 34,  34,  34], [ 55,  55,  55],
    [ 80,  80,  80], [108, 108, 108], [138, 138, 138], [165, 165, 165],
    [188, 188, 188], [208, 208, 208], [224, 224, 224], [238, 238, 238],
    [248, 248, 248], [252, 252, 252], [235, 235, 235], [205, 205, 205],
  ],
  sunset: [
    [  5,   0,  15], [ 22,   0,  38], [ 55,   0,  58], [ 95,   5,  55],
    [145,  15,  48], [195,  35,  58], [235,  65,  78], [255, 105,  98],
    [255, 152, 118], [255, 193, 128], [255, 225, 140], [255, 242, 158],
    [255, 248, 192], [242, 232, 222], [198, 200, 232], [158, 168, 222],
  ],
};

export interface GLRenderParams {
  canvasW:      number;
  canvasH:      number;
  centerX:      number;
  centerY:      number;
  zoom:         number;
  maxIter:      number;
  paletteId:    PaletteId;
  colorSpeed:   number;
  colorOffset:  number;
  invertColors: boolean;
  juliaMode:    boolean;
  juliaRe:      number;
  juliaIm:      number;
}

export interface WebGLRenderer {
  render(params: GLRenderParams): void;
  dispose(): void;
}

/** Returns true if WebGL2 is available in this browser. */
export function detectWebGL(): boolean {
  try {
    return !!document.createElement('canvas').getContext('webgl2');
  } catch {
    return false;
  }
}

/** Split a JS float64 into a (hi, lo) float32 pair for double-double upload. */
function splitDD(x: number): [number, number] {
  const hi = Math.fround(x);
  return [hi, Math.fround(x - hi)];
}

// ── Shaders ──────────────────────────────────────────────────────────────────

const VERT_SRC = `#version 300 es
in vec2 a_pos;
void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }
`;

const FRAG_SRC = `#version 300 es
precision highp float;
precision highp int;

// ── Double-double arithmetic (Shewchuk / Dekker) ─────────────────────────────

vec2 twoSum(float a, float b) {
  float s = a + b;
  float v = s - a;
  return vec2(s, (a - (s - v)) + (b - v));
}

vec2 ddAdd(vec2 a, vec2 b) {
  vec2 s = twoSum(a.x, b.x);
  s.y += a.y + b.y;
  float v = s.x + s.y;
  return vec2(v, s.y - (v - s.x));
}

vec2 ddSub(vec2 a, vec2 b) { return ddAdd(a, vec2(-b.x, -b.y)); }

vec2 twoProd(float a, float b) {
  float p = a * b;
  const float C = 4097.0; // Veltkamp splitter: 2^12 + 1
  float ca = C * a; float ahi = ca - (ca - a); float alo = a - ahi;
  float cb = C * b; float bhi = cb - (cb - b); float blo = b - bhi;
  return vec2(p, ((ahi * bhi - p) + ahi * blo + alo * bhi) + alo * blo);
}

vec2 ddMul(vec2 a, vec2 b) {
  vec2 p = twoProd(a.x, b.x);
  p.y += a.x * b.y + a.y * b.x;
  float v = p.x + p.y;
  return vec2(v, p.y - (v - p.x));
}

vec2 ddSqr(vec2 a) {
  vec2 p = twoProd(a.x, a.x);
  p.y += 2.0 * a.x * a.y;
  float v = p.x + p.y;
  return vec2(v, p.y - (v - p.x));
}

// Multiply by 2 is exact in IEEE 754 (no rounding error).
vec2 ddScale2(vec2 a) { return vec2(a.x * 2.0, a.y * 2.0); }

// ── Uniforms ─────────────────────────────────────────────────────────────────

uniform vec2  u_resolution;
uniform vec2  u_centerX;      // double-double (hi, lo)
uniform vec2  u_centerY;      // double-double (hi, lo)
uniform float u_scale;        // 1.0 / zoom
uniform int   u_maxIter;
uniform float u_colorSpeed;
uniform float u_colorOffset;
uniform bool  u_invertColors;
uniform vec3  u_palette[16];
uniform bool  u_juliaMode;
uniform float u_juliaRe;
uniform float u_juliaIm;

out vec4 fragColor;

void main() {
  // Pixel → complex coordinate (double-double).
  // gl_FragCoord.y = 0 is bottom in WebGL; flip y to match canvas-2D convention.
  float dx =  (gl_FragCoord.x - u_resolution.x * 0.5) * u_scale;
  float dy = -(gl_FragCoord.y - u_resolution.y * 0.5) * u_scale;

  vec2 cre = ddAdd(u_centerX, vec2(dx, 0.0));
  vec2 cim = ddAdd(u_centerY, vec2(dy, 0.0));

  vec2 zre, zim, fre, fim;
  if (u_juliaMode) {
    zre = cre; zim = cim;
    fre = vec2(u_juliaRe, 0.0); fim = vec2(u_juliaIm, 0.0);
  } else {
    zre = vec2(0.0); zim = vec2(0.0);
    fre = cre; fim = cim;
  }

  // Iteration loop — mirrors the CPU worker exactly:
  //   while (re2 + im2 <= 4 && iter < maxIter) { update; re2=re²; im2=im²; iter++ }
  vec2 re2 = ddSqr(zre);
  vec2 im2 = ddSqr(zim);
  int iter = 0;
  for (int i = 0; i < 2000; i++) {
    if (i >= u_maxIter)          break; // interior: all iterations exhausted
    if (re2.x + im2.x > 4.0)    break; // escaped
    vec2 newIm = ddAdd(ddScale2(ddMul(zre, zim)), fim);
    zre = ddAdd(ddSub(re2, im2), fre);
    zim = newIm;
    re2 = ddSqr(zre);
    im2 = ddSqr(zim);
    iter++;
  }

  if (iter >= u_maxIter) {
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  // Smooth colouring — identical formula to CPU worker.
  float r2       = re2.x + im2.x;
  float smooth_v = float(iter) + 1.0 - log(log(r2) * 0.5) / log(2.0);

  const int N = 16;
  float t = mod(smooth_v * u_colorSpeed + u_colorOffset, float(N));
  int   lo = int(t);
  int   hi = (lo + 1) % N;
  float f  = fract(t);

  vec3 colA = u_invertColors ? u_palette[N - 1 - lo] : u_palette[lo];
  vec3 colB = u_invertColors ? u_palette[N - 1 - hi] : u_palette[hi];
  fragColor = vec4(mix(colA, colB, f), 1.0);
}
`;

// ── Setup helpers ─────────────────────────────────────────────────────────────

function compileShader(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error(`Shader compile error:\n${log}`);
  }
  return sh;
}

// ── Public API ────────────────────────────────────────────────────────────────

export function createWebGLRenderer(canvas: HTMLCanvasElement): WebGLRenderer | null {
  // preserveDrawingBuffer allows canvas.toDataURL() to work for PNG export.
  const glOrNull = canvas.getContext('webgl2', { preserveDrawingBuffer: true });
  if (!glOrNull) return null;
  const gl: WebGL2RenderingContext = glOrNull;

  const vert = compileShader(gl, gl.VERTEX_SHADER,   VERT_SRC);
  const frag = compileShader(gl, gl.FRAGMENT_SHADER, FRAG_SRC);

  const prog = gl.createProgram()!;
  gl.attachShader(prog, vert);
  gl.attachShader(prog, frag);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error(`Program link error:\n${gl.getProgramInfoLog(prog)}`);
  }
  gl.useProgram(prog);

  // Full-screen quad covering NDC [-1, 1]²
  const vbo = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1,  1, -1, -1,  1,
    -1,  1,  1, -1,  1,  1,
  ]), gl.STATIC_DRAW);
  const aPos = gl.getAttribLocation(prog, 'a_pos');
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

  // Cache uniform locations once at setup time.
  const U = {
    resolution:   gl.getUniformLocation(prog, 'u_resolution')!,
    centerX:      gl.getUniformLocation(prog, 'u_centerX')!,
    centerY:      gl.getUniformLocation(prog, 'u_centerY')!,
    scale:        gl.getUniformLocation(prog, 'u_scale')!,
    maxIter:      gl.getUniformLocation(prog, 'u_maxIter')!,
    colorSpeed:   gl.getUniformLocation(prog, 'u_colorSpeed')!,
    colorOffset:  gl.getUniformLocation(prog, 'u_colorOffset')!,
    invertColors: gl.getUniformLocation(prog, 'u_invertColors')!,
    palette:      gl.getUniformLocation(prog, 'u_palette[0]')!,
    juliaMode:    gl.getUniformLocation(prog, 'u_juliaMode')!,
    juliaRe:      gl.getUniformLocation(prog, 'u_juliaRe')!,
    juliaIm:      gl.getUniformLocation(prog, 'u_juliaIm')!,
  };

  const palCache = new Float32Array(16 * 3);

  function render(p: GLRenderParams): void {
    gl.viewport(0, 0, p.canvasW, p.canvasH);

    const [cxHi, cxLo] = splitDD(p.centerX);
    const [cyHi, cyLo] = splitDD(p.centerY);

    gl.uniform2f(U.resolution,   p.canvasW, p.canvasH);
    gl.uniform2f(U.centerX,      cxHi, cxLo);
    gl.uniform2f(U.centerY,      cyHi, cyLo);
    gl.uniform1f(U.scale,        1.0 / p.zoom);
    gl.uniform1i(U.maxIter,      p.maxIter);
    gl.uniform1f(U.colorSpeed,   p.colorSpeed);
    gl.uniform1f(U.colorOffset,  p.colorOffset);
    gl.uniform1i(U.invertColors, p.invertColors ? 1 : 0);
    gl.uniform1i(U.juliaMode,    p.juliaMode    ? 1 : 0);
    gl.uniform1f(U.juliaRe,      p.juliaRe);
    gl.uniform1f(U.juliaIm,      p.juliaIm);

    const pal = PALETTES[p.paletteId];
    for (let i = 0; i < 16; i++) {
      palCache[i * 3]     = pal[i][0] / 255;
      palCache[i * 3 + 1] = pal[i][1] / 255;
      palCache[i * 3 + 2] = pal[i][2] / 255;
    }
    gl.uniform3fv(U.palette, palCache);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  function dispose(): void {
    gl.deleteProgram(prog);
    gl.deleteShader(vert);
    gl.deleteShader(frag);
    gl.deleteBuffer(vbo);
  }

  return { render, dispose };
}
