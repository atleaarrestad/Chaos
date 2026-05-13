/// WebGL2 Mandelbrot / Julia renderer using perturbation theory.
///
/// The view-center orbit is computed in JS float64 and uploaded as an RG32F texture.
/// Each pixel iterates only its tiny perturbation δ (plain float32 — δ stays small by design).
///
/// Mandelbrot: Z₀=0,  Zₙ₊₁ = Zₙ²+C_ref;  δₙ₊₁ = (2Zₙ+δₙ)·δₙ + ε,  δ₀=0
/// Julia:      Z₀=center, Zₙ₊₁ = Zₙ²+c;   δₙ₊₁ = (2Zₙ+δₙ)·δₙ,        δ₀=ε
/// ε = pixel offset from center in complex-plane units (tiny at high zoom).
///
/// Precision ceiling: JS float64 center coordinate precision (~1e-15), well beyond zoom 1e13.
///
/// Precision fixes applied:
///   1. Fallback corrected — centre passed as double-double so c_pixel = centre+eps is precise
///      at any zoom; fallback continues from the accurate z computed by perturbation rather
///      than restarting from z=0 with a float32-truncated centre.
///   2. Reference boundary: the loop now reads Z[orbitLen] (the first escaped reference point)
///      before stopping, catching pixels that should escape at that exact step.

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

/**
 * Compute the reference orbit for perturbation theory.
 *   Mandelbrot:  z₀ = 0,           zₙ₊₁ = zₙ² + (cRe, cIm)
 *   Julia:       z₀ = (z0Re, z0Im),  zₙ₊₁ = zₙ² + (cRe, cIm)
 *
 * Returns Float32Array of size maxIter×4: [re_hi, re_lo, im_hi, im_lo, …].
 * Each value is a double-double split: hi = float32(x), lo = float32(x − hi).
 * Computed in native float64, stored as float32 DD pairs (~48-bit precision).
 * orbitLen: number of valid entries (< maxIter when the reference orbit escapes early).
 * Remaining entries beyond orbitLen are zero-padded but must NOT be read by the shader.
 */
function computeReferenceOrbit(
  z0Re: number, z0Im: number,
  cRe: number,  cIm: number,
  maxIter: number,
): { data: Float32Array; orbitLen: number } {
  const data = new Float32Array(maxIter * 4);
  let re = z0Re, im = z0Im;
  let orbitLen = maxIter;
  for (let n = 0; n < maxIter; n++) {
    const reHi = Math.fround(re);
    const reLo = Math.fround(re - reHi);
    const imHi = Math.fround(im);
    const imLo = Math.fround(im - imHi);
    data[n * 4]     = reHi;
    data[n * 4 + 1] = reLo;
    data[n * 4 + 2] = imHi;
    data[n * 4 + 3] = imLo;
    // Stop at the standard Mandelbrot escape radius — post-escape values are garbage.
    if (re * re + im * im > 4) { orbitLen = n; break; }
    const newIm = 2 * re * im + cIm;
    re = re * re - im * im + cRe;
    im = newIm;
  }
  return { data, orbitLen };
}

// ── Shaders ──────────────────────────────────────────────────────────────────

const VERT_SRC = `#version 300 es
in vec2 a_pos;
void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }
`;

const FRAG_SRC = `#version 300 es
precision highp float;
precision highp int;

// ── Uniforms ──────────────────────────────────────────────────────────────────
uniform vec2  u_resolution;
uniform float u_scale;        // 1.0 / zoom (plain float — pixel offsets are tiny)
uniform int   u_maxIter;
uniform float u_colorSpeed;
uniform float u_colorOffset;
uniform bool  u_invertColors;
uniform vec3  u_palette[16];
uniform bool  u_juliaMode;
uniform int   u_orbitLen;   // number of valid orbit entries (≤ u_maxIter)
// Centre as double-double so c_pixel = centre + eps keeps full precision at deep zoom.
uniform vec2  u_centerDDRe; // (re_hi, re_lo)
uniform vec2  u_centerDDIm; // (im_hi, im_lo)
uniform vec2  u_juliaC;     // Julia constant (float32 is fine — not zoom-dependent)
// Reference orbit texture: RGBA32F, one texel per step = (re_hi, re_lo, im_hi, im_lo) of Z_n
uniform sampler2D u_orbitTex;

out vec4 fragColor;

// Complex multiply (float32)
vec2 cMul(vec2 a, vec2 b) {
  return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Add a float32 value b into a double-double (hi, lo), returning the improved float32 sum.
// Gives ~48-bit precision for the escape check even when b << hi.
float ddPlusF(vec2 dd, float b) {
  float s = dd.x + b;
  float e = b - (s - dd.x);
  return s + (dd.y + e);
}

void main() {
  // gl_FragCoord.y=0 is bottom in WebGL; flip to match canvas-2D (top=0) convention.
  float pixDx =  (gl_FragCoord.x - u_resolution.x * 0.5);
  float pixDy = -(gl_FragCoord.y - u_resolution.y * 0.5);

  // Pixel offset from view center in complex-plane units.
  // At high zoom this is tiny, so plain float32 carries it without loss.
  vec2 eps = vec2(pixDx, pixDy) * u_scale;

  int   iter    = 0;
  float r2      = 0.0;
  bool  escaped = false;

  // Perturbation δ:
  //   Mandelbrot — δ₀ = 0  (every Mandelbrot pixel starts z=0; c differs by ε)
  //   Julia      — δ₀ = ε  (pixel's offset from the reference start point)
  vec2 d = u_juliaMode ? eps : vec2(0.0);

  // Declared outside the loop so the fallback can read the last valid orbit entry.
  vec2 Zre = vec2(0.0);
  vec2 Zim = vec2(0.0);

  for (int i = 0; i < 2000; i++) {
    if (i >= u_maxIter) break;

    // Read orbit entry i.  When i == u_orbitLen the reference has just escaped;
    // we read that escaped point once to give pixels a chance to escape at the
    // same step, then stop regardless of the outcome.
    vec4 Ztex = texelFetch(u_orbitTex, ivec2(i, 0), 0);
    Zre = Ztex.rg;
    Zim = Ztex.ba;

    // Full iterate z = Z_n + δ_n using DD for precision at deep zoom.
    float zre = ddPlusF(Zre, d.x);
    float zim = ddPlusF(Zim, d.y);
    r2 = zre * zre + zim * zim;
    if (r2 > 4.0) { escaped = true; break; }

    // Reference orbit exhausted — pixel didn't escape at this boundary step.
    if (i >= u_orbitLen) break;

    // Perturbation recurrence uses only Z hi-parts (δ is tiny so error × δ ≈ 0).
    vec2 Z = vec2(Zre.x, Zim.x);
    d = cMul(2.0 * Z + d, d);
    if (!u_juliaMode) d += eps;

    iter++;
  }

  // Fallback: perturbation ended early (reference orbit shorter than maxIter).
  // Continue from the accurate z already computed by perturbation using direct
  // float32 iteration.  At low/medium zoom float32 is sufficient for the tail;
  // at deep zoom the perturbation already handled the precision-critical steps,
  // so only a short float32 tail remains.
  // c_pixel is computed via DD so eps is never swallowed at deep zoom.
  if (!escaped) {
    float fzRe = ddPlusF(Zre, d.x);
    float fzIm = ddPlusF(Zim, d.y);
    float fcRe, fcIm;
    if (u_juliaMode) {
      fcRe = u_juliaC.x;
      fcIm = u_juliaC.y;
    } else {
      fcRe = ddPlusF(u_centerDDRe, eps.x);
      fcIm = ddPlusF(u_centerDDIm, eps.y);
    }
    for (int i = iter; i < 2000; i++) {
      if (i >= u_maxIter) break;
      r2 = fzRe * fzRe + fzIm * fzIm;
      if (r2 > 4.0) { escaped = true; iter = i; break; }
      float newRe = fzRe * fzRe - fzIm * fzIm + fcRe;
      fzIm = 2.0 * fzRe * fzIm + fcIm;
      fzRe = newRe;
    }
  }

  if (!escaped) {
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  // Smooth colouring — identical formula to CPU worker.
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

  // Pre-allocate reference orbit texture (MAX_ORBIT×1, RGBA32F = re_hi,re_lo,im_hi,im_lo per texel).
  const MAX_ORBIT = 2000;
  const orbitTex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, orbitTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, MAX_ORBIT, 1, 0, gl.RGBA, gl.FLOAT, null);
  gl.bindTexture(gl.TEXTURE_2D, null);

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
    scale:        gl.getUniformLocation(prog, 'u_scale')!,
    maxIter:      gl.getUniformLocation(prog, 'u_maxIter')!,
    colorSpeed:   gl.getUniformLocation(prog, 'u_colorSpeed')!,
    colorOffset:  gl.getUniformLocation(prog, 'u_colorOffset')!,
    invertColors: gl.getUniformLocation(prog, 'u_invertColors')!,
    palette:      gl.getUniformLocation(prog, 'u_palette[0]')!,
    juliaMode:    gl.getUniformLocation(prog, 'u_juliaMode')!,
    orbitLen:     gl.getUniformLocation(prog, 'u_orbitLen')!,
    centerDDRe:   gl.getUniformLocation(prog, 'u_centerDDRe')!,
    centerDDIm:   gl.getUniformLocation(prog, 'u_centerDDIm')!,
    juliaC:       gl.getUniformLocation(prog, 'u_juliaC')!,
    orbitTex:     gl.getUniformLocation(prog, 'u_orbitTex')!,
  };

  const palCache = new Float32Array(16 * 3);
  // Track last orbit params to avoid redundant recomputes.
  let lastOrbitCX = NaN, lastOrbitCY = NaN, lastOrbitMaxIter = -1;
  let lastOrbitJRe = NaN, lastOrbitJIm = NaN, lastOrbitMode = false;
  let lastOrbitLen = 0;

  function render(p: GLRenderParams): void {
    gl.viewport(0, 0, p.canvasW, p.canvasH);

    // Recompute reference orbit whenever view center, Julia params, or iteration depth changes.
    if (
      p.centerX  !== lastOrbitCX   || p.centerY  !== lastOrbitCY   ||
      p.maxIter  !== lastOrbitMaxIter ||
      p.juliaMode !== lastOrbitMode ||
      p.juliaRe  !== lastOrbitJRe  || p.juliaIm  !== lastOrbitJIm
    ) {
      // Mandelbrot: reference starts at z=0, iterates with c = view center.
      // Julia:      reference starts at view center, iterates with c = juliaC.
      const { data: orbitData, orbitLen } = p.juliaMode
        ? computeReferenceOrbit(p.centerX, p.centerY, p.juliaRe, p.juliaIm, p.maxIter)
        : computeReferenceOrbit(0, 0, p.centerX, p.centerY, p.maxIter);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, orbitTex);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, p.maxIter, 1, gl.RGBA, gl.FLOAT, orbitData);
      lastOrbitLen       = orbitLen;
      lastOrbitCX        = p.centerX;
      lastOrbitCY        = p.centerY;
      lastOrbitMaxIter   = p.maxIter;
      lastOrbitMode      = p.juliaMode;
      lastOrbitJRe       = p.juliaRe;
      lastOrbitJIm       = p.juliaIm;
    }

    gl.uniform2f(U.resolution,   p.canvasW, p.canvasH);
    gl.uniform1f(U.scale,        1.0 / p.zoom);
    gl.uniform1i(U.maxIter,      p.maxIter);
    gl.uniform1i(U.orbitLen,     lastOrbitLen);
    // Centre as double-double: hi = float32(centre), lo = float32(centre − hi).
    gl.uniform2f(U.centerDDRe,   Math.fround(p.centerX), Math.fround(p.centerX - Math.fround(p.centerX)));
    gl.uniform2f(U.centerDDIm,   Math.fround(p.centerY), Math.fround(p.centerY - Math.fround(p.centerY)));
    gl.uniform2f(U.juliaC,       p.juliaRe, p.juliaIm);
    gl.uniform1f(U.colorSpeed,   p.colorSpeed);
    gl.uniform1f(U.colorOffset,  p.colorOffset);
    gl.uniform1i(U.invertColors, p.invertColors ? 1 : 0);
    gl.uniform1i(U.juliaMode,    p.juliaMode    ? 1 : 0);

    const pal = PALETTES[p.paletteId];
    for (let i = 0; i < 16; i++) {
      palCache[i * 3]     = pal[i][0] / 255;
      palCache[i * 3 + 1] = pal[i][1] / 255;
      palCache[i * 3 + 2] = pal[i][2] / 255;
    }
    gl.uniform3fv(U.palette, palCache);

    // Bind orbit texture to unit 0.
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, orbitTex);
    gl.uniform1i(U.orbitTex, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  function dispose(): void {
    gl.deleteProgram(prog);
    gl.deleteShader(vert);
    gl.deleteShader(frag);
    gl.deleteBuffer(vbo);
    gl.deleteTexture(orbitTex);
  }

  return { render, dispose };
}
