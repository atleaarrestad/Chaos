/// WebGL2 Mandelbrot / Julia renderer using perturbation theory.
///
/// The view-center orbit is computed in JS (float64 or arbitrary-precision via decimal.js) and
/// uploaded to a RGBA32F texture of height 2 — "quad float32" (QF) format, 4 × float32 per
/// orbit value, giving ~96 bits / ~29 decimal digits of precision per component.
///
/// Mandelbrot: Z₀=0,  Zₙ₊₁ = Zₙ²+C_ref;  δₙ₊₁ = (2Zₙ+δₙ)·δₙ + ε,  δ₀=0
/// Julia:      Z₀=center, Zₙ₊₁ = Zₙ²+c;   δₙ₊₁ = (2Zₙ+δₙ)·δₙ,        δ₀=ε
/// ε = pixel offset from center in complex-plane units (tiny at high zoom).
///
/// Precision ceiling: float32 u_scale = 1/zoom underflows at zoom ~1e38; practical safe cap ~1e28
/// (quad-float orbit precision limit).  Above HP_THRESHOLD (1e13) decimal.js is used for the
/// orbit and the centre is passed as a quad-float uniform.

import Decimal from 'decimal.js';
import { compileShader } from '@/lib/gpu/shader';
import { hpPrecision, decimalToQF, f64ToQF } from './hp';
import { PALETTES, type PaletteId } from './palettes';

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
  /** HP Decimal strings for centre — required when zoom > HP_THRESHOLD. */
  hpCenterX?:   string;
  hpCenterY?:   string;
  /**
   * When true, skip reference-orbit recomputation even if the centre changed.
   * Used during fast drag to avoid blocking the main thread with Decimal arithmetic.
   * The previous orbit (from a nearby reference point) is reused — the result is
   * slightly inaccurate but visually acceptable while panning.
   */
  skipOrbit?:   boolean;
}

export interface WebGLRenderer {
  render(params: GLRenderParams): void;
  dispose(): void;
}

/** Returns true if WebGL2 is available in this browser. */
export function detectWebGL(): boolean {
  try {
    const c = document.createElement('canvas');
    const gl = c.getContext('webgl2');
    if (!gl) return false;
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return true;
  } catch {
    return false;
  }
}

/**
 * Compute the reference orbit for perturbation theory.
 *   Mandelbrot:  z₀ = 0,           zₙ₊₁ = zₙ² + (cRe, cIm)
 *   Julia:       z₀ = (z0Re, z0Im),  zₙ₊₁ = zₙ² + (cRe, cIm)
 *
 * Returns a Float32Array of size maxIter×8 laid out as a maxIter×2 RGBA texture:
 *   Row 0, texel i: (re_a, re_b, im_a, im_b) — leading quad-float components
 *   Row 1, texel i: (re_c, re_d, im_c, im_d) — correction components
 * All four floats together give ~96 bits / ~29 decimal digits per component.
 *
 * When hpCRe/hpCIm are provided (zoom > HP_THRESHOLD) the orbit is computed with
 * decimal.js so precision matches the centre's HP string, otherwise float64 is used
 * (row 1 is then zero-filled since float32 DD already captures all float64 bits).
 */
function computeReferenceOrbit(
  z0Re: number, z0Im: number,
  cRe:  number, cIm:  number,
  maxIter: number,
  hpCRe?: string, hpCIm?: string,
): { data: Float32Array; orbitLen: number } {
  const data = new Float32Array(maxIter * 8); // height=2 texture
  let orbitLen = maxIter;

  if (hpCRe && hpCIm) {
    // ── High-precision Decimal orbit ─────────────────────────────────────────
    const prec = hpPrecision(1 / Math.abs(cRe || cIm || 1e-30));  // rough proxy
    Decimal.set({ precision: Math.max(prec, hpCRe.length + 10) });
    const DC = new Decimal(hpCRe), DIm = new Decimal(hpCIm);
    let re = new Decimal(z0Re), im = new Decimal(z0Im);
    for (let n = 0; n < maxIter; n++) {
      const [ra, rb, rc, rd] = decimalToQF(re);
      const [ia, ib, ic, id] = decimalToQF(im);
      data[n * 4]               = ra;  data[n * 4 + 1]               = rb;
      data[n * 4 + 2]           = ia;  data[n * 4 + 3]               = ib;
      data[(maxIter + n) * 4]   = rc;  data[(maxIter + n) * 4 + 1]   = rd;
      data[(maxIter + n) * 4 + 2] = ic; data[(maxIter + n) * 4 + 3]  = id;
      if (re.mul(re).plus(im.mul(im)).gt(4)) { orbitLen = n; break; }
      const ni = re.mul(im).mul(2).plus(DIm);
      re = re.mul(re).minus(im.mul(im)).plus(DC);
      im = ni;
    }
  } else {
    // ── Float64 orbit (low/medium zoom) ──────────────────────────────────────
    let re = z0Re, im = z0Im;
    for (let n = 0; n < maxIter; n++) {
      const [ra, rb] = f64ToQF(re);  // c=d=0 — float32 DD captures full float64
      const [ia, ib] = f64ToQF(im);
      data[n * 4] = ra;  data[n * 4 + 1] = rb;
      data[n * 4 + 2] = ia;  data[n * 4 + 3] = ib;
      // Row 1 stays zero (already initialised to 0 by new Float32Array)
      if (re * re + im * im > 4) { orbitLen = n; break; }
      const newIm = 2 * re * im + cIm;
      re = re * re - im * im + cRe;
      im = newIm;
    }
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
uniform float u_scale;        // 1.0 / zoom (float32 — safe for zoom ≤ 1e37)
uniform int   u_maxIter;
uniform float u_colorSpeed;
uniform float u_colorOffset;
uniform bool  u_invertColors;
uniform vec3  u_palette[16];
uniform bool  u_juliaMode;
uniform int   u_orbitLen;   // number of valid orbit entries (≤ u_maxIter)
// Centre as quad float32 (4 × float32, ~96 bits) for deep-zoom precision.
uniform vec4  u_centerQFRe; // (a, b, c, d) where a+b+c+d ≈ centre.re
uniform vec4  u_centerQFIm;
// Pan delta between view centre and the orbit reference centre (complex coords, float32).
// Non-zero only when the orbit is reused from a previous position (skipOrbit drag at HP zoom).
// ε_orbit = eps + u_orbitDelta keeps perturbation correct while the orbit is stale.
uniform vec2  u_orbitDelta;
uniform vec2  u_juliaC;     // Julia constant (float32 fine — not zoom-dependent)
// Reference orbit texture: RGBA32F, height=2, width=maxIter.
//   Row 0, texel i: (re_a, re_b, im_a, im_b)
//   Row 1, texel i: (re_c, re_d, im_c, im_d)
// Together: re ≈ re_a + re_b + re_c + re_d  (~96-bit quad float32)
uniform sampler2D u_orbitTex;

out vec4 fragColor;

// Complex multiply (float32)
vec2 cMul(vec2 a, vec2 b) {
  return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Add float32 b to quad-float qf, returning the best float32 sum.
// Uses compensated addition on the two leading terms, then folds in corrections.
float qfPlusF(vec4 qf, float b) {
  float s = qf.x + b;
  float e = b - (s - qf.x);          // exact error from adding b into qf.x
  return s + (qf.y + e) + qf.z + qf.w;
}

void main() {
  // gl_FragCoord.y=0 is bottom in WebGL; flip to match canvas-2D (top=0) convention.
  float pixDx =  (gl_FragCoord.x - u_resolution.x * 0.5);
  float pixDy = -(gl_FragCoord.y - u_resolution.y * 0.5);

  // Pixel offset from view center in complex-plane units.
  // At high zoom this is tiny; float32 carries it fine up to zoom ~1e37.
  vec2 eps = vec2(pixDx, pixDy) * u_scale;

  int   iter    = 0;
  float r2      = 0.0;
  bool  escaped = false;

  // Perturbation δ:
  //   Mandelbrot — δ₀ = 0  (every Mandelbrot pixel starts z=0; c differs by ε)
  //   Julia      — δ₀ = ε  (pixel's offset from the reference start point)
  vec2 d = u_juliaMode ? eps : vec2(0.0);

  // Declared outside the loop so the fallback can read the last valid orbit entry.
  vec4 Zre = vec4(0.0);
  vec4 Zim = vec4(0.0);

  for (int i = 0; i < 2000; i++) {
    if (i >= u_maxIter) break;

    // Read quad-float orbit entry i from the height-2 texture.
    vec4 Zt0 = texelFetch(u_orbitTex, ivec2(i, 0), 0);
    vec4 Zt1 = texelFetch(u_orbitTex, ivec2(i, 1), 0);
    Zre = vec4(Zt0.rg, Zt1.rg);   // (re_a, re_b, re_c, re_d)
    Zim = vec4(Zt0.ba, Zt1.ba);   // (im_a, im_b, im_c, im_d)

    // Full iterate z = Z_n + δ_n using quad-float for precision at deep zoom.
    float zre = qfPlusF(Zre, d.x);
    float zim = qfPlusF(Zim, d.y);
    r2 = zre * zre + zim * zim;
    if (r2 > 4.0) { escaped = true; break; }

    // Reference orbit exhausted — pixel didn't escape at this boundary step.
    if (i >= u_orbitLen) break;

    // Perturbation recurrence uses only Z leading-float (δ is tiny so error×δ ≈ 0).
    // ε must be the pixel's offset from the orbit reference centre, not the view centre.
    // u_orbitDelta = (C_view − C_orbit) bridges the two when the orbit is reused during drag.
    vec2 Z = vec2(Zre.x, Zim.x);
    d = cMul(2.0 * Z + d, d);
    if (!u_juliaMode) d += eps + u_orbitDelta;

    iter++;
  }

  // Fallback: reference orbit shorter than maxIter (centre outside Mandelbrot set).
  // Continue from the accurate z computed by perturbation using direct float32 iteration.
  // c_pixel is computed via quad-float so eps is never swallowed at deep zoom.
  if (!escaped) {
    float fzRe = qfPlusF(Zre, d.x);
    float fzIm = qfPlusF(Zim, d.y);
    float fcRe, fcIm;
    if (u_juliaMode) {
      fcRe = u_juliaC.x;
      fcIm = u_juliaC.y;
    } else {
      fcRe = qfPlusF(u_centerQFRe, eps.x);
      fcIm = qfPlusF(u_centerQFIm, eps.y);
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
  int   lo = clamp(int(t), 0, N - 1);
  int   hi = (lo + 1) % N;
  float f  = fract(t);

  vec3 colA = u_invertColors ? u_palette[N - 1 - lo] : u_palette[lo];
  vec3 colB = u_invertColors ? u_palette[N - 1 - hi] : u_palette[hi];
  fragColor = vec4(mix(colA, colB, f), 1.0);
}
`;

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

  // Pre-allocate reference orbit texture:
  // RGBA32F, width = MAX_ORBIT, height = 2 (quad-float: row0 = hi pair, row1 = lo pair).
  const MAX_ORBIT = 2000;
  const orbitTex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, orbitTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, MAX_ORBIT, 2, 0, gl.RGBA, gl.FLOAT, null);
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
    centerQFRe:   gl.getUniformLocation(prog, 'u_centerQFRe')!,
    centerQFIm:   gl.getUniformLocation(prog, 'u_centerQFIm')!,
    orbitDelta:   gl.getUniformLocation(prog, 'u_orbitDelta')!,
    juliaC:       gl.getUniformLocation(prog, 'u_juliaC')!,
    orbitTex:     gl.getUniformLocation(prog, 'u_orbitTex')!,
  };

  const palCache = new Float32Array(16 * 3);
  // Track last orbit params to avoid redundant recomputes.
  let lastOrbitCX = NaN, lastOrbitCY = NaN, lastOrbitMaxIter = -1;
  let lastOrbitJRe = NaN, lastOrbitJIm = NaN, lastOrbitMode = false;
  let lastOrbitHPCX: string | null = null, lastOrbitHPCY: string | null = null;
  let lastOrbitLen = 0;

  function render(p: GLRenderParams): void {
    gl.viewport(0, 0, p.canvasW, p.canvasH);

    // Determine HP orbit key: if HP strings provided, use them; otherwise use float64 pair.
    const hpX = p.hpCenterX ?? null;
    const hpY = p.hpCenterY ?? null;
    const orbitKeyChanged =
      (hpX !== null
        ? (hpX !== lastOrbitHPCX || hpY !== lastOrbitHPCY)
        : (p.centerX !== lastOrbitCX || p.centerY !== lastOrbitCY)
      ) ||
      p.maxIter  !== lastOrbitMaxIter ||
      p.juliaMode !== lastOrbitMode   ||
      p.juliaRe  !== lastOrbitJRe     || p.juliaIm !== lastOrbitJIm;

    if (orbitKeyChanged && !p.skipOrbit) {
      // Mandelbrot: reference z₀=0, c = view centre.
      // Julia:      reference z₀ = view centre, c = juliaC (HP not needed for Julia c).
      const { data: orbitData, orbitLen } = p.juliaMode
        ? computeReferenceOrbit(p.centerX, p.centerY, p.juliaRe, p.juliaIm, p.maxIter)
        : computeReferenceOrbit(0, 0, p.centerX, p.centerY, p.maxIter,
            hpX ?? undefined, hpY ?? undefined);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, orbitTex);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, p.maxIter, 2, gl.RGBA, gl.FLOAT, orbitData);
      lastOrbitLen       = orbitLen;
      lastOrbitCX        = p.centerX;
      lastOrbitCY        = p.centerY;
      lastOrbitHPCX      = hpX;
      lastOrbitHPCY      = hpY;
      lastOrbitMaxIter   = p.maxIter;
      lastOrbitMode      = p.juliaMode;
      lastOrbitJRe       = p.juliaRe;
      lastOrbitJIm       = p.juliaIm;
    }

    gl.uniform2f(U.resolution,   p.canvasW, p.canvasH);
    gl.uniform1f(U.scale,        1.0 / p.zoom);
    gl.uniform1i(U.maxIter,      p.maxIter);
    gl.uniform1i(U.orbitLen,     lastOrbitLen);
    // Centre as quad-float32: provide HP strings when zoom > HP_THRESHOLD, else f64 split.
    const qfRe = hpX ? decimalToQF(new Decimal(hpX)) : f64ToQF(p.centerX);
    const qfIm = hpY ? decimalToQF(new Decimal(hpY)) : f64ToQF(p.centerY);
    gl.uniform4f(U.centerQFRe,   qfRe[0], qfRe[1], qfRe[2], qfRe[3]);
    gl.uniform4f(U.centerQFIm,   qfIm[0], qfIm[1], qfIm[2], qfIm[3]);
    // Orbit delta: C_view − C_orbit_reference in complex coordinates.
    // Non-zero only during skipOrbit drag at HP zoom; keeps perturbation epsilon correct
    // when the orbit was computed at a different reference point than the current view centre.
    // Use HP subtraction when available (Decimal strings both present) to preserve digits.
    let orbitDeltaRe = 0, orbitDeltaIm = 0;
    if (hpX && lastOrbitHPCX && hpX !== lastOrbitHPCX) {
      orbitDeltaRe = new Decimal(hpX).minus(new Decimal(lastOrbitHPCX)).toNumber();
      orbitDeltaIm = new Decimal(hpY!).minus(new Decimal(lastOrbitHPCY!)).toNumber();
    } else if (!hpX && !isNaN(lastOrbitCX)) {
      orbitDeltaRe = p.centerX - lastOrbitCX;
      orbitDeltaIm = p.centerY - lastOrbitCY;
    }
    gl.uniform2f(U.orbitDelta,   orbitDeltaRe, orbitDeltaIm);
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
