/**
 * High-precision arithmetic utilities for deep Mandelbrot zoom.
 *
 * At zoom > HP_THRESHOLD the view centre coordinate needs more than float64's ~15 significant
 * digits.  We store it as a Decimal string and use decimal.js for all centre arithmetic.
 *
 * Orbit precision is also upgraded: instead of float32 DD (2 floats, ~14 digits), the reference
 * orbit is stored in "quad float32" format (4 floats, ~29 digits), enabling correct rendering
 * up to zoom ~1e28 — well within the float32 scale-underflow ceiling (~1e38).
 */

import Decimal from 'decimal.js';

// ── Constants ─────────────────────────────────────────────────────────────────

/** Zoom level above which HP arithmetic is used for centre + orbit computation. */
export const HP_THRESHOLD = 1e13;

/** Returns the number of significant decimal digits needed at a given zoom. */
export function hpPrecision(zoom: number): number {
  return Math.max(35, Math.ceil(Math.log10(zoom)) + 20);
}

// ── Centre type ───────────────────────────────────────────────────────────────

export interface HPCentre { re: string; im: string }

/** Wrap float64 coordinates into HP strings (adequate up to zoom ~1e14). */
export function f64ToHPCentre(x: number, y: number): HPCentre {
  return {
    re: new Decimal(x).toSignificantDigits(20).toString(),
    im: new Decimal(y).toSignificantDigits(20).toString(),
  };
}

// ── Arithmetic ────────────────────────────────────────────────────────────────

/**
 * Pan centre by (dx, dy) canvas pixels at the given zoom.
 * new_centre = old_centre − (dx, dy) / zoom
 */
export function hpPan(cx: string, cy: string, dx: number, dy: number, zoom: number): HPCentre {
  const prec = hpPrecision(zoom);
  const D = Decimal.clone({ precision: prec });
  const scale = new D(1).div(new D(zoom));
  return {
    re: new D(cx).minus(new D(dx).mul(scale)).toSignificantDigits(prec).toString(),
    im: new D(cy).minus(new D(dy).mul(scale)).toSignificantDigits(prec).toString(),
  };
}

/**
 * Compute new HP centre after zooming by factor (newZoom/oldZoom), keeping the canvas point
 * (pixDx, pixDy) pixels from the canvas-centre fixed in complex space.
 */
export function hpZoomTo(
  cx: string, cy: string,
  pixDx: number, pixDy: number,
  oldZoom: number, newZoom: number,
): HPCentre {
  const prec = hpPrecision(Math.max(oldZoom, newZoom));
  const D = Decimal.clone({ precision: prec });
  const oldScale = new D(1).div(new D(oldZoom));
  const newScale = new D(1).div(new D(newZoom));
  // Mouse position in complex plane: centre + pixel_offset * (1/oldZoom)
  const mouseRe = new D(cx).plus(new D(pixDx).mul(oldScale));
  const mouseIm = new D(cy).plus(new D(pixDy).mul(oldScale));
  // New centre: mouseComplex − pixel_offset * (1/newZoom)
  return {
    re: mouseRe.minus(new D(pixDx).mul(newScale)).toSignificantDigits(prec).toString(),
    im: mouseIm.minus(new D(pixDy).mul(newScale)).toSignificantDigits(prec).toString(),
  };
}

// ── Quad float32 splitting ────────────────────────────────────────────────────

/**
 * Split a Decimal value into 4 float32s (a, b, c, d) such that a+b+c+d ≈ D
 * with ~96 bits of combined precision ("quad float32").
 *
 * Components are ordered largest-first.  The GPU sums them with qfPlusF().
 */
export function decimalToQF(D: Decimal): [number, number, number, number] {
  const a  = Math.fround(D.toNumber());
  const r1 = D.minus(new Decimal(a));
  const b  = Math.fround(r1.toNumber());
  const r2 = r1.minus(new Decimal(b));
  const c  = Math.fround(r2.toNumber());
  const r3 = r2.minus(new Decimal(c));
  const d  = Math.fround(r3.toNumber());
  return [a, b, c, d];
}

/**
 * Split a float64 into quad float32.
 * The float32 DD (a + b) already recovers the full float64, so c = d = 0.
 */
export function f64ToQF(x: number): [number, number, number, number] {
  const a = Math.fround(x);
  const b = Math.fround(x - a);
  return [a, b, 0, 0];
}
