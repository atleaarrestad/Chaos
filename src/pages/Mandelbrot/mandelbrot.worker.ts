/// <reference lib="webworker" />

/**
 * Classic 16-stop "electric" palette — cycles smoothly through
 * deep indigo → royal blue → pale cyan → gold → dark brown → back.
 */
const PALETTE: readonly [number, number, number][] = [
  [  9,   1,  47],
  [  4,   4,  73],
  [  0,   7, 100],
  [ 12,  44, 138],
  [ 24,  82, 177],
  [ 57, 125, 209],
  [134, 181, 229],
  [211, 236, 248],
  [241, 233, 191],
  [248, 201,  95],
  [255, 170,   0],
  [204, 128,   0],
  [153,  87,   0],
  [106,  52,   3],
  [ 66,  30,  15],
  [ 25,   7,  26],
] as const;

const N      = PALETTE.length;
const LN2    = Math.LN2;
const SPEED  = 0.28; // palette steps per iteration

interface RenderMsg {
  width:   number;
  height:  number;
  centerX: number;
  centerY: number;
  zoom:    number;
  maxIter: number;
  id:      number;
}

self.onmessage = (e: MessageEvent<RenderMsg>) => {
  const { width, height, centerX, centerY, zoom, maxIter, id } = e.data;

  const buf   = new Uint8ClampedArray(width * height * 4);
  const scale = 1 / zoom;
  const halfW = width  * 0.5;
  const halfH = height * 0.5;

  let i = 0;
  for (let py = 0; py < height; py++) {
    const c_im = centerY + (py - halfH) * scale;
    for (let px = 0; px < width; px++) {
      const c_re = centerX + (px - halfW) * scale;

      // Optimised iteration: cache x² and y² to avoid redundant multiplies
      let re = 0, im = 0, re2 = 0, im2 = 0, iter = 0;
      while (re2 + im2 <= 4 && iter < maxIter) {
        im  = 2 * re * im + c_im;
        re  = re2 - im2 + c_re;
        re2 = re * re;
        im2 = im * im;
        iter++;
      }

      if (iter === maxIter) {
        // Interior — black
        buf[i] = buf[i + 1] = buf[i + 2] = 0;
      } else {
        // Smooth escape time: removes discrete iteration bands
        const smooth = iter + 1 - Math.log(Math.log(re2 + im2) * 0.5) / LN2;

        // Map to cyclic palette with linear interpolation
        const t  = (smooth * SPEED) % N;
        const lo = Math.floor(t);
        const hi = (lo + 1) % N;
        const f  = t - lo;
        const nlo = (lo + N) % N;

        const a = PALETTE[nlo];
        const b = PALETTE[hi];
        buf[i]     = (a[0] + (b[0] - a[0]) * f) | 0;
        buf[i + 1] = (a[1] + (b[1] - a[1]) * f) | 0;
        buf[i + 2] = (a[2] + (b[2] - a[2]) * f) | 0;
      }
      buf[i + 3] = 255;
      i += 4;
    }
  }

  self.postMessage({ buf, id }, [buf.buffer] as unknown as Transferable[]);
};
