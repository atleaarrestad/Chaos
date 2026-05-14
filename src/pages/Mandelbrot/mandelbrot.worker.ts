/// <reference lib="webworker" />

import { PALETTES, type PaletteId, type RGB } from './palettes';
export type { PaletteId } from './palettes';

const LN2 = Math.LN2;

interface Tile { x: number; y: number; w: number; h: number; }

interface RenderMsg {
  tileList:     Tile[];
  canvasW:      number;
  canvasH:      number;
  centerX:      number;
  centerY:      number;
  zoom:         number;
  maxIter:      number;
  id:           number;
  paletteId:    PaletteId;
  colorSpeed:   number;
  colorOffset:  number;
  invertColors: boolean;
  juliaMode:    boolean;
  juliaRe:      number;
  juliaIm:      number;
}

self.onmessage = (e: MessageEvent<RenderMsg>) => {
  const {
    tileList, canvasW, canvasH, centerX, centerY, zoom, maxIter, id,
    paletteId, colorSpeed, colorOffset, invertColors,
    juliaMode, juliaRe, juliaIm,
  } = e.data;

  const raw = PALETTES[paletteId] ?? PALETTES.classic;
  const palette: readonly RGB[] = invertColors ? [...raw].reverse() : raw;
  const N = palette.length;

  const scale = 1 / zoom;
  const halfW = canvasW * 0.5;
  const halfH = canvasH * 0.5;

  for (const { x, y, w, h } of tileList) {
    const buf = new Uint8ClampedArray(w * h * 4);
    let i = 0;

    for (let py = y; py < y + h; py++) {
      const c_im = centerY + (py - halfH) * scale;
      for (let px = x; px < x + w; px++) {
        const c_re = centerX + (px - halfW) * scale;

        // Mandelbrot: z₀ = 0, c = pixel.  Julia: z₀ = pixel, c = juliaC.
        let re = juliaMode ? c_re : 0;
        let im = juliaMode ? c_im : 0;
        const fixRe = juliaMode ? juliaRe : c_re;
        const fixIm = juliaMode ? juliaIm : c_im;
        let re2 = re * re, im2 = im * im, iter = 0;
        while (re2 + im2 <= 4 && iter < maxIter) {
          im  = 2 * re * im + fixIm;
          re  = re2 - im2 + fixRe;
          re2 = re * re;
          im2 = im * im;
          iter++;
        }

        if (iter === maxIter) {
          buf[i] = buf[i + 1] = buf[i + 2] = 0;
        } else {
          const smooth = iter + 1 - Math.log(Math.log(re2 + im2) * 0.5) / LN2;
          const t  = ((smooth * colorSpeed + colorOffset) % N + N) % N;
          const lo = Math.floor(t) % N;
          const hi = (lo + 1) % N;
          const f  = t - Math.floor(t);
          const a  = palette[lo];
          const b  = palette[hi];
          buf[i]     = (a[0] + (b[0] - a[0]) * f) | 0;
          buf[i + 1] = (a[1] + (b[1] - a[1]) * f) | 0;
          buf[i + 2] = (a[2] + (b[2] - a[2]) * f) | 0;
        }
        buf[i + 3] = 255;
        i += 4;
      }
    }

    // Send each tile back immediately so it paints as soon as it's ready
    self.postMessage(
      { buf, id, tileX: x, tileY: y, tileW: w, tileH: h },
      [buf.buffer] as unknown as Transferable[],
    );
  }
};

