/// <reference lib="webworker" />

export type PaletteId = 'classic' | 'fire' | 'ice' | 'electric' | 'mono' | 'sunset';

type RGB = readonly [number, number, number];

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
}

self.onmessage = (e: MessageEvent<RenderMsg>) => {
  const {
    tileList, canvasW, canvasH, centerX, centerY, zoom, maxIter, id,
    paletteId, colorSpeed, colorOffset, invertColors,
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

        let re = 0, im = 0, re2 = 0, im2 = 0, iter = 0;
        while (re2 + im2 <= 4 && iter < maxIter) {
          im  = 2 * re * im + c_im;
          re  = re2 - im2 + c_re;
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

