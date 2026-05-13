/// <reference lib="webworker" />

const PALETTE: readonly [number, number, number][] = [
  [  9,   1,  47], [  4,   4,  73], [  0,   7, 100], [ 12,  44, 138],
  [ 24,  82, 177], [ 57, 125, 209], [134, 181, 229], [211, 236, 248],
  [241, 233, 191], [248, 201,  95], [255, 170,   0], [204, 128,   0],
  [153,  87,   0], [106,  52,   3], [ 66,  30,  15], [ 25,   7,  26],
] as const;

const N     = PALETTE.length;
const LN2   = Math.LN2;
const SPEED = 0.28;

interface Tile { x: number; y: number; w: number; h: number; }

interface RenderMsg {
  tileList: Tile[];
  canvasW:  number;
  canvasH:  number;
  centerX:  number;
  centerY:  number;
  zoom:     number;
  maxIter:  number;
  id:       number;
}

self.onmessage = (e: MessageEvent<RenderMsg>) => {
  const { tileList, canvasW, canvasH, centerX, centerY, zoom, maxIter, id } = e.data;
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
          const t  = (smooth * SPEED) % N;
          const lo = Math.floor(t);
          const hi = (lo + 1) % N;
          const f  = t - lo;
          const a  = PALETTE[lo];
          const b  = PALETTE[hi];
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

