// ─── Types ────────────────────────────────────────────────────────────────────

export interface IFSTransform {
  a: number; b: number;
  c: number; d: number;
  e: number; f: number;
  p: number; // relative probability
}

export interface IFSPreset {
  id: string;
  label: string;
  name: string;
  transforms: IFSTransform[];
  cdf: number[]; // precomputed cumulative distribution (normalized)
  xMin: number; xMax: number;
  yMin: number; yMax: number;
}

export type ColorSchemeId = 'fern' | 'heat' | 'frost' | 'mono' | 'plasma';

export interface RenderViewport {
  zoom: number;
  panX: number; // fraction of half-width  (matches Koch convention)
  panY: number; // fraction of half-height
}

// ─── Color palettes ───────────────────────────────────────────────────────────
// Up to 6 entries, one per transform slot (cycles if more transforms).

export const COLOR_PALETTES: Record<ColorSchemeId, [number, number, number][]> = {
  fern:   [[22,101,52],[34,197,94],[134,239,172],[187,247,208],[167,243,208],[209,250,229]],
  heat:   [[127,29,21],[239,68,68],[251,146,60],[251,191,36],[253,224,71],[254,240,138]],
  frost:  [[30,58,138],[59,130,246],[147,197,253],[219,234,254],[224,242,254],[240,249,255]],
  mono:   [[55,65,81],[107,114,128],[156,163,175],[209,213,219],[229,231,235],[249,250,251]],
  plasma: [[88,28,135],[168,85,247],[232,121,249],[240,171,252],[245,208,254],[253,240,255]],
};

// ─── IFS preset definitions ───────────────────────────────────────────────────

function buildCDF(transforms: { p: number }[]): number[] {
  const total = transforms.reduce((s, t) => s + t.p, 0);
  let acc = 0;
  return transforms.map(t => (acc += t.p / total, acc));
}

const rawPresets: Omit<IFSPreset, 'cdf'>[] = [
  {
    id: 'barnsley',
    label: 'Fern',
    name: 'Barnsley Fern',
    transforms: [
      { a: 0,     b: 0,     c: 0,     d: 0.16,  e: 0,    f: 0,    p: 0.01 },
      { a: 0.85,  b: 0.04,  c: -0.04, d: 0.85,  e: 0,    f: 1.6,  p: 0.85 },
      { a: 0.20,  b: -0.26, c: 0.23,  d: 0.22,  e: 0,    f: 1.6,  p: 0.07 },
      { a: -0.15, b: 0.28,  c: 0.26,  d: 0.24,  e: 0,    f: 0.44, p: 0.07 },
    ],
    xMin: -2.5, xMax: 2.5, yMin: 0, yMax: 10,
  },
  {
    id: 'sierpinski',
    label: 'Sierpiński',
    name: 'Sierpiński Triangle',
    transforms: [
      { a: 0.5, b: 0, c: 0, d: 0.5, e: 0,   f: 0,     p: 1 },
      { a: 0.5, b: 0, c: 0, d: 0.5, e: 1,   f: 0,     p: 1 },
      { a: 0.5, b: 0, c: 0, d: 0.5, e: 0.5, f: 0.866, p: 1 },
    ],
    xMin: -0.1, xMax: 2.1, yMin: -0.1, yMax: 1.97,
  },
  {
    id: 'dragon',
    label: 'Dragon',
    name: 'Heighway Dragon',
    transforms: [
      { a: 0.5,  b: -0.5, c: 0.5, d:  0.5, e: 0, f: 0, p: 1 },
      { a: -0.5, b: -0.5, c: 0.5, d: -0.5, e: 1, f: 0, p: 1 },
    ],
    xMin: -0.6, xMax: 1.6, yMin: -0.5, yMax: 1.1,
  },
  {
    id: 'levy',
    label: 'Lévy',
    name: 'Lévy C Curve',
    transforms: [
      { a: 0.5, b: -0.5, c:  0.5, d: 0.5, e: 0,   f: 0,   p: 1 },
      { a: 0.5, b:  0.5, c: -0.5, d: 0.5, e: 0.5, f: 0.5, p: 1 },
    ],
    xMin: -0.2, xMax: 1.2, yMin: -0.4, yMax: 1.2,
  },
  {
    id: 'tree',
    label: 'Tree',
    name: 'Fractal Tree',
    transforms: [
      { a: 0,    b: 0,     c: 0,     d: 0.5,  e: 0, f: 0,   p: 0.05  },
      { a: 0.42, b: -0.42, c: 0.42,  d: 0.42, e: 0, f: 0.2, p: 0.475 },
      { a: 0.42, b:  0.42, c: -0.42, d: 0.42, e: 0, f: 0.2, p: 0.475 },
    ],
    xMin: -1.2, xMax: 1.2, yMin: -0.3, yMax: 1.3,
  },
];

export const PRESETS: IFSPreset[] = rawPresets.map(p => ({
  ...p,
  cdf: buildCDF(p.transforms),
}));

// ─── Core math ────────────────────────────────────────────────────────────────

/** One chaos-game step. Returns [newX, newY, transformIndex]. */
export function chaosStep(
  x: number,
  y: number,
  preset: IFSPreset,
): [number, number, number] {
  const r = Math.random();
  let ti = 0;
  while (ti < preset.cdf.length - 1 && r > preset.cdf[ti]) ti++;
  const t = preset.transforms[ti];
  return [t.a * x + t.b * y + t.e, t.c * x + t.d * y + t.f, ti];
}

/** Map an IFS coordinate to canvas pixel coordinates (integer). */
export function ifsToScreen(
  ifsX: number,
  ifsY: number,
  preset: IFSPreset,
  W: number,
  H: number,
  vp: RenderViewport,
): [number, number] {
  const cx = (preset.xMin + preset.xMax) / 2;
  const cy = (preset.yMin + preset.yMax) / 2;
  const baseScale = 0.88 * Math.min(W / (preset.xMax - preset.xMin), H / (preset.yMax - preset.yMin));
  const scale = baseScale * vp.zoom;
  const sx = W / 2 + (ifsX - cx) * scale + vp.panX * W / 2;
  const sy = H / 2 - (ifsY - cy) * scale - vp.panY * H / 2;
  return [sx | 0, sy | 0];
}

// ─── Off-screen render (for export & mini-preview) ───────────────────────────

export const BG_COLOR: [number, number, number] = [7, 7, 18];

/**
 * Renders `iterations` chaos-game points into `imgData` using density accumulation.
 * Call ctx.putImageData(imgData, 0, 0) afterwards to display.
 */
export function renderIntoImageData(
  imgData: ImageData,
  preset: IFSPreset,
  palette: [number, number, number][],
  vp: RenderViewport,
  iterations: number,
  warmup = 20,
): void {
  const { width: W, height: H } = imgData;
  const { counts, txIdx } = makeCountBuffers(W, H);
  let x = 0, y = 0;
  let maxCount = 1;

  for (let i = 0; i < iterations + warmup; i++) {
    const [nx, ny, ti] = chaosStep(x, y, preset);
    x = nx; y = ny;
    if (i < warmup) continue;

    const [sx, sy] = ifsToScreen(x, y, preset, W, H, vp);
    if (sx >= 0 && sx < W && sy >= 0 && sy < H) {
      const pidx = sy * W + sx;
      const c = ++counts[pidx];
      if (c > maxCount) maxCount = c;
      txIdx[pidx] = ti;
    }
  }

  countsToImageData(imgData, counts, txIdx, palette, maxCount);
}

/** Creates a fresh ImageData pre-filled with the background colour. */
export function makeDarkImageData(ctx: CanvasRenderingContext2D): ImageData {
  const { width: W, height: H } = ctx.canvas;
  const img = ctx.createImageData(W, H);
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    d[i] = BG_COLOR[0]; d[i + 1] = BG_COLOR[1]; d[i + 2] = BG_COLOR[2]; d[i + 3] = 255;
  }
  return img;
}

/** Creates zero-initialised hit-count and transform-index buffers for W×H pixels. */
export function makeCountBuffers(W: number, H: number): { counts: Uint32Array; txIdx: Uint8Array } {
  return { counts: new Uint32Array(W * H), txIdx: new Uint8Array(W * H) };
}

/**
 * Converts per-pixel hit counts into RGBA using log-scale brightness mapping.
 * Dense pixels glow at full palette colour; sparse pixels fade toward the background.
 * maxCount should be the current maximum value in `counts` (tracked incrementally).
 */
export function countsToImageData(
  imgData: ImageData,
  counts: Uint32Array,
  txIdx: Uint8Array,
  palette: [number, number, number][],
  maxCount: number,
): void {
  const { data, width: W, height: H } = imgData;
  const len = W * H;
  const logMax = Math.log(1 + maxCount);
  const [br, bg, bb] = BG_COLOR;
  for (let i = 0; i < len; i++) {
    const idx = i << 2;
    const c = counts[i];
    if (c === 0) {
      data[idx] = br; data[idx + 1] = bg; data[idx + 2] = bb; data[idx + 3] = 255;
    } else {
      const t = Math.log(1 + c) / logMax;
      const col = palette[txIdx[i] % palette.length];
      data[idx]     = (col[0] * t + br * (1 - t)) | 0;
      data[idx + 1] = (col[1] * t + bg * (1 - t)) | 0;
      data[idx + 2] = (col[2] * t + bb * (1 - t)) | 0;
      data[idx + 3] = 255;
    }
  }
}
