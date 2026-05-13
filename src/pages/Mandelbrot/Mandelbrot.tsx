import { useEffect, useRef, useCallback } from 'react';
import styles from './Mandelbrot.module.css';

interface View { centerX: number; centerY: number; zoom: number; }
interface Tile  { x: number; y: number; w: number; h: number; }
interface TileResult {
  buf: Uint8ClampedArray;
  id: number;
  tileX: number; tileY: number;
  tileW: number; tileH: number;
}

const INITIAL: View = { centerX: -0.5, centerY: 0, zoom: 250 };
const TILE = 256;
const RENDER_DELAY = 300; // ms idle before starting render

function adaptiveMaxIter(zoom: number) {
  return Math.min(2000, Math.max(80, (40 + 18 * Math.log2(zoom)) | 0));
}

function buildTileList(cw: number, ch: number, dirty?: { x: number; y: number }): Tile[] {
  const cols = Math.ceil(cw / TILE);
  const rows = Math.ceil(ch / TILE);
  const cx = cols / 2, cy = rows / 2;
  const tiles: (Tile & { _d: number })[] = [];

  // Clean region = the rect of already-correct pixels after accumlated pan (dirty offset)
  // If dirty.x = 300 (panned right 300px): clean columns are [300, cw], left strip is new
  // If dirty.x = -300 (panned left 300px): clean columns are [0, cw-300], right strip is new
  const cleanX1 = dirty ? Math.max(0, dirty.x)        : 0;
  const cleanX2 = dirty ? Math.min(cw, cw + dirty.x)  : 0;
  const cleanY1 = dirty ? Math.max(0, dirty.y)        : 0;
  const cleanY2 = dirty ? Math.min(ch, ch + dirty.y)  : 0;

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const tx = col * TILE, ty = row * TILE;
      const tw = Math.min(TILE, cw - tx), th = Math.min(TILE, ch - ty);
      // Skip tile if it lies entirely within the clean (already-rendered) region
      if (dirty && tx >= cleanX1 && tx + tw <= cleanX2 && ty >= cleanY1 && ty + th <= cleanY2) {
        continue;
      }
      tiles.push({
        x: tx, y: ty, w: tw, h: th,
        _d: (col + 0.5 - cx) ** 2 + (row + 0.5 - cy) ** 2,
      });
    }
  }
  tiles.sort((a, b) => a._d - b._d);
  return tiles.map(({ _d, ...t }) => t);
}

function fmtZoom(zoom: number): string {
  const f = zoom / INITIAL.zoom;
  if (f < 1000) return `${f.toFixed(1)}\u00d7`;
  if (f < 1e6)  return `${(f / 1e3).toFixed(2)}k\u00d7`;
  return f.toExponential(2) + '\u00d7';
}

export default function Mandelbrot() {
  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const backRef     = useRef<HTMLCanvasElement | null>(null);
  const view        = useRef<View>({ ...INITIAL });
  const workerRef   = useRef<Worker | null>(null);
  const renderIdRef = useRef(0);
  const renderTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const dragRef     = useRef<{ x: number; y: number } | null>(null);
  const zoomLabel   = useRef<HTMLSpanElement>(null);
  // Accumulated pan (px) since the last render started — used to skip clean tiles on drag
  const panSinceRenderRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  // Zoom level at the last render start — if it changed, all tiles are dirty
  const renderZoomRef     = useRef<number>(INITIAL.zoom);

  // ── helpers ────────────────────────────────────────────────────────────────

  const syncBack = useCallback(() => {
    const c = canvasRef.current, b = backRef.current;
    if (c && b) b.getContext('2d')!.drawImage(c, 0, 0);
  }, []);

  const clearTimer = useCallback(() => {
    if (renderTimer.current) { clearTimeout(renderTimer.current); renderTimer.current = null; }
  }, []);

  /** Kill any running render immediately. In-flight worker messages are dropped via renderId. */
  const cancelRender = useCallback(() => {
    workerRef.current?.terminate();
    workerRef.current = null;
    renderIdRef.current++;
    clearTimer();
  }, [clearTimer]);

  /** Start a fresh full-quality render at the current view.
   *  skipClean=true: skip tiles still correct from the previous render (pure pan only). */
  const startRender = useCallback((skipClean = false) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    cancelRender();
    if (zoomLabel.current) zoomLabel.current.textContent = fmtZoom(view.current.zoom);

    // Only use dirty filtering when zoom hasn't changed since the last render
    const sameZoom = view.current.zoom === renderZoomRef.current;
    const dirty = (skipClean && sameZoom) ? { ...panSinceRenderRef.current } : undefined;
    panSinceRenderRef.current = { x: 0, y: 0 };
    renderZoomRef.current = view.current.zoom;

    const id = ++renderIdRef.current;
    const w = new Worker(new URL('./mandelbrot.worker.ts', import.meta.url), { type: 'module' });
    w.onmessage = (e: MessageEvent<TileResult>) => {
      if (e.data.id !== id) return;
      const r = e.data;
      const img = new ImageData(new Uint8ClampedArray(r.buf.buffer as ArrayBuffer), r.tileW, r.tileH);
      canvasRef.current?.getContext('2d')!.putImageData(img, r.tileX, r.tileY);
      backRef.current?.getContext('2d')!.putImageData(img, r.tileX, r.tileY);
    };
    workerRef.current = w;

    w.postMessage({
      tileList: buildTileList(canvas.width, canvas.height, dirty),
      canvasW: canvas.width, canvasH: canvas.height,
      ...view.current,
      maxIter: adaptiveMaxIter(view.current.zoom),
      id,
    });
  }, [cancelRender]);

  /** Schedule a render after the user stops interacting. */
  const scheduleRender = useCallback(() => {
    clearTimer();
    renderTimer.current = setTimeout(startRender, RENDER_DELAY);
  }, [startRender, clearTimer]);

  // ── canvas transforms (instant, no worker) ─────────────────────────────────

  const applyZoom = useCallback((factor: number, mx: number, my: number) => {
    const c = canvasRef.current, b = backRef.current;
    if (!c || !b) return;
    const ctx = c.getContext('2d')!;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.save();
    ctx.translate(mx, my);
    ctx.scale(factor, factor);
    ctx.translate(-mx, -my);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(b, 0, 0);
    ctx.restore();
    syncBack();
  }, [syncBack]);

  const applyPan = useCallback((dx: number, dy: number) => {
    panSinceRenderRef.current = {
      x: panSinceRenderRef.current.x + dx,
      y: panSinceRenderRef.current.y + dy,
    };
    const c = canvasRef.current, b = backRef.current;
    if (!c || !b) return;
    const ctx = c.getContext('2d')!;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(b, dx, dy);
    syncBack();
  }, [syncBack]);

  // ── lifecycle ──────────────────────────────────────────────────────────────

  useEffect(() => {
    backRef.current = document.createElement('canvas');
    return () => { cancelRender(); };
  }, [cancelRender]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      canvas.width  = (rect.width  * dpr) | 0;
      canvas.height = (rect.height * dpr) | 0;
      const back = backRef.current;
      if (back) { back.width = canvas.width; back.height = canvas.height; }
      startRender();
    });
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [startRender]);

  // ── wheel ──────────────────────────────────────────────────────────────────

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect    = canvas.getBoundingClientRect();
      const dpr     = canvas.width / rect.width;
      const mx      = (e.clientX - rect.left) * dpr;
      const my      = (e.clientY - rect.top)  * dpr;
      const { centerX, centerY, zoom } = view.current;
      const mouseRe = centerX + (mx - canvas.width  * 0.5) / zoom;
      const mouseIm = centerY + (my - canvas.height * 0.5) / zoom;
      const factor  = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      const newZoom = Math.max(100, Math.min(1e13, zoom * factor));
      if (newZoom === zoom) return; // already at zoom limit — nothing changed, keep current render
      view.current  = {
        centerX: mouseRe - (mx - canvas.width  * 0.5) / newZoom,
        centerY: mouseIm - (my - canvas.height * 0.5) / newZoom,
        zoom: newZoom,
      };
      cancelRender();   // stop any running render immediately
      applyZoom(factor, mx, my);
      scheduleRender(); // restart countdown
    };
    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', onWheel);
  }, [applyZoom, scheduleRender, cancelRender]);

  // ── drag ───────────────────────────────────────────────────────────────────

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onDown = (e: MouseEvent) => {
      dragRef.current = { x: e.clientX, y: e.clientY };
      canvas.style.cursor = 'grabbing';
      cancelRender();
    };
    const onMove = (e: MouseEvent) => {
      if (!dragRef.current) return;
      const rect = canvas.getBoundingClientRect();
      const dpr  = canvas.width / rect.width;
      const dx   = (e.clientX - dragRef.current.x) * dpr;
      const dy   = (e.clientY - dragRef.current.y) * dpr;
      dragRef.current = { x: e.clientX, y: e.clientY };
      const { centerX, centerY, zoom } = view.current;
      view.current = { centerX: centerX - dx / zoom, centerY: centerY - dy / zoom, zoom };
      applyPan(dx, dy);
    };
    const onUp = () => {
      if (!dragRef.current) return;
      dragRef.current = null;
      canvas.style.cursor = 'crosshair';
      startRender(true); // render immediately — skip tiles still correct from last render
    };
    canvas.addEventListener('mousedown', onDown);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup',   onUp);
    return () => {
      canvas.removeEventListener('mousedown', onDown);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup',   onUp);
    };
  }, [applyPan, startRender, cancelRender]);

  // ── reset ──────────────────────────────────────────────────────────────────

  const reset = useCallback(() => {
    view.current = { ...INITIAL };
    startRender();
  }, [startRender]);

  return (
    <div className={styles.container}>
      <canvas ref={canvasRef} className={styles.canvas} />
      <div className={styles.hud}>
        <div className={styles.hudLeft}>
          <span className={styles.hudTitle}>Mandelbrot Set</span>
          <span ref={zoomLabel} className={styles.hudZoom}>1.0&times;</span>
        </div>
        <div className={styles.hudRight}>
          <span className={styles.hudHint}>scroll to zoom &middot; drag to pan</span>
          <button className={styles.resetBtn} type="button" onClick={reset}>Reset</button>
        </div>
      </div>
    </div>
  );
}
