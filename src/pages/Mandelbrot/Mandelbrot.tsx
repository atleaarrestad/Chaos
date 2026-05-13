import { useEffect, useRef, useCallback } from 'react';
import styles from './Mandelbrot.module.css';

interface View { centerX: number; centerY: number; zoom: number; }
type Stage = 'fast' | 'full';
interface Tile { x: number; y: number; w: number; h: number; }
interface TileResult {
  buf: Uint8ClampedArray;
  id: number;
  tileX: number; tileY: number;
  tileW: number; tileH: number;
}

const INITIAL: View = { centerX: -0.5, centerY: 0, zoom: 250 };
const TILE = 256;

function adaptiveMaxIter(zoom: number) {
  return Math.min(2000, Math.max(80, (40 + 18 * Math.log2(zoom)) | 0));
}
function stageMaxIter(zoom: number, stage: Stage) {
  const full = adaptiveMaxIter(zoom);
  return stage === 'fast' ? Math.max(30, (full * 0.28) | 0) : full;
}

function buildTileList(cw: number, ch: number): Tile[] {
  const cols = Math.ceil(cw / TILE);
  const rows = Math.ceil(ch / TILE);
  const cx = cols / 2, cy = rows / 2;
  const tiles = [];
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      tiles.push({
        x: col * TILE, y: row * TILE,
        w: Math.min(TILE, cw - col * TILE),
        h: Math.min(TILE, ch - row * TILE),
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
  const canvasRef  = useRef<HTMLCanvasElement>(null);
  const backRef    = useRef<HTMLCanvasElement | null>(null);
  const view       = useRef<View>({ ...INITIAL });
  const workerRef  = useRef<Worker | null>(null);
  const renderIdRef = useRef(0);
  const fastTimer  = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fullTimer  = useRef<ReturnType<typeof setTimeout> | null>(null);
  const dragRef    = useRef<{ x: number; y: number } | null>(null);
  const zoomLabel  = useRef<HTMLSpanElement>(null);

  const syncBack = useCallback(() => {
    const c = canvasRef.current, b = backRef.current;
    if (c && b) b.getContext('2d')!.drawImage(c, 0, 0);
  }, []);

  const drawTile = useCallback((r: TileResult) => {
    const c = canvasRef.current, b = backRef.current;
    if (!c || !b) return;
    const img = new ImageData(new Uint8ClampedArray(r.buf.buffer as ArrayBuffer), r.tileW, r.tileH);
    c.getContext('2d')!.putImageData(img, r.tileX, r.tileY);
    b.getContext('2d')!.putImageData(img, r.tileX, r.tileY);
  }, []);

  const spawnWorker = useCallback(() => {
    workerRef.current?.terminate();
    const w = new Worker(new URL('./mandelbrot.worker.ts', import.meta.url), { type: 'module' });
    w.onmessage = (e: MessageEvent<TileResult>) => {
      if (e.data.id !== renderIdRef.current) return;
      drawTile(e.data);
    };
    workerRef.current = w;
  }, [drawTile]);

  const startRender = useCallback((stage: Stage) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    spawnWorker();
    const id = ++renderIdRef.current;
    if (zoomLabel.current) zoomLabel.current.textContent = fmtZoom(view.current.zoom);
    workerRef.current!.postMessage({
      tileList: buildTileList(canvas.width, canvas.height),
      canvasW: canvas.width, canvasH: canvas.height,
      ...view.current,
      maxIter: stageMaxIter(view.current.zoom, stage),
      id,
    });
  }, [spawnWorker]);

  const clearTimers = useCallback(() => {
    if (fastTimer.current) { clearTimeout(fastTimer.current); fastTimer.current = null; }
    if (fullTimer.current) { clearTimeout(fullTimer.current); fullTimer.current = null; }
  }, []);

  const scheduleRender = useCallback(() => {
    clearTimers();
    fastTimer.current = setTimeout(() => startRender('fast'), 150);
    fullTimer.current = setTimeout(() => startRender('full'), 650);
  }, [startRender, clearTimers]);

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
    const c = canvasRef.current, b = backRef.current;
    if (!c || !b) return;
    const ctx = c.getContext('2d')!;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(b, dx, dy);
    syncBack();
  }, [syncBack]);

  useEffect(() => {
    backRef.current = document.createElement('canvas');
    spawnWorker();
    return () => { workerRef.current?.terminate(); clearTimers(); };
  }, [spawnWorker, clearTimers]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      const cw = (rect.width  * dpr) | 0;
      const ch = (rect.height * dpr) | 0;
      canvas.width = cw; canvas.height = ch;
      const back = backRef.current;
      if (back) { back.width = cw; back.height = ch; }
      startRender('full');
    });
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [startRender]);

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
      view.current  = {
        centerX: mouseRe - (mx - canvas.width  * 0.5) / newZoom,
        centerY: mouseIm - (my - canvas.height * 0.5) / newZoom,
        zoom: newZoom,
      };
      applyZoom(factor, mx, my);
      scheduleRender();
    };
    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', onWheel);
  }, [applyZoom, scheduleRender]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onDown = (e: MouseEvent) => {
      dragRef.current = { x: e.clientX, y: e.clientY };
      canvas.style.cursor = 'grabbing';
      clearTimers();
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
      clearTimers();
      startRender('full');
    };
    canvas.addEventListener('mousedown', onDown);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup',   onUp);
    return () => {
      canvas.removeEventListener('mousedown', onDown);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup',   onUp);
    };
  }, [applyPan, startRender, clearTimers]);

  const reset = useCallback(() => {
    view.current = { ...INITIAL };
    clearTimers();
    startRender('full');
  }, [startRender, clearTimers]);

  return (
    <div className={styles.container}>
      <canvas ref={canvasRef} className={styles.canvas} />
      <div className={styles.hud}>
        <div className={styles.hudLeft}>
          <span className={styles.hudTitle}>Mandelbrot Set</span>
          <span ref={zoomLabel} className={styles.hudZoom}>1.0&times;</span>
        </div>
        <div className={styles.hudRight}>
          <span className={styles.hudHint}>scroll to zoom · drag to pan</span>
          <button className={styles.resetBtn} type="button" onClick={reset}>Reset</button>
        </div>
      </div>
    </div>
  );
}
