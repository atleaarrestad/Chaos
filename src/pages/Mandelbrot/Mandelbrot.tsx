import { useEffect, useRef, useCallback } from 'react';
import styles from './Mandelbrot.module.css';

interface View { centerX: number; centerY: number; zoom: number; }
type Stage = 'draft' | 'refine1' | 'refine2';
interface WorkerResult { buf: Uint8ClampedArray; id: number; width: number; height: number; }

const INITIAL: View = { centerX: -0.5, centerY: 0, zoom: 250 };
const DRAFT_SCALE = 4; // 1/16th the pixel count → ~20-50ms per frame

function adaptiveMaxIter(zoom: number) {
  return Math.min(2000, Math.max(80, (40 + 18 * Math.log2(zoom)) | 0));
}

function stageConfig(zoom: number, stage: Stage) {
  const full = adaptiveMaxIter(zoom);
  switch (stage) {
    case 'draft':   return { resScale: DRAFT_SCALE, maxIter: 20 };
    case 'refine1': return { resScale: 1, maxIter: Math.max(40, (full * 0.35) | 0) };
    case 'refine2': return { resScale: 1, maxIter: full };
  }
}

function fmtZoom(zoom: number): string {
  const f = zoom / INITIAL.zoom;
  if (f < 1000) return `${f.toFixed(1)}×`;
  if (f < 1e6)  return `${(f / 1e3).toFixed(2)}k×`;
  return f.toExponential(2) + '×';
}

/** Blit a small buffer up to fill the full canvas. */
function drawScaled(ctx: CanvasRenderingContext2D, buf: Uint8ClampedArray,
                    rw: number, rh: number, cw: number, ch: number) {
  const tmp = document.createElement('canvas');
  tmp.width = rw; tmp.height = rh;
  tmp.getContext('2d')!.putImageData(new ImageData(buf, rw, rh), 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tmp, 0, 0, cw, ch);
}

export default function Mandelbrot() {
  const canvasRef     = useRef<HTMLCanvasElement>(null);
  const view          = useRef<View>({ ...INITIAL });
  const workerRef     = useRef<Worker | null>(null);
  const renderIdRef   = useRef(0);
  /** What the worker is currently computing. */
  const inFlightRef   = useRef<'none' | 'draft' | 'refine'>('none');
  /** Did the view change while a draft was in-flight? */
  const draftDirtyRef = useRef(false);
  const r1TimerRef    = useRef<ReturnType<typeof setTimeout> | null>(null);
  const r2TimerRef    = useRef<ReturnType<typeof setTimeout> | null>(null);
  const dragRef       = useRef<{ x: number; y: number } | null>(null);
  const zoomLabelRef  = useRef<HTMLSpanElement>(null);
  /** Stable ref so worker's onmessage can call render without stale closure. */
  const renderFnRef   = useRef<(stage: Stage) => void>(() => {});

  // ── Worker message handler ─────────────────────────────────────────────
  const handleWorkerMsg = useCallback((e: MessageEvent<WorkerResult>) => {
    const { buf, id: rid, width: rw, height: rh } = e.data;
    if (rid !== renderIdRef.current) return; // stale result, discard

    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    if (rw === canvas.width && rh === canvas.height) {
      ctx.putImageData(new ImageData(buf, rw, rh), 0, 0);
    } else {
      drawScaled(ctx, buf, rw, rh, canvas.width, canvas.height);
    }

    inFlightRef.current = 'none';

    // Chain: if view moved while we were computing this draft, render again
    if (draftDirtyRef.current) {
      draftDirtyRef.current = false;
      renderFnRef.current('draft');
    }
  }, []);

  // ── Terminate current worker and spin up a fresh one ──────────────────
  const spawnWorker = useCallback(() => {
    workerRef.current?.terminate();
    const w = new Worker(new URL('./mandelbrot.worker.ts', import.meta.url), { type: 'module' });
    w.onmessage = handleWorkerMsg;
    workerRef.current = w;
    inFlightRef.current = 'none';
    draftDirtyRef.current = false;
  }, [handleWorkerMsg]);

  // ── Send a render job to the existing worker ───────────────────────────
  const postJob = useCallback((stage: Stage) => {
    const canvas = canvasRef.current;
    const worker = workerRef.current;
    if (!canvas || !worker) return;

    const { resScale, maxIter } = stageConfig(view.current.zoom, stage);
    const rw = Math.max(1, (canvas.width  / resScale) | 0);
    const rh = Math.max(1, (canvas.height / resScale) | 0);
    const id = ++renderIdRef.current;

    if (zoomLabelRef.current) zoomLabelRef.current.textContent = fmtZoom(view.current.zoom);

    inFlightRef.current = stage === 'draft' ? 'draft' : 'refine';
    draftDirtyRef.current = false;
    worker.postMessage({ width: rw, height: rh, ...view.current, maxIter, id });
  }, []);

  // ── Main render entry point ────────────────────────────────────────────
  const render = useCallback((stage: Stage) => {
    if (stage === 'draft') {
      switch (inFlightRef.current) {
        case 'draft':
          // Never interrupt a fast draft; just note the view changed.
          // handleWorkerMsg will chain another draft when this one finishes.
          draftDirtyRef.current = true;
          return;
        case 'refine':
          // Abort the slow refine so the draft can start immediately.
          spawnWorker();
          break;
        // case 'none': fall through
      }
    } else {
      // Refine always aborts whatever is running.
      spawnWorker();
    }
    postJob(stage);
  }, [spawnWorker, postJob]);

  // Keep renderFnRef current so worker callbacks never hold a stale render.
  useEffect(() => { renderFnRef.current = render; }, [render]);

  const clearRefineTimers = useCallback(() => {
    if (r1TimerRef.current) { clearTimeout(r1TimerRef.current); r1TimerRef.current = null; }
    if (r2TimerRef.current) { clearTimeout(r2TimerRef.current); r2TimerRef.current = null; }
  }, []);

  /** After interaction stops: refine1 at 200 ms, refine2 at 800 ms. */
  const scheduleRefine = useCallback(() => {
    clearRefineTimers();
    r1TimerRef.current = setTimeout(() => render('refine1'), 200);
    r2TimerRef.current = setTimeout(() => render('refine2'), 800);
  }, [render, clearRefineTimers]);

  // ── Lifecycle ──────────────────────────────────────────────────────────
  useEffect(() => {
    spawnWorker();
    return () => { workerRef.current?.terminate(); clearRefineTimers(); };
  }, [spawnWorker, clearRefineTimers]);

  // ── Resize → update physical canvas dimensions, full re-render ─────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      canvas.width  = (rect.width  * dpr) | 0;
      canvas.height = (rect.height * dpr) | 0;
      render('refine2');
    });
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [render]);

  // ── Wheel: draft immediately, queue refinement after idle ──────────────
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
      render('draft');
      scheduleRefine();
    };
    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', onWheel);
  }, [render, scheduleRefine]);

  // ── Drag: draft while panning, full quality on release ─────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onDown = (e: MouseEvent) => {
      dragRef.current = { x: e.clientX, y: e.clientY };
      canvas.style.cursor = 'grabbing';
      clearRefineTimers();
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
      render('draft');
    };
    const onUp = () => {
      if (!dragRef.current) return;
      dragRef.current = null;
      canvas.style.cursor = 'crosshair';
      clearRefineTimers();
      render('refine2'); // skip refine1, go straight to full quality
    };
    canvas.addEventListener('mousedown', onDown);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup',   onUp);
    return () => {
      canvas.removeEventListener('mousedown', onDown);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup',   onUp);
    };
  }, [render, clearRefineTimers]);

  const reset = useCallback(() => {
    view.current = { ...INITIAL };
    clearRefineTimers();
    render('refine2');
  }, [render, clearRefineTimers]);

  return (
    <div className={styles.container}>
      <canvas ref={canvasRef} className={styles.canvas} />
      <div className={styles.hud}>
        <div className={styles.hudLeft}>
          <span className={styles.hudTitle}>Mandelbrot Set</span>
          <span ref={zoomLabelRef} className={styles.hudZoom}>1.0×</span>
        </div>
        <div className={styles.hudRight}>
          <span className={styles.hudHint}>scroll to zoom · drag to pan</span>
          <button className={styles.resetBtn} type="button" onClick={reset}>Reset</button>
        </div>
      </div>
    </div>
  );
}
