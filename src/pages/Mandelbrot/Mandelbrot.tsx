import { useEffect, useRef, useCallback } from 'react';
import styles from './Mandelbrot.module.css';

interface View {
  centerX: number;
  centerY: number;
  zoom: number;
}

type Quality = 'coarse' | 'full';

const INITIAL: View = { centerX: -0.5, centerY: 0, zoom: 250 };
/** How many canvas pixels map to 1 render pixel in coarse mode. */
const COARSE_SCALE = 4;

function calcMaxIter(zoom: number, quality: Quality): number {
  if (quality === 'coarse') return 28;
  return Math.min(2000, Math.max(80, Math.floor(40 + 18 * Math.log2(zoom))));
}

function fmtZoom(zoom: number): string {
  const f = zoom / INITIAL.zoom;
  if (f < 1000) return `${f.toFixed(1)}×`;
  if (f < 1e6)  return `${(f / 1e3).toFixed(2)}k×`;
  return f.toExponential(2) + '×';
}

function createWorker() {
  return new Worker(new URL('./mandelbrot.worker.ts', import.meta.url), { type: 'module' });
}

export default function Mandelbrot() {
  const canvasRef    = useRef<HTMLCanvasElement>(null);
  const view         = useRef<View>({ ...INITIAL });
  const workerRef    = useRef<Worker | null>(null);
  const renderIdRef  = useRef(0);
  const timerRef     = useRef<ReturnType<typeof setTimeout> | null>(null);
  const dragRef      = useRef<{ x: number; y: number } | null>(null);
  const zoomLabelRef = useRef<HTMLSpanElement>(null);

  // ── Core render: terminate any in-flight job, spin up fresh worker ─────
  const render = useCallback((quality: Quality) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Cancel previous computation immediately (no waiting for it to finish)
    if (workerRef.current) workerRef.current.terminate();
    const worker = createWorker();
    workerRef.current = worker;

    const { centerX, centerY, zoom } = view.current;
    const id = ++renderIdRef.current;

    if (zoomLabelRef.current) zoomLabelRef.current.textContent = fmtZoom(zoom);

    const scale   = quality === 'coarse' ? COARSE_SCALE : 1;
    const renderW = Math.max(1, Math.floor(canvas.width  / scale));
    const renderH = Math.max(1, Math.floor(canvas.height / scale));

    worker.onmessage = (
      e: MessageEvent<{ buf: Uint8ClampedArray; id: number; width: number; height: number }>,
    ) => {
      const { buf, id: rid, width: rw, height: rh } = e.data;
      if (rid !== renderIdRef.current) return; // stale result

      const cvs = canvasRef.current;
      if (!cvs) return;
      const ctx = cvs.getContext('2d');
      if (!ctx) return;

      if (rw === cvs.width && rh === cvs.height) {
        // Full-res: direct pixel write
        ctx.putImageData(new ImageData(buf, rw, rh), 0, 0);
      } else {
        // Coarse: paint to an OffscreenCanvas then scale up, pixelated
        const tmp    = new OffscreenCanvas(rw, rh);
        const tmpCtx = tmp.getContext('2d')!;
        tmpCtx.putImageData(new ImageData(buf, rw, rh), 0, 0);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(tmp, 0, 0, cvs.width, cvs.height);
      }
    };

    worker.postMessage({ width: renderW, height: renderH, centerX, centerY, zoom,
      maxIter: calcMaxIter(zoom, quality), id });
  }, []);

  // ── Schedule the crisp full render after interaction stops ─────────────
  const scheduleFullRender = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => render('full'), 350);
  }, [render]);

  // ── Initial worker setup (just needs cleanup on unmount) ───────────────
  useEffect(() => {
    return () => {
      workerRef.current?.terminate();
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  // ── Resize → update physical canvas size, kick off full render ─────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      canvas.width  = Math.floor(rect.width  * dpr);
      canvas.height = Math.floor(rect.height * dpr);
      render('full');
    });
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [render]);

  // ── Mouse wheel: coarse immediately, full after idle ───────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();

      const rect = canvas.getBoundingClientRect();
      const dpr  = canvas.width / rect.width;
      const mx   = (e.clientX - rect.left) * dpr;
      const my   = (e.clientY - rect.top)  * dpr;

      const { centerX, centerY, zoom } = view.current;
      const mouseRe = centerX + (mx - canvas.width  * 0.5) / zoom;
      const mouseIm = centerY + (my - canvas.height * 0.5) / zoom;

      const factor  = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      const newZoom = Math.max(100, Math.min(1e13, zoom * factor));

      view.current = {
        centerX: mouseRe - (mx - canvas.width  * 0.5) / newZoom,
        centerY: mouseIm - (my - canvas.height * 0.5) / newZoom,
        zoom:    newZoom,
      };

      render('coarse');
      scheduleFullRender();
    };

    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', onWheel);
  }, [render, scheduleFullRender]);

  // ── Drag to pan ────────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const onDown = (e: MouseEvent) => {
      dragRef.current = { x: e.clientX, y: e.clientY };
      canvas.style.cursor = 'grabbing';
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
      render('coarse');
    };
    const onUp = () => {
      if (!dragRef.current) return;
      dragRef.current = null;
      canvas.style.cursor = 'crosshair';
      if (timerRef.current) clearTimeout(timerRef.current);
      render('full');
    };

    canvas.addEventListener('mousedown', onDown);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup',   onUp);
    return () => {
      canvas.removeEventListener('mousedown', onDown);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup',   onUp);
    };
  }, [render]);

  // ── Reset ──────────────────────────────────────────────────────────────
  const reset = useCallback(() => {
    view.current = { ...INITIAL };
    render('full');
  }, [render]);

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
          <button className={styles.resetBtn} type="button" onClick={reset}>
            Reset
          </button>
        </div>
      </div>
    </div>
  );
}
