import { useEffect, useRef, useCallback } from 'react';
import styles from './Mandelbrot.module.css';

interface View {
  centerX: number;
  centerY: number;
  zoom: number;
}

const INITIAL: View = { centerX: -0.5, centerY: 0, zoom: 250 };

/** Scale max-iterations with zoom depth so detail keeps appearing. */
function calcMaxIter(zoom: number, quality: 'preview' | 'full'): number {
  const full = Math.min(2000, Math.max(80, Math.floor(40 + 18 * Math.log2(zoom))));
  return quality === 'preview' ? Math.max(40, (full * 0.25) | 0) : full;
}

function fmtZoom(zoom: number): string {
  const f = zoom / INITIAL.zoom;
  if (f < 1000) return `${f.toFixed(1)}×`;
  if (f < 1e6)  return `${(f / 1e3).toFixed(2)}k×`;
  return f.toExponential(2) + '×';
}

export default function Mandelbrot() {
  const canvasRef    = useRef<HTMLCanvasElement>(null);
  const view         = useRef<View>({ ...INITIAL });
  const workerRef    = useRef<Worker | null>(null);
  const renderIdRef  = useRef(0);
  const rafRef       = useRef(0);
  const timerRef     = useRef<ReturnType<typeof setTimeout> | null>(null);
  const dragRef      = useRef<{ x: number; y: number } | null>(null);
  const zoomLabelRef = useRef<HTMLSpanElement>(null);

  // ── Post a render request to the worker ────────────────────────────────
  const render = useCallback((quality: 'preview' | 'full') => {
    cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      const canvas = canvasRef.current;
      const worker = workerRef.current;
      if (!canvas || !worker) return;

      const { centerX, centerY, zoom } = view.current;
      const id = ++renderIdRef.current;

      if (zoomLabelRef.current) zoomLabelRef.current.textContent = fmtZoom(zoom);

      worker.postMessage({
        width:   canvas.width,
        height:  canvas.height,
        centerX, centerY, zoom,
        maxIter: calcMaxIter(zoom, quality),
        id,
      });
    });
  }, []);

  // ── After interaction stops, upgrade to full quality ───────────────────
  const scheduleFullRender = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => render('full'), 220);
  }, [render]);

  // ── Spin up the worker ─────────────────────────────────────────────────
  useEffect(() => {
    const worker = new Worker(
      new URL('./mandelbrot.worker.ts', import.meta.url),
      { type: 'module' },
    );
    workerRef.current = worker;

    worker.onmessage = (e: MessageEvent<{ buf: Uint8ClampedArray; id: number }>) => {
      // Discard stale renders
      if (e.data.id !== renderIdRef.current) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      ctx?.putImageData(new ImageData(e.data.buf, canvas.width, canvas.height), 0, 0);
    };

    return () => {
      worker.terminate();
      cancelAnimationFrame(rafRef.current);
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  // ── Resize → set physical canvas resolution, trigger render ───────────
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

  // ── Mouse wheel zoom toward pointer ────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();

      const rect = canvas.getBoundingClientRect();
      const dpr  = canvas.width / rect.width;
      const mx   = (e.clientX - rect.left)  * dpr;
      const my   = (e.clientY - rect.top)   * dpr;

      const { centerX, centerY, zoom } = view.current;

      // Complex coordinate currently under the mouse
      const mouseRe = centerX + (mx - canvas.width  * 0.5) / zoom;
      const mouseIm = centerY + (my - canvas.height * 0.5) / zoom;

      const factor  = e.deltaY < 0 ? 1.12 : 1 / 1.12;
      const newZoom = Math.max(100, Math.min(1e13, zoom * factor));

      // Shift center so the complex point stays fixed under the mouse
      view.current = {
        centerX: mouseRe - (mx - canvas.width  * 0.5) / newZoom,
        centerY: mouseIm - (my - canvas.height * 0.5) / newZoom,
        zoom:    newZoom,
      };

      render('preview');
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
      render('preview');
    };
    const onUp = () => {
      if (!dragRef.current) return;
      dragRef.current = null;
      canvas.style.cursor = 'crosshair';
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

  // ── Reset view ─────────────────────────────────────────────────────────
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
