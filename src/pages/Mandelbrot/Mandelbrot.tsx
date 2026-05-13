import { useEffect, useRef, useCallback, useState } from 'react';
import {
  Slider, Toggle, SelectControl,
  ControlPanel, ControlGroup,
} from '@/components/Controls';
import type { PaletteId } from './mandelbrot.worker';
import styles from './Mandelbrot.module.css';

interface View { centerX: number; centerY: number; zoom: number; }
interface Tile  { x: number; y: number; w: number; h: number; }
interface TileResult {
  buf: Uint8ClampedArray;
  id: number;
  tileX: number; tileY: number;
  tileW: number; tileH: number;
}

interface ColorParams {
  paletteId:    PaletteId;
  colorSpeed:   number;
  colorOffset:  number;
  invertColors: boolean;
  maxIterMode:  'auto' | 'manual';
  maxIterManual: number;
  juliaMode:    boolean;
  juliaRe:      number;
  juliaIm:      number;
}

const PRESETS = [
  { label: 'Seahorse',  centerX: -0.7435, centerY:  0.1312, zoom:  5_000 },
  { label: 'Elephant',  centerX:  0.3245046418, centerY:  0.0485510182, zoom: 50_000 },
  { label: 'Spirals',   centerX: -0.7269, centerY:  0.1889, zoom: 40_000 },
  { label: 'Mini-Brot', centerX: -1.7549, centerY:  0.0000, zoom:  8_000 },
] as const;

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
  const canvasRef      = useRef<HTMLCanvasElement>(null);
  const backRef        = useRef<HTMLCanvasElement | null>(null);
  const view           = useRef<View>({ ...INITIAL });
  const workersRef     = useRef<Worker[]>([]);
  const renderIdRef    = useRef(0);
  const renderTimer    = useRef<ReturnType<typeof setTimeout> | null>(null);
  const dragRef        = useRef<{ x: number; y: number } | null>(null);
  const dragStartRef   = useRef<{ x: number; y: number } | null>(null);
  const zoomLabel      = useRef<HTMLSpanElement>(null);
  const crosshairRef   = useRef<HTMLDivElement>(null);
  // Accumulated pan (px) since the last render started — used to skip clean tiles on drag
  const panSinceRenderRef  = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  // Zoom level at the last render start — if it changed, all tiles are dirty
  const renderZoomRef      = useRef<number>(INITIAL.zoom);
  // Whether the last render completed fully — only skip clean tiles if it did
  const renderCompleteRef  = useRef(false);
  const tilesExpectedRef   = useRef(0);
  const tilesReceivedRef   = useRef(0);

  // ── Color / quality / Julia params ────────────────────────────────────────

  const [paletteId,     setPaletteId]     = useState<PaletteId>('classic');
  const [colorSpeed,    setColorSpeed]    = useState(0.28);
  const [colorOffset,   setColorOffset]   = useState(0);
  const [invertColors,  setInvertColors]  = useState(false);
  const [maxIterMode,   setMaxIterMode]   = useState<'auto' | 'manual'>('auto');
  const [maxIterManual, setMaxIterManual] = useState(500);
  const [juliaMode,     setJuliaMode]     = useState(false);
  const [juliaRe,       setJuliaRe]       = useState(-0.7);
  const [juliaIm,       setJuliaIm]       = useState(0.27015);

  // Mutable ref read by startRender — avoids stale closures without needing
  // to recreate callbacks every time a UI param changes.
  const cpRef = useRef<ColorParams>({
    paletteId: 'classic', colorSpeed: 0.28, colorOffset: 0,
    invertColors: false, maxIterMode: 'auto', maxIterManual: 500,
    juliaMode: false, juliaRe: -0.7, juliaIm: 0.27015,
  });

  // ── helpers ────────────────────────────────────────────────────────────────

  /** Repositions the crosshair overlay to the current Julia c point. */
  const updateCrosshair = useCallback(() => {
    const el = crosshairRef.current, canvas = canvasRef.current;
    if (!el || !canvas) return;
    if (cpRef.current.juliaMode) { el.hidden = true; return; }
    const rect = canvas.getBoundingClientRect();
    const dpr  = canvas.width / rect.width;
    const { centerX, centerY, zoom } = view.current;
    const cssX = (cpRef.current.juliaRe - centerX) * zoom / dpr + rect.width  * 0.5;
    const cssY = (cpRef.current.juliaIm - centerY) * zoom / dpr + rect.height * 0.5;
    const pad  = 16;
    const inView = cssX > pad && cssX < rect.width - pad && cssY > pad && cssY < rect.height - pad;
    el.hidden = !inView;
    if (inView) { el.style.left = `${cssX}px`; el.style.top = `${cssY}px`; }
  }, []);

  const syncBack = useCallback(() => {
    const c = canvasRef.current, b = backRef.current;
    if (c && b) b.getContext('2d')!.drawImage(c, 0, 0);
  }, []);

  const clearTimer = useCallback(() => {
    if (renderTimer.current) { clearTimeout(renderTimer.current); renderTimer.current = null; }
  }, []);

  /** Kill any running render immediately. In-flight worker messages are dropped via renderId. */
  const cancelRender = useCallback(() => {
    workersRef.current.forEach(w => w.terminate());
    workersRef.current = [];
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

    // Only skip clean tiles if: pan-only (zoom unchanged) AND last render fully completed.
    // If the previous render was interrupted, the canvas is a quality mix — render everything.
    const sameZoom    = view.current.zoom === renderZoomRef.current;
    const canSkip     = skipClean && sameZoom && renderCompleteRef.current;
    const dirty       = canSkip ? { ...panSinceRenderRef.current } : undefined;
    panSinceRenderRef.current   = { x: 0, y: 0 };
    renderZoomRef.current       = view.current.zoom;
    renderCompleteRef.current   = false;

    const tileList = buildTileList(canvas.width, canvas.height, dirty);
    tilesExpectedRef.current = tileList.length;
    tilesReceivedRef.current = 0;

    const id = ++renderIdRef.current;
    const nWorkers = Math.max(1, Math.min(navigator.hardwareConcurrency ?? 4, 8));

    // Distribute tiles round-robin so each worker gets a center-out spread
    const chunks: Tile[][] = Array.from({ length: nWorkers }, () => []);
    tileList.forEach((tile, i) => chunks[i % nWorkers].push(tile));

    const onMessage = (e: MessageEvent<TileResult>) => {
      if (e.data.id !== renderIdRef.current) return;
      const r = e.data;
      const img = new ImageData(new Uint8ClampedArray(r.buf.buffer as ArrayBuffer), r.tileW, r.tileH);
      canvasRef.current?.getContext('2d')!.putImageData(img, r.tileX, r.tileY);
      backRef.current?.getContext('2d')!.putImageData(img, r.tileX, r.tileY);
      tilesReceivedRef.current++;
      if (tilesReceivedRef.current >= tilesExpectedRef.current) {
        renderCompleteRef.current = true;
      }
    };

    const cp = cpRef.current;
    const baseMsg = {
      canvasW: canvas.width, canvasH: canvas.height,
      ...view.current,
      maxIter: cp.maxIterMode === 'auto'
        ? adaptiveMaxIter(view.current.zoom)
        : cp.maxIterManual,
      id,
      paletteId:    cp.paletteId,
      colorSpeed:   cp.colorSpeed,
      colorOffset:  cp.colorOffset,
      invertColors: cp.invertColors,
      juliaMode:    cp.juliaMode,
      juliaRe:      cp.juliaRe,
      juliaIm:      cp.juliaIm,
    };

    workersRef.current = chunks
      .filter(chunk => chunk.length > 0)
      .map(chunk => {
        const w = new Worker(new URL('./mandelbrot.worker.ts', import.meta.url), { type: 'module' });
        w.onmessage = onMessage;
        w.postMessage({ ...baseMsg, tileList: chunk });
        return w;
      });
    updateCrosshair();
  }, [cancelRender, updateCrosshair]);

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
    updateCrosshair();
  }, [syncBack, updateCrosshair]);

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
    updateCrosshair();
  }, [syncBack, updateCrosshair]);

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
      const mx      = Math.round((e.clientX - rect.left) * dpr);
      const my      = Math.round((e.clientY - rect.top)  * dpr);
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
      dragRef.current = dragStartRef.current = { x: e.clientX, y: e.clientY };
      canvas.style.cursor = 'grabbing';
      cancelRender();
    };
    const onMove = (e: MouseEvent) => {
      if (!dragRef.current) return;
      const rect = canvas.getBoundingClientRect();
      const dpr  = canvas.width / rect.width;
      const dx   = Math.round((e.clientX - dragRef.current.x) * dpr);
      const dy   = Math.round((e.clientY - dragRef.current.y) * dpr);
      dragRef.current = { x: e.clientX, y: e.clientY };
      const { centerX, centerY, zoom } = view.current;
      view.current = { centerX: centerX - dx / zoom, centerY: centerY - dy / zoom, zoom };
      applyPan(dx, dy);
    };
    const onUp = (e: MouseEvent) => {
      if (!dragRef.current) return;
      const start = dragStartRef.current;
      const wasDrag = start && (Math.hypot(e.clientX - start.x, e.clientY - start.y) > 4);
      dragRef.current = dragStartRef.current = null;
      canvas.style.cursor = 'crosshair';

      // Click (no drag) while Julia mode is active → update the c parameter
      if (!wasDrag && cpRef.current.juliaMode) {
        const rect = canvas.getBoundingClientRect();
        const dpr  = canvas.width / rect.width;
        const { centerX, centerY, zoom } = view.current;
        const px = (e.clientX - rect.left) * dpr;
        const py = (e.clientY - rect.top)  * dpr;
        const re = centerX + (px - canvas.width  * 0.5) / zoom;
        const im = centerY + (py - canvas.height * 0.5) / zoom;
        cpRef.current.juliaRe = re;
        cpRef.current.juliaIm = im;
        setJuliaRe(re);
        setJuliaIm(im);
        startRender();
        return;
      }
      startRender(true);
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

  // ── color / quality param handlers ─────────────────────────────────────────

  const handlePaletteChange = useCallback((id: PaletteId) => {
    cpRef.current.paletteId = id;
    setPaletteId(id);
    startRender();
  }, [startRender]);

  const handleColorSpeedChange = useCallback((v: number) => {
    cpRef.current.colorSpeed = v;
    setColorSpeed(v);
    startRender();
  }, [startRender]);

  const handleColorOffsetChange = useCallback((v: number) => {
    cpRef.current.colorOffset = v;
    setColorOffset(v);
    startRender();
  }, [startRender]);

  const handleInvertChange = useCallback((v: boolean) => {
    cpRef.current.invertColors = v;
    setInvertColors(v);
    startRender();
  }, [startRender]);

  const handleMaxIterModeChange = useCallback((adaptive: boolean) => {
    const mode = adaptive ? 'auto' : 'manual';
    cpRef.current.maxIterMode = mode;
    setMaxIterMode(mode);
    startRender();
  }, [startRender]);

  const handleMaxIterManualChange = useCallback((v: number) => {
    cpRef.current.maxIterManual = v;
    cpRef.current.maxIterMode = 'manual';
    setMaxIterManual(v);
    setMaxIterMode('manual');
    startRender();
  }, [startRender]);

  const handleJuliaModeChange = useCallback((on: boolean) => {
    cpRef.current.juliaMode = on;
    setJuliaMode(on);
    if (on) view.current = { centerX: 0, centerY: 0, zoom: 250 };
    else view.current = { ...INITIAL };
    startRender();
  }, [startRender]);

  const handleJuliaReChange = useCallback((v: number) => {
    cpRef.current.juliaRe = v;
    setJuliaRe(v);
    updateCrosshair();
    startRender();
  }, [startRender, updateCrosshair]);

  const handleJuliaImChange = useCallback((v: number) => {
    cpRef.current.juliaIm = v;
    setJuliaIm(v);
    updateCrosshair();
    startRender();
  }, [startRender, updateCrosshair]);

  const goToPreset = useCallback((p: typeof PRESETS[number]) => {
    view.current = { centerX: p.centerX, centerY: p.centerY, zoom: p.zoom };
    startRender();
  }, [startRender]);

  const saveImage = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const a = document.createElement('a');
    a.href     = canvas.toDataURL('image/png');
    a.download = 'mandelbrot.png';
    a.click();
  }, []);

  return (
    <div className={styles.container}>
      <canvas ref={canvasRef} className={styles.canvas} />

      <div className={styles.sidebar}>
        <ControlPanel title="Explore">
          <ControlGroup>
            <div className={styles.snapRow}>
              {PRESETS.map(p => (
                <button key={p.label} className={styles.snapBtn} type="button"
                  onClick={() => goToPreset(p)}>
                  {p.label}
                </button>
              ))}
            </div>
          </ControlGroup>
        </ControlPanel>

        <ControlPanel title="Colors">
          <ControlGroup>
            <SelectControl
              label="Palette"
              value={paletteId}
              onChange={handlePaletteChange}
              options={[
                { value: 'classic'  as const, label: 'Classic' },
                { value: 'fire'     as const, label: 'Fire' },
                { value: 'ice'      as const, label: 'Ice' },
                { value: 'electric' as const, label: 'Electric' },
                { value: 'mono'     as const, label: 'Monochrome' },
                { value: 'sunset'   as const, label: 'Sunset' },
              ]}
            />
          </ControlGroup>
          <ControlGroup>
            <Slider
              label="Color Speed"
              value={colorSpeed}
              min={0.05} max={3} step={0.01}
              onChange={handleColorSpeedChange}
            />
            <Slider
              label="Color Offset"
              value={colorOffset}
              min={0} max={16} step={0.1}
              onChange={handleColorOffsetChange}
            />
          </ControlGroup>
          <ControlGroup>
            <Toggle label="Invert Colors" value={invertColors} onChange={handleInvertChange} />
          </ControlGroup>
        </ControlPanel>

        <ControlPanel title="Julia Set" defaultOpen={false}>
          <ControlGroup>
            <Toggle
              label="Julia Mode"
              value={juliaMode}
              onChange={handleJuliaModeChange}
              description="Enable, then click canvas to pick c"
            />
          </ControlGroup>
          <ControlGroup>
            <Slider
              label="Re(c)"
              value={juliaRe}
              min={-2} max={2} step={0.001}
              onChange={handleJuliaReChange}
            />
            <Slider
              label="Im(c)"
              value={juliaIm}
              min={-2} max={2} step={0.001}
              onChange={handleJuliaImChange}
            />
          </ControlGroup>
        </ControlPanel>

        <ControlPanel title="Quality" defaultOpen={false}>
          <ControlGroup>
            <Toggle
              label="Adaptive Iterations"
              value={maxIterMode === 'auto'}
              onChange={handleMaxIterModeChange}
              description="Scales with zoom depth"
            />
            <Slider
              label="Max Iterations"
              value={maxIterManual}
              min={50} max={2000} step={50}
              onChange={handleMaxIterManualChange}
            />
          </ControlGroup>
        </ControlPanel>

        <button className={styles.saveBtn} type="button" onClick={saveImage}>
          ↓ Save PNG
        </button>

        <button className={styles.resetBtn} type="button" onClick={reset}>
          Reset View
        </button>
      </div>

      <div className={styles.hud}>
        <div className={styles.hudLeft}>
          <span className={styles.hudTitle}>
            {juliaMode ? 'Julia Set' : 'Mandelbrot Set'}
          </span>
          <span ref={zoomLabel} className={styles.hudZoom}>1.0&times;</span>
        </div>
        <div className={styles.hudRight}>
          <span className={styles.hudHint}>
            {juliaMode
              ? 'click to pick c · scroll to zoom · drag to pan'
              : 'scroll to zoom · drag to pan'}
          </span>
        </div>
      </div>
    </div>
  );
}
