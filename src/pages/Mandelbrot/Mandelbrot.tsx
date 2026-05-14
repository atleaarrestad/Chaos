import { useEffect, useRef, useCallback, useState } from 'react';
import {
  Slider, Toggle, SelectControl,
  ControlPanel, ControlGroup,
} from '@/components/Controls';
import type { PaletteId } from './mandelbrot.worker';
import {
  detectWebGL, createWebGLRenderer,
  type WebGLRenderer, type GLRenderParams,
} from './mandelbrot-webgl';
import { HP_THRESHOLD, hpPan, hpZoomTo } from './hp';
import styles from './Mandelbrot.module.css';

interface View { centerX: number; centerY: number; zoom: number; hpCX?: string; hpCY?: string; }
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
  const glCanvasRef    = useRef<HTMLCanvasElement>(null);
  const snapshotRef    = useRef<HTMLCanvasElement>(null);
  const backRef        = useRef<HTMLCanvasElement | null>(null);
  const view           = useRef<View>({ ...INITIAL });
  const workersRef     = useRef<Worker[]>([]);
  const webglRef       = useRef<WebGLRenderer | null>(null);
  const useGPURef      = useRef(detectWebGL());
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
  const [gpuAvailable,  setGpuAvailable]  = useState(false);
  const [useGPU,        setUseGPU]        = useState(() => detectWebGL());

  // ── Animation state ────────────────────────────────────────────────────────
  const [animating,        setAnimating]        = useState(false);
  const [animMode,         setAnimMode]         = useState<'zoom' | 'julia' | 'both'>('zoom');
  const [animSpeed,        setAnimSpeed]        = useState(0.5);
  const [animZoomDir,      setAnimZoomDir]      = useState<'in' | 'out' | 'pingpong'>('pingpong');
  const [juliaOrbitRadius, setJuliaOrbitRadius] = useState(0.7);
  const [colorCycle,       setColorCycle]       = useState(false);

  // Animation refs — read inside rAF loop to avoid stale closures
  const animModeRef          = useRef<'zoom' | 'julia' | 'both'>('zoom');
  const animSpeedRef         = useRef(0.5);
  const animZoomDirRef       = useRef<'in' | 'out' | 'pingpong'>('pingpong');
  const juliaOrbitRadiusRef  = useRef(0.7);
  const juliaAngleRef        = useRef(0.0);
  const colorCycleRef        = useRef(false);
  const zoomDirRef           = useRef(1); // 1 = zooming in, -1 = zooming out

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

  /** Render the current view immediately via WebGL (GPU mode). */
  const renderGPU = useCallback((skipOrbit = false) => {
    const glCanvas = glCanvasRef.current;
    const renderer = webglRef.current;
    if (!glCanvas || !renderer) return;
    if (zoomLabel.current) zoomLabel.current.textContent = fmtZoom(view.current.zoom);
    const cp = cpRef.current;
    const v  = view.current;
    const params: GLRenderParams = {
      canvasW:      glCanvas.width,
      canvasH:      glCanvas.height,
      centerX:      v.centerX,
      centerY:      v.centerY,
      zoom:         v.zoom,
      maxIter:      cp.maxIterMode === 'auto'
        ? adaptiveMaxIter(v.zoom)
        : cp.maxIterManual,
      paletteId:    cp.paletteId,
      colorSpeed:   cp.colorSpeed,
      colorOffset:  cp.colorOffset,
      invertColors: cp.invertColors,
      juliaMode:    cp.juliaMode,
      juliaRe:      cp.juliaRe,
      juliaIm:      cp.juliaIm,
      hpCenterX:    v.hpCX,
      hpCenterY:    v.hpCY,
      skipOrbit,
    };
    try {
      renderer.render(params);
      // Keep snapshot in sync so context loss can be covered seamlessly.
      const snap = snapshotRef.current;
      if (snap) {
        if (snap.width !== glCanvas.width || snap.height !== glCanvas.height) {
          snap.width  = glCanvas.width;
          snap.height = glCanvas.height;
        }
        snap.getContext('2d')?.drawImage(glCanvas, 0, 0);
      }
    } catch (e) {
      console.error('[Mandelbrot] GPU render error, attempting recovery:', e);
      // Recreate the renderer — the most likely cause is a lost WebGL context.
      webglRef.current?.dispose();
      webglRef.current = null;
      const fresh = createWebGLRenderer(glCanvas);
      if (fresh) {
        webglRef.current = fresh;
        try { fresh.render(params); } catch { /* give up */ }
      }
    }
    updateCrosshair();
  }, [updateCrosshair]);

  /** Render using whichever mode is currently active. */
  const triggerRender = useCallback(() => {
    if (useGPURef.current) renderGPU();
    else startRender();
  }, [renderGPU, startRender]);

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
    const available = detectWebGL();
    setGpuAvailable(available);
    if (available && glCanvasRef.current) {
      try {
        webglRef.current = createWebGLRenderer(glCanvasRef.current);
        renderGPU();
      } catch (e) {
        console.warn('WebGL renderer failed to initialise:', e);
        setGpuAvailable(false);
        useGPURef.current = false;
        setUseGPU(false);
      }
    } else {
      useGPURef.current = false;
      setUseGPU(false);
    }

    const glCanvas = glCanvasRef.current;
    if (glCanvas) {
      const onContextLost = (e: Event) => {
        e.preventDefault(); // allow restoration
        webglRef.current = null;
        if (snapshotRef.current) snapshotRef.current.style.display = 'block';
      };
      const onContextRestored = () => {
        try {
          webglRef.current = createWebGLRenderer(glCanvas);
          renderGPU(false);
          // Hide snapshot on the frame after the new render is presented.
          requestAnimationFrame(() => requestAnimationFrame(() => {
            if (snapshotRef.current) snapshotRef.current.style.display = 'none';
          }));
        } catch (e) {
          console.warn('WebGL context restore failed:', e);
        }
      };
      glCanvas.addEventListener('webglcontextlost', onContextLost);
      glCanvas.addEventListener('webglcontextrestored', onContextRestored);
      return () => {
        cancelRender();
        webglRef.current?.dispose();
        webglRef.current = null;
        glCanvas.removeEventListener('webglcontextlost', onContextLost);
        glCanvas.removeEventListener('webglcontextrestored', onContextRestored);
      };
    }

    return () => {
      cancelRender();
      webglRef.current?.dispose();
      webglRef.current = null;
    };
  }, [cancelRender, renderGPU]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      const w = (rect.width  * dpr) | 0;
      const h = (rect.height * dpr) | 0;
      // Only reset canvas dimensions when they actually change — setting canvas.width/height
      // unconditionally clears the drawing buffer and can cause WebGL context loss on some GPUs.
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width  = w;
        canvas.height = h;
        const back = backRef.current;
        if (back) { back.width = w; back.height = h; }
      }
      const glCanvas = glCanvasRef.current;
      if (glCanvas && (glCanvas.width !== w || glCanvas.height !== h)) {
        if (snapshotRef.current) snapshotRef.current.style.display = 'block';
        glCanvas.width  = w;
        glCanvas.height = h;
      }
      triggerRender();
      if (snapshotRef.current?.style.display === 'block') {
        requestAnimationFrame(() => requestAnimationFrame(() => {
          if (snapshotRef.current) snapshotRef.current.style.display = 'none';
        }));
      }
    });
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [triggerRender]);

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
      const maxZoom = useGPURef.current ? 1e28 : 1e13;
      const newZoom = Math.max(100, Math.min(maxZoom, zoom * factor));
      if (newZoom === zoom) return;

      if (useGPURef.current && newZoom > HP_THRESHOLD) {
        // HP zoom: use decimal.js to update the HP centre strings.
        const prev = view.current;
        const hp = hpZoomTo(
          prev.hpCX ?? String(prev.centerX),
          prev.hpCY ?? String(prev.centerY),
          mx - canvas.width  * 0.5,
          my - canvas.height * 0.5,
          prev.zoom,
          newZoom,
        );
        view.current = {
          centerX: Number(hp.re),
          centerY: Number(hp.im),
          zoom: newZoom,
          hpCX: hp.re,
          hpCY: hp.im,
        };
      } else {
        view.current  = {
          centerX: mouseRe - (mx - canvas.width  * 0.5) / newZoom,
          centerY: mouseIm - (my - canvas.height * 0.5) / newZoom,
          zoom: newZoom,
        };
      }
      if (useGPURef.current) {
        renderGPU(); // instant — no tile delay needed
      } else {
        cancelRender();
        applyZoom(factor, mx, my);
        scheduleRender();
      }
    };
    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', onWheel);
  }, [applyZoom, scheduleRender, cancelRender, renderGPU]);

  // ── drag ───────────────────────────────────────────────────────────────────

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onDown = (e: MouseEvent) => {
      dragRef.current = dragStartRef.current = { x: e.clientX, y: e.clientY };
      canvas.style.cursor = 'grabbing';
      if (!useGPURef.current) cancelRender();
    };
    const onMove = (e: MouseEvent) => {
      if (!dragRef.current) return;
      const rect = canvas.getBoundingClientRect();
      const dpr  = canvas.width / rect.width;
      const dx   = Math.round((e.clientX - dragRef.current.x) * dpr);
      const dy   = Math.round((e.clientY - dragRef.current.y) * dpr);
      dragRef.current = { x: e.clientX, y: e.clientY };
      if (useGPURef.current && view.current.zoom > HP_THRESHOLD) {
        const prev = view.current;
        const hp = hpPan(
          prev.hpCX ?? String(prev.centerX),
          prev.hpCY ?? String(prev.centerY),
          dx, dy,
          prev.zoom,
        );
        view.current = {
          centerX: Number(hp.re),
          centerY: Number(hp.im),
          zoom: prev.zoom,
          hpCX: hp.re,
          hpCY: hp.im,
        };
      } else {
        const { centerX, centerY, zoom } = view.current;
        view.current = { centerX: centerX - dx / zoom, centerY: centerY - dy / zoom, zoom };
      }
      if (useGPURef.current) {
        // Skip orbit recompute only at deep zoom where Decimal arithmetic is slow (~50ms).
        // At normal zoom the float64 orbit is instant and must update every frame.
        renderGPU(view.current.zoom > HP_THRESHOLD);
      } else {
        applyPan(dx, dy);
      }
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
        triggerRender();
        return;
      }
      // GPU already renders at full quality during drag; CPU needs a final full render.
      // Force orbit recompute after drag ends (was skipped during drag for performance).
      if (!useGPURef.current) startRender(true);
      else renderGPU(false);
    };
    canvas.addEventListener('mousedown', onDown);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup',   onUp);
    return () => {
      canvas.removeEventListener('mousedown', onDown);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup',   onUp);
    };
  }, [applyPan, startRender, cancelRender, renderGPU, triggerRender]);

  const reset = useCallback(() => {
    view.current = { ...INITIAL, hpCX: undefined, hpCY: undefined };
    triggerRender();
  }, [triggerRender]);

  // ── animation loop ─────────────────────────────────────────────────────────

  /** Maximum zoom depth for the animation — matches the GPU manual zoom ceiling. */
  const ANIM_MAX_ZOOM = 1e28;

  useEffect(() => {
    if (!animating) return;

    // Auto-enable GPU if available but toggled off
    if (gpuAvailable && !useGPURef.current) {
      useGPURef.current = true;
      setUseGPU(true);
      cancelRender();
    }
    if (!useGPURef.current) return;

    // Enable Julia mode when animation needs it
    if (animModeRef.current !== 'zoom' && !cpRef.current.juliaMode) {
      cpRef.current.juliaMode = true;
      setJuliaMode(true);
      view.current = { centerX: 0, centerY: 0, zoom: 250 };
    }

    let rafId: number;
    let lastT: number | null = null;

    function loop(t: number) {
      if (lastT === null) { lastT = t; rafId = requestAnimationFrame(loop); return; }
      const dt  = Math.min((t - lastT) / 1000, 0.1);
      lastT = t;

      const mode  = animModeRef.current;
      const speed = animSpeedRef.current;

      // Zoom
      if (mode === 'zoom' || mode === 'both') {
        const zoomMode = animZoomDirRef.current;
        if (zoomMode === 'in')       zoomDirRef.current =  1;
        else if (zoomMode === 'out') zoomDirRef.current = -1;
        // 'pingpong': zoomDirRef flips at the limits (below)

        const factor  = Math.pow(2, speed * 0.8 * zoomDirRef.current * dt);
        const newZoom = view.current.zoom * factor;
        if (newZoom > ANIM_MAX_ZOOM && zoomMode === 'pingpong') zoomDirRef.current = -1;
        else if (newZoom < INITIAL.zoom && zoomMode === 'pingpong') zoomDirRef.current = 1;
        view.current = {
          ...view.current,
          zoom: Math.max(INITIAL.zoom, Math.min(ANIM_MAX_ZOOM, newZoom)),
          // Preserve hpCX/hpCY — required for correct rendering above zoom ~1e13.
          // Only clear them once zoom drops back into float64-safe range.
          hpCX: newZoom < 1e13 ? undefined : view.current.hpCX,
          hpCY: newZoom < 1e13 ? undefined : view.current.hpCY,
        };
      }

      // Julia c-parameter orbit (c = radius * e^(i·angle))
      if (mode === 'julia' || mode === 'both') {
        juliaAngleRef.current = (juliaAngleRef.current + speed * 0.6 * dt) % (2 * Math.PI);
        const r  = juliaOrbitRadiusRef.current;
        const re = r * Math.cos(juliaAngleRef.current);
        const im = r * Math.sin(juliaAngleRef.current);
        cpRef.current.juliaRe = re;
        cpRef.current.juliaIm = im;
        setJuliaRe(re);
        setJuliaIm(im);
      }

      // Color cycle
      if (colorCycleRef.current) {
        const newOff = (cpRef.current.colorOffset + speed * 3 * dt) % 16;
        cpRef.current.colorOffset = newOff;
        setColorOffset(newOff);
      }

      renderGPU(false);
      rafId = requestAnimationFrame(loop);
    }

    rafId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafId);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [animating, gpuAvailable, cancelRender, renderGPU]);

  // ── animation handlers ─────────────────────────────────────────────────────

  const handleAnimModeChange = useCallback((v: 'zoom' | 'julia' | 'both') => {
    animModeRef.current = v;
    setAnimMode(v);
    // If switching to a Julia mode, ensure Julia fractal is active
    if (v !== 'zoom' && !cpRef.current.juliaMode) {
      cpRef.current.juliaMode = true;
      setJuliaMode(true);
      view.current = { centerX: 0, centerY: 0, zoom: 250 };
      triggerRender();
    }
  }, [triggerRender]);

  const handleAnimZoomDirChange = useCallback((v: 'in' | 'out' | 'pingpong') => {
    animZoomDirRef.current = v;
    setAnimZoomDir(v);
    // Prime the pingpong direction based on current zoom
    if (v === 'in')  zoomDirRef.current =  1;
    if (v === 'out') zoomDirRef.current = -1;
  }, []);

  const handleAnimSpeedChange = useCallback((v: number) => {
    animSpeedRef.current = v;
    setAnimSpeed(v);
  }, []);

  const handleJuliaOrbitRadiusChange = useCallback((v: number) => {
    juliaOrbitRadiusRef.current = v;
    setJuliaOrbitRadius(v);
  }, []);

  const handleColorCycleChange = useCallback((v: boolean) => {
    colorCycleRef.current = v;
    setColorCycle(v);
  }, []);

  const toggleAnimation = useCallback(() => {
    setAnimating(a => !a);
  }, []);

  // ── color / quality param handlers ─────────────────────────────────────────

  const handlePaletteChange = useCallback((id: PaletteId) => {
    cpRef.current.paletteId = id;
    setPaletteId(id);
    if (useGPURef.current) renderGPU(view.current.zoom > HP_THRESHOLD);
    else startRender();
  }, [renderGPU, startRender]);

  const handleColorSpeedChange = useCallback((v: number) => {
    cpRef.current.colorSpeed = v;
    setColorSpeed(v);
    if (useGPURef.current) renderGPU(view.current.zoom > HP_THRESHOLD);
    else startRender();
  }, [renderGPU, startRender]);

  const handleColorOffsetChange = useCallback((v: number) => {
    cpRef.current.colorOffset = v;
    setColorOffset(v);
    if (useGPURef.current) renderGPU(view.current.zoom > HP_THRESHOLD);
    else startRender();
  }, [renderGPU, startRender]);

  const handleInvertChange = useCallback((v: boolean) => {
    cpRef.current.invertColors = v;
    setInvertColors(v);
    if (useGPURef.current) renderGPU(view.current.zoom > HP_THRESHOLD);
    else startRender();
  }, [renderGPU, startRender]);

  const handleMaxIterModeChange = useCallback((adaptive: boolean) => {
    const mode = adaptive ? 'auto' : 'manual';
    cpRef.current.maxIterMode = mode;
    setMaxIterMode(mode);
    triggerRender();
  }, [triggerRender]);

  const handleMaxIterManualChange = useCallback((v: number) => {
    cpRef.current.maxIterManual = v;
    cpRef.current.maxIterMode = 'manual';
    setMaxIterManual(v);
    setMaxIterMode('manual');
    triggerRender();
  }, [triggerRender]);

  const handleJuliaModeChange = useCallback((on: boolean) => {
    cpRef.current.juliaMode = on;
    setJuliaMode(on);
    if (on) view.current = { centerX: 0, centerY: 0, zoom: 250 };
    else view.current = { ...INITIAL };
    triggerRender();
  }, [triggerRender]);

  const handleJuliaReChange = useCallback((v: number) => {
    cpRef.current.juliaRe = v;
    setJuliaRe(v);
    updateCrosshair();
    triggerRender();
  }, [triggerRender, updateCrosshair]);

  const handleJuliaImChange = useCallback((v: number) => {
    cpRef.current.juliaIm = v;
    setJuliaIm(v);
    updateCrosshair();
    triggerRender();
  }, [triggerRender, updateCrosshair]);

  const goToPreset = useCallback((p: typeof PRESETS[number]) => {
    view.current = { centerX: p.centerX, centerY: p.centerY, zoom: p.zoom };
    triggerRender();
  }, [triggerRender]);

  const handleGPUToggle = useCallback((on: boolean) => {
    useGPURef.current = on;
    setUseGPU(on);
    if (on) {
      cancelRender(); // stop any in-flight CPU render
      renderGPU();
    } else {
      startRender();
    }
  }, [cancelRender, renderGPU, startRender]);

  const saveImage = useCallback(() => {
    // In GPU mode the rendered image lives on the WebGL canvas.
    const canvas = useGPURef.current ? glCanvasRef.current : canvasRef.current;
    if (!canvas) return;
    const a = document.createElement('a');
    a.href     = canvas.toDataURL('image/png');
    a.download = 'mandelbrot.png';
    a.click();
  }, []);

  return (
    <div className={styles.container}>
      {/* GPU canvas sits behind; CPU canvas sits in front and always captures mouse events */}
      <canvas ref={glCanvasRef} className={styles.glCanvas}
        style={{ opacity: useGPU ? 1 : 0 }} />
      {/* Snapshot canvas: covers glCanvas during context loss/resize to prevent black flash */}
      <canvas ref={snapshotRef} className={styles.glCanvas}
        style={{ display: 'none', pointerEvents: 'none' }} />
      <canvas ref={canvasRef} className={styles.canvas}
        style={{ opacity: useGPU ? 0 : 1 }} />

      <div className={styles.sidebar}>
        <div className={styles.sidebarPanels}>
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

        <ControlPanel title="Animation" defaultOpen={false}>
          <ControlGroup>
            <button
              className={`${styles.animBtn} ${animating ? styles.animBtnActive : ''}`}
              type="button"
              onClick={toggleAnimation}
              disabled={!gpuAvailable}
            >
              {animating ? '⏸ Pause' : '▶ Play'}
            </button>
            {!gpuAvailable && (
              <span className={styles.animNote}>Requires GPU rendering</span>
            )}
          </ControlGroup>
          <ControlGroup>
            <SelectControl
              label="Mode"
              value={animMode}
              onChange={handleAnimModeChange}
              options={[
                { value: 'zoom'  as const, label: 'Auto Zoom' },
                { value: 'julia' as const, label: 'Julia Orbit' },
                { value: 'both'  as const, label: 'Zoom + Julia' },
              ]}
            />
            {(animMode === 'zoom' || animMode === 'both') && (
              <SelectControl
                label="Zoom Direction"
                value={animZoomDir}
                onChange={handleAnimZoomDirChange}
                options={[
                  { value: 'pingpong' as const, label: 'Ping-pong' },
                  { value: 'in'       as const, label: 'Zoom in' },
                  { value: 'out'      as const, label: 'Zoom out' },
                ]}
              />
            )}
          </ControlGroup>
          <ControlGroup>
            <Slider
              label="Speed"
              value={animSpeed}
              min={0.1} max={2} step={0.05}
              onChange={handleAnimSpeedChange}
            />
            {(animMode === 'julia' || animMode === 'both') && (
              <Slider
                label="Orbit Radius"
                value={juliaOrbitRadius}
                min={0.1} max={1.5} step={0.01}
                onChange={handleJuliaOrbitRadiusChange}
              />
            )}
          </ControlGroup>
          <ControlGroup>
            <Toggle
              label="Cycle Colors"
              value={colorCycle}
              onChange={handleColorCycleChange}
              description="Animate color offset"
            />
          </ControlGroup>
        </ControlPanel>

        <ControlPanel title="Quality" defaultOpen={false}>
          <ControlGroup>
            <Toggle
              label="GPU Rendering"
              value={useGPU}
              onChange={handleGPUToggle}
              disabled={!gpuAvailable}
              description={gpuAvailable ? 'WebGL2 · instant re-render' : 'WebGL2 unavailable'}
            />
          </ControlGroup>
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
        </div>{/* end sidebarPanels */}

        <div className={styles.sidebarActions}>
        <button className={styles.saveBtn} type="button" onClick={saveImage}>
          ↓ Save PNG
        </button>

        <button className={styles.resetBtn} type="button" onClick={reset}>
          Reset View
        </button>
        </div>
      </div>

      <div className={styles.hud}>
        <div className={styles.hudLeft}>
          <span className={styles.hudTitle}>
            {juliaMode ? 'Julia Set' : 'Mandelbrot Set'}
          </span>
          <span ref={zoomLabel} className={styles.hudZoom}>1.0&times;</span>
          {animating && (
            <span className={styles.hudAnim}>▶ animating</span>
          )}
        </div>
        <div className={styles.hudRight}>
          <span className={styles.hudHint}>
            {animating
              ? 'click Animation panel to pause'
              : juliaMode
                ? 'click to pick c · scroll to zoom · drag to pan'
                : 'scroll to zoom · drag to pan'}
          </span>
        </div>
      </div>
    </div>
  );
}
