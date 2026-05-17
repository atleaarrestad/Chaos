import { useState, useEffect, useRef, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Slider, SelectControl,
  ControlPanel, ControlGroup, SimControls,
} from '@/components/Controls';
import { InfoDialog } from '@/components/InfoDialog';
import { useFullscreen } from '@/hooks/useFullscreen';
import { getNumParam, getStrParam, useShareUrl } from '@/hooks/useUrlParams';
import ExportDialog from '../../components/ExportDialog/ExportDialog';
import { downloadCanvas } from '../../lib/exportImage';
import {
  PRESETS, COLOR_PALETTES, BG_COLOR, chaosStep, ifsToScreen,
  renderIntoImageData, makeDarkImageData,
  type ColorSchemeId, type RenderViewport,
} from './barnsley-fern';
import styles from './BarnsleyFern.module.css';

// ─── Constants ────────────────────────────────────────────────────────────────

const DEFAULT_PRESET = 0;
const DEFAULT_COLOR: ColorSchemeId = 'fern';
const DEFAULT_ITERS = 30_000;
const MAX_POINTS = 15_000_000;

const COLOR_OPTS = [
  { value: 'fern',   label: 'Fern'   },
  { value: 'heat',   label: 'Heat'   },
  { value: 'frost',  label: 'Frost'  },
  { value: 'mono',   label: 'Mono'   },
  { value: 'plasma', label: 'Plasma' },
];

// ─── Component ────────────────────────────────────────────────────────────────

export default function BarnsleyFern() {
  // ─── URL params ─────────────────────────────────────────────────────────
  const [searchParams] = useSearchParams();
  const initialPreset = Math.min(
    Math.max(0, Math.round(getNumParam(searchParams, 'p', DEFAULT_PRESET))),
    PRESETS.length - 1,
  );
  const rawColor = getStrParam(searchParams, 'c', DEFAULT_COLOR);
  const initialColor = (rawColor in COLOR_PALETTES ? rawColor : DEFAULT_COLOR) as ColorSchemeId;

  // ─── State ──────────────────────────────────────────────────────────────
  const [presetIdx,    setPresetIdx]    = useState(initialPreset);
  const [colorScheme,  setColorScheme]  = useState<ColorSchemeId>(initialColor);
  const [itersPerFrame,setItersPerFrame]= useState(DEFAULT_ITERS);
  const [building,     setBuilding]     = useState(true);
  const [zoom,         setZoom]         = useState(1);
  const [panX,         setPanX]         = useState(0);
  const [panY,         setPanY]         = useState(0);
  const [totalPts,     setTotalPts]     = useState(0);
  const [showInfo,     setShowInfo]     = useState(false);
  const [showExport,   setShowExport]   = useState(false);
  const [copied,       setCopied]       = useState(false);

  // ─── Refs ────────────────────────────────────────────────────────────────
  const canvasRef      = useRef<HTMLCanvasElement>(null);
  const containerRef   = useRef<HTMLDivElement>(null);
  const rafRef         = useRef(0);
  const needsClearRef  = useRef(false);
  const imgDataRef     = useRef<ImageData | null>(null);
  const fernXRef       = useRef(0);
  const fernYRef       = useRef(0);
  const totalPtsRef    = useRef(0);
  const pendingSizeRef = useRef<{ w: number; h: number } | null>(null);
  const dragRef        = useRef<{ x: number; y: number } | null>(null);
  const copiedTimerRef = useRef<number | null>(null);
  const ctxRef         = useRef<CanvasRenderingContext2D | null>(null);
  const clearTimerRef  = useRef<number | null>(null);

  // Mirror mutable state into a ref so the RAF loop always sees fresh values.
  const pRef = useRef({
    preset:       PRESETS[initialPreset],
    colorScheme:  initialColor,
    itersPerFrame: DEFAULT_ITERS,
    building:     true,
    zoom: 1, panX: 0, panY: 0,
  });
  pRef.current.preset        = PRESETS[presetIdx];
  pRef.current.colorScheme   = colorScheme;
  pRef.current.itersPerFrame = itersPerFrame;
  pRef.current.building      = building;
  pRef.current.zoom          = zoom;
  pRef.current.panX          = panX;
  pRef.current.panY          = panY;

  const { isFullscreen, toggleFullscreen } = useFullscreen(containerRef);
  const { shareUrl } = useShareUrl();

  // Trigger a full clear when the preset or colour scheme changes.
  useEffect(() => { needsClearRef.current = true; }, [presetIdx, colorScheme]);

  // ─── RAF render loop ─────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let stopped = false;
    let ctx: CanvasRenderingContext2D | null = null;

    function getCtx(): CanvasRenderingContext2D | null {
      if (ctx) return ctx;
      ctx = canvas!.getContext('2d', { willReadFrequently: true });
      ctxRef.current = ctx;
      return ctx;
    }

    function clearCanvas() {
      const c = getCtx();
      if (!c || !canvas) return;
      imgDataRef.current = makeDarkImageData(c);
      c.putImageData(imgDataRef.current, 0, 0);
      fernXRef.current = 0;
      fernYRef.current = 0;
      totalPtsRef.current = 0;
      setTotalPts(0);
    }

    function frame() {
      if (stopped) return;

      // Apply pending canvas resize atomically.
      const pending = pendingSizeRef.current;
      if (pending) {
        pendingSizeRef.current = null;
        if (canvas!.width !== pending.w || canvas!.height !== pending.h) {
          canvas!.width  = pending.w;
          canvas!.height = pending.h;
          ctx = null;
          imgDataRef.current = null;
        }
        needsClearRef.current = true;
      }

      if (!getCtx()) { rafRef.current = requestAnimationFrame(frame); return; }

      if (needsClearRef.current) {
        needsClearRef.current = false;
        clearCanvas();
      }

      const { preset, colorScheme: cs, itersPerFrame: N, building } = pRef.current;

      // While paused, or after reaching max density, just idle.
      if (!building || totalPtsRef.current >= MAX_POINTS) {
        rafRef.current = requestAnimationFrame(frame);
        return;
      }

      if (!imgDataRef.current) clearCanvas();

      const img    = imgDataRef.current!;
      const data   = img.data;
      const W      = canvas!.width;
      const H      = canvas!.height;
      const palette = COLOR_PALETTES[cs];
      const vp: RenderViewport = {
        zoom: pRef.current.zoom,
        panX: pRef.current.panX,
        panY: pRef.current.panY,
      };

      let x = fernXRef.current;
      let y = fernYRef.current;

      for (let i = 0; i < N; i++) {
        const [nx, ny, ti] = chaosStep(x, y, preset);
        x = nx; y = ny;
        const [sx, sy] = ifsToScreen(x, y, preset, W, H, vp);
        if (sx >= 0 && sx < W && sy >= 0 && sy < H) {
          const idx = (sy * W + sx) << 2;
          const [r, g, b] = palette[ti % palette.length];
          data[idx] = r; data[idx + 1] = g; data[idx + 2] = b;
        }
      }

      fernXRef.current = x;
      fernYRef.current = y;
      totalPtsRef.current += N;

      ctx!.putImageData(img, 0, 0);

      // Update the displayed counter every ~500k points to avoid React overhead.
      if (totalPtsRef.current % 500_000 < N) {
        setTotalPts(totalPtsRef.current);
      }

      rafRef.current = requestAnimationFrame(frame);
    }

    rafRef.current = requestAnimationFrame(frame);
    return () => {
      stopped = true;
      cancelAnimationFrame(rafRef.current);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ─── Resize observer ─────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const obs = new ResizeObserver(entries => {
      for (const e of entries) {
        const { width, height } = e.contentRect;
        const dpr = window.devicePixelRatio || 1;
        const w = Math.round(width * dpr);
        const h = Math.round(height * dpr);
        if (w > 0 && h > 0) pendingSizeRef.current = { w, h };
      }
    });
    obs.observe(canvas.parentElement!);
    return () => obs.disconnect();
  }, []);

  // ─── Pan & zoom ──────────────────────────────────────────────────────────
  const onPointerDown = useCallback((e: React.PointerEvent) => {
    dragRef.current = { x: e.clientX, y: e.clientY };
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }, []);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (!dragRef.current) return;
    const dx = e.clientX - dragRef.current.x;
    const dy = e.clientY - dragRef.current.y;
    dragRef.current = { x: e.clientX, y: e.clientY };

    const canvas = canvasRef.current;
    const ctx = ctxRef.current;
    const rect = (e.target as HTMLElement).getBoundingClientRect();

    // ifsToScreen: sy = H/2 - … - panY*H/2  →  positive panY moves the fractal UP.
    // Dragging down (dy > 0) should move the fractal down, so we subtract dy.
    const newPanX = pRef.current.panX + (2 * dx) / rect.width;
    const newPanY = pRef.current.panY - (2 * dy) / rect.height;

    // Shift accumulated pixels by the drag delta (canvas-pixel units) instead of
    // clearing, so the rendered points are preserved while new ones fill in.
    if (canvas && ctx && imgDataRef.current) {
      const W = canvas.width;
      const H = canvas.height;
      const cdx = dx * (W / rect.width);
      const cdy = dy * (H / rect.height);
      const tmp = document.createElement('canvas');
      tmp.width = W; tmp.height = H;
      tmp.getContext('2d')!.putImageData(imgDataRef.current, 0, 0);
      const [br, bg, bb] = BG_COLOR;
      ctx.fillStyle = `rgb(${br},${bg},${bb})`;
      ctx.fillRect(0, 0, W, H);
      ctx.save();
      ctx.setTransform(1, 0, 0, 1, cdx, cdy);
      ctx.drawImage(tmp, 0, 0);
      ctx.restore();
      imgDataRef.current = ctx.getImageData(0, 0, W, H);
    }

    pRef.current.panX = newPanX;
    pRef.current.panY = newPanY;
    setPanX(newPanX);
    setPanY(newPanY);
  }, []);

  const onPointerUp = useCallback(() => { dragRef.current = null; }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const handler = (e: WheelEvent) => {
      e.preventDefault();
      const canvas = canvasRef.current;
      const ctx = ctxRef.current;
      if (!canvas || !ctx) return;

      const W = canvas.width;
      const H = canvas.height;
      // Mouse in canvas pixel space (accounting for DPR)
      const rect = canvas.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * (W / rect.width);
      const my = (e.clientY - rect.top)  * (H / rect.height);

      const factor = e.deltaY > 0 ? 1 / 1.15 : 1.15;
      const oldZoom = pRef.current.zoom;
      const newZoom = Math.max(0.1, Math.min(20, oldZoom * factor));
      const f = newZoom / oldZoom;

      // Zoom-to-mouse: adjust pan so the IFS point under the cursor stays fixed.
      // In "half-canvas" units: nx = mx/(W/2) - 1, ny = my/(H/2) - 1
      const nx = mx / (W / 2) - 1;
      const ny = my / (H / 2) - 1;
      const newPanX = nx * (1 - f) + pRef.current.panX * f;
      const newPanY = ny * (f - 1) + pRef.current.panY * f;

      // Zoom-in: pixel-transform preview while scrolling, sharp redraw 250 ms after stopping.
      // Zoom-out: clear immediately so old pixels don't layer over the new viewport.
      if (f > 1 && imgDataRef.current) {
        const tmp = document.createElement('canvas');
        tmp.width = W; tmp.height = H;
        tmp.getContext('2d')!.putImageData(imgDataRef.current, 0, 0);
        const [br, bg, bb] = BG_COLOR;
        ctx.fillStyle = `rgb(${br},${bg},${bb})`;
        ctx.fillRect(0, 0, W, H);
        ctx.save();
        ctx.setTransform(f, 0, 0, f, (1 - f) * mx, (1 - f) * my);
        ctx.drawImage(tmp, 0, 0);
        ctx.restore();
        imgDataRef.current = ctx.getImageData(0, 0, W, H);

        if (clearTimerRef.current !== null) window.clearTimeout(clearTimerRef.current);
        clearTimerRef.current = window.setTimeout(() => {
          needsClearRef.current = true;
          clearTimerRef.current = null;
        }, 250);
      } else {
        // Cancel any in-flight debounce and clear immediately.
        if (clearTimerRef.current !== null) {
          window.clearTimeout(clearTimerRef.current);
          clearTimerRef.current = null;
        }
        needsClearRef.current = true;
      }

      // Update pRef immediately — the RAF loop reads these on its very next frame,
      // before React has a chance to re-render, preventing any stale-viewport frame.
      pRef.current.zoom = newZoom;
      pRef.current.panX = newPanX;
      pRef.current.panY = newPanY;
      setZoom(newZoom);
      setPanX(newPanX);
      setPanY(newPanY);
    };
    el.addEventListener('wheel', handler, { passive: false });
    return () => el.removeEventListener('wheel', handler);
  }, []);

  // ─── Actions ─────────────────────────────────────────────────────────────
  const applyPreset = useCallback((idx: number) => {
    setPresetIdx(idx);
    pRef.current.zoom = 1; pRef.current.panX = 0; pRef.current.panY = 0;
    setZoom(1); setPanX(0); setPanY(0);
  }, []);

  const reset = useCallback(() => {
    pRef.current.zoom = 1; pRef.current.panX = 0; pRef.current.panY = 0;
    setZoom(1); setPanX(0); setPanY(0);
    needsClearRef.current = true;
  }, []);

  const handleShare = useCallback(() => {
    shareUrl({ p: presetIdx, c: colorScheme });
    setCopied(true);
    if (copiedTimerRef.current !== null) window.clearTimeout(copiedTimerRef.current);
    copiedTimerRef.current = window.setTimeout(() => setCopied(false), 2000);
  }, [presetIdx, colorScheme, shareUrl]);

  useEffect(() => () => {
    if (copiedTimerRef.current !== null) window.clearTimeout(copiedTimerRef.current);
    if (clearTimerRef.current  !== null) window.clearTimeout(clearTimerRef.current);
  }, []);

  // ─── Keyboard shortcuts ──────────────────────────────────────────────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.code === 'Space') { e.preventDefault(); setBuilding(b => !b); }
      if (e.code === 'KeyR')  { e.preventDefault(); reset(); }
      if (e.code === 'KeyF')  { e.preventDefault(); toggleFullscreen(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [reset, toggleFullscreen]);

  // ─── Export ──────────────────────────────────────────────────────────────
  const handleExport = useCallback(({ width, height, format }: { width: number; height: number; format: Parameters<typeof downloadCanvas>[1] }) => {
    const off = document.createElement('canvas');
    off.width = width; off.height = height;
    const ctx = off.getContext('2d')!;

    const img = ctx.createImageData(width, height);
    const d = img.data;
    const [br, bg, bb] = [7, 7, 18];
    for (let i = 0; i < d.length; i += 4) {
      d[i] = br; d[i + 1] = bg; d[i + 2] = bb; d[i + 3] = 255;
    }

    const N = Math.min(10_000_000, Math.max(2_000_000, width * height));
    renderIntoImageData(img, pRef.current.preset, COLOR_PALETTES[pRef.current.colorScheme], {
      zoom: pRef.current.zoom, panX: pRef.current.panX, panY: pRef.current.panY,
    }, N);

    ctx.putImageData(img, 0, 0);
    downloadCanvas(off, format, 'barnsley-fern');
  }, []);

  // ─── Render ──────────────────────────────────────────────────────────────
  const preset = PRESETS[presetIdx];
  const ptLabel = totalPts >= 1_000_000
    ? `${(totalPts / 1_000_000).toFixed(1)}M pts`
    : `${(totalPts / 1_000).toFixed(0)}k pts`;

  return (
    <div ref={containerRef} className={styles.container}>
      <canvas
        ref={canvasRef}
        className={styles.canvas}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      />

      {/* ─── Sidebar ──────────────────────────────────────────────────── */}
      <div className={styles.sidebar}>
        <div className={styles.sidebarPanels}>

          <ControlPanel title="Presets">
            <div className={styles.presetGrid}>
              {PRESETS.map((p, i) => (
                <button
                  key={p.id}
                  className={[styles.presetBtn, presetIdx === i ? styles.presetBtnActive : ''].join(' ')}
                  onClick={() => applyPreset(i)}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </ControlPanel>

          <ControlPanel title="Appearance">
            <ControlGroup>
              <SelectControl
                label="Color"
                value={colorScheme}
                options={COLOR_OPTS}
                onChange={v => setColorScheme(v as ColorSchemeId)}
              />
            </ControlGroup>
          </ControlPanel>

          <ControlPanel title="Performance" defaultOpen={false}>
            <ControlGroup>
              <Slider
                label="Speed"
                value={itersPerFrame}
                min={1_000} max={100_000} step={1_000}
                onChange={setItersPerFrame}
              />
            </ControlGroup>
          </ControlPanel>

        </div>

        <div className={styles.sidebarActions}>
          <SimControls
            running={building}
            onToggle={() => setBuilding(b => !b)}
            onReset={reset}
            onExport={() => setShowExport(true)}
          />
        </div>
      </div>

      {/* ─── Export dialog ────────────────────────────────────────────── */}
      {showExport && (
        <ExportDialog
          onClose={() => setShowExport(false)}
          onDownload={({ width, height, format }) => {
            handleExport({ width, height, format });
            setShowExport(false);
          }}
        />
      )}

      {/* ─── HUD ──────────────────────────────────────────────────────── */}
      <div className={styles.hud}>
        <div className={styles.hudLeft}>
          <span className={styles.hudTitle}>{preset.name}</span>
          <span className={styles.hudSub}>
            {preset.transforms.length} maps · {ptLabel}
          </span>
        </div>
        <div className={styles.hudRight}>
          <span className={styles.hudHint}>scroll to zoom · drag to pan</span>
          <button className={styles.hudBtn} onClick={handleShare} title="Copy shareable link">
            {copied ? '✓' : '⎘'}
          </button>
          <button
            className={styles.hudBtn}
            onClick={toggleFullscreen}
            title={isFullscreen ? 'Exit fullscreen (F)' : 'Fullscreen (F)'}
          >
            {isFullscreen ? '⤡' : '⤢'}
          </button>
          <button className={styles.infoBtn} onClick={() => setShowInfo(true)} title="About Barnsley Fern">ⓘ</button>
        </div>
      </div>

      {/* ─── Info dialog ──────────────────────────────────────────────── */}
      {showInfo && (
        <InfoDialog title="Iterated Function Systems" onClose={() => setShowInfo(false)}>
          <p>
            An Iterated Function System (IFS) is a finite set of contraction mappings.
            The <em>chaos game</em> produces the attractor: start anywhere, then repeatedly
            apply a randomly chosen mapping — the orbit converges to a fractal regardless of the start point.
          </p>
          <h3>Barnsley Fern</h3>
          <p>
            Introduced by Michael Barnsley in 1988. Four affine transformations encode
            the self-similar structure of a Black Spleenwort fern with just 28 numbers.
            85% of iterations grow the body; the remaining 15% form the stem and leaflets.
          </p>
          <h3>Other presets</h3>
          <ul>
            <li><strong>Sierpiński:</strong> three equal contractions to triangle vertices</li>
            <li><strong>Dragon:</strong> Heighway dragon — two 45° rotations</li>
            <li><strong>Lévy:</strong> the Lévy C curve, a self-similar fractal cloud</li>
            <li><strong>Tree:</strong> symmetric binary branching at 45°</li>
          </ul>
          <h3>Controls</h3>
          <ul>
            <li><strong>Scroll:</strong> zoom</li>
            <li><strong>Drag:</strong> pan</li>
            <li><strong>Space:</strong> pause / resume build</li>
            <li><strong>R:</strong> reset view &amp; restart</li>
            <li><strong>F:</strong> fullscreen</li>
          </ul>
        </InfoDialog>
      )}
    </div>
  );
}
