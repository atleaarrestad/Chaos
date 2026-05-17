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
  PRESETS, COLOR_PALETTES, chaosStep, ifsToScreen,
  renderIntoImageData, makeDarkImageData, makeCountBuffers, countsToImageData,
  type ColorSchemeId, type RenderViewport,
} from './barnsley-fern';
import styles from './BarnsleyFern.module.css';

// ─── Constants ────────────────────────────────────────────────────────────────

const DEFAULT_PRESET = 0;

const PRESET_INFO: Record<string, { category: string; maps: number; rules: string[] }> = {
  barnsley: {
    category: 'IFS Fractal',
    maps: 4,
    rules: [
      'Drop a dot anywhere. Each step randomly picks one of 4 rules. Plot it. Repeat millions of times.',
      'Rule 1 (85%): shrink and tilt the dot slightly upward. Draws the leafy body.',
      'Rule 2 (7%): rotate and shrink the dot toward the left. Draws the left leaflet.',
      'Rule 3 (7%): rotate and shrink the dot toward the right. Draws the right leaflet.',
      'Rule 4 (1%): flatten the dot straight down to the base. Draws the stem.',
    ],
  },
  sierpinski: {
    category: 'IFS Fractal',
    maps: 3,
    rules: [
      'Drop a dot anywhere. Each step picks a random corner and jumps halfway there. Plot it. Repeat.',
      'Rule 1 (33%): jump halfway toward the bottom-left corner',
      'Rule 2 (33%): jump halfway toward the bottom-right corner',
      'Rule 3 (33%): jump halfway toward the top corner',
      'The triangular holes appear by themselves: no rule ever lands inside them',
    ],
  },
  dragon: {
    category: 'IFS Fractal',
    maps: 2,
    rules: [
      'Imagine folding a strip of paper in half over and over, always the same direction',
      'Then unfold it so every crease is a 90 degree angle',
    ],
  },
  levy: {
    category: 'IFS Fractal',
    maps: 2,
    rules: [
      'Drop a dot anywhere. Each step randomly picks one of 2 rules. Plot it. Repeat millions of times.',
      'Rule 1 (50%): rotate the dot 45 degrees left and move it halfway toward the bottom-left corner',
      'Rule 2 (50%): rotate the dot 45 degrees right and move it halfway toward the bottom-right corner',
    ],
  },
  tree: {
    category: 'IFS Fractal',
    maps: 3,
    rules: [
      'Drop a dot anywhere. Each step randomly picks one of 3 rules. Plot it. Repeat millions of times.',
      'Rule 1 (5%): pull the dot straight down to the center line and halve its height. Builds the trunk.',
      'Rule 2 (47.5%): rotate the dot 45 degrees left and shrink it slightly. Grows the left branches.',
      'Rule 3 (47.5%): rotate the dot 45 degrees right and shrink it slightly. Grows the right branches.',
    ],
  },
};
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
  const [showInfo,       setShowInfo]       = useState(false);
  const [showExport,     setShowExport]     = useState(false);
  const [copied,         setCopied]         = useState(false);
  const [infoCollapsed,  setInfoCollapsed]  = useState(false);

  // ─── Refs ────────────────────────────────────────────────────────────────
  const canvasRef       = useRef<HTMLCanvasElement>(null);
  const containerRef    = useRef<HTMLDivElement>(null);
  const rafRef          = useRef(0);
  const needsClearRef   = useRef(false);
  const needsRedrawRef  = useRef(false);
  const imgDataRef      = useRef<ImageData | null>(null);
  const countsRef       = useRef<Uint32Array | null>(null);
  const transformIdxRef = useRef<Uint8Array | null>(null);
  const maxCountRef     = useRef(0);
  const fernXRef        = useRef(0);
  const fernYRef        = useRef(0);
  const totalPtsRef     = useRef(0);
  const pendingSizeRef  = useRef<{ w: number; h: number } | null>(null);
  const dragRef         = useRef<{ x: number; y: number } | null>(null);
  const draggingRef     = useRef(false);
  const copiedTimerRef  = useRef<number | null>(null);
  const ctxRef          = useRef<CanvasRenderingContext2D | null>(null);

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

  // Trigger a full clear when the preset changes; color-only change just redraws from counts.
  useEffect(() => { needsClearRef.current = true; }, [presetIdx]);
  useEffect(() => { needsRedrawRef.current = true; }, [colorScheme]);

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
      const W = canvas.width;
      const H = canvas.height;
      const { counts, txIdx } = makeCountBuffers(W, H);
      countsRef.current       = counts;
      transformIdxRef.current = txIdx;
      maxCountRef.current     = 0;
      imgDataRef.current = makeDarkImageData(c);
      c.putImageData(imgDataRef.current, 0, 0);
      fernXRef.current    = 0;
      fernYRef.current    = 0;
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
          imgDataRef.current      = null;
          countsRef.current       = null;
          transformIdxRef.current = null;
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
        if (needsRedrawRef.current && imgDataRef.current && countsRef.current && transformIdxRef.current) {
          needsRedrawRef.current = false;
          countsToImageData(imgDataRef.current, countsRef.current, transformIdxRef.current, COLOR_PALETTES[pRef.current.colorScheme], maxCountRef.current);
          ctx!.putImageData(imgDataRef.current, 0, 0);
        }
        rafRef.current = requestAnimationFrame(frame);
        return;
      }

      if (!imgDataRef.current || !countsRef.current || !transformIdxRef.current) clearCanvas();

      const img    = imgDataRef.current!;
      const counts = countsRef.current!;
      const txIdx  = transformIdxRef.current!;
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
      let mc = maxCountRef.current;

      for (let i = 0; i < N; i++) {
        const [nx, ny, ti] = chaosStep(x, y, preset);
        x = nx; y = ny;
        const [sx, sy] = ifsToScreen(x, y, preset, W, H, vp);
        if (sx >= 0 && sx < W && sy >= 0 && sy < H) {
          const pidx = sy * W + sx;
          const c = ++counts[pidx];
          if (c > mc) mc = c;
          txIdx[pidx] = ti;
        }
      }

      fernXRef.current    = x;
      fernYRef.current    = y;
      maxCountRef.current = mc;
      totalPtsRef.current += N;

      // Rebuild the image from density counts and flush to canvas.
      // Skip during drag — the viewport is changing; a clean rebuild follows on pointer-up.
      if (!draggingRef.current) {
        needsRedrawRef.current = false;
        countsToImageData(img, counts, txIdx, palette, mc);
        ctx!.putImageData(img, 0, 0);
      }

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
    draggingRef.current = true;
    dragRef.current = { x: e.clientX, y: e.clientY };
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }, []);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (!dragRef.current) return;
    const dx = e.clientX - dragRef.current.x;
    const dy = e.clientY - dragRef.current.y;
    dragRef.current = { x: e.clientX, y: e.clientY };

    const rect = (e.target as HTMLElement).getBoundingClientRect();
    const newPanX = pRef.current.panX + (2 * dx) / rect.width;
    const newPanY = pRef.current.panY - (2 * dy) / rect.height;

    pRef.current.panX = newPanX;
    pRef.current.panY = newPanY;
    setPanX(newPanX);
    setPanY(newPanY);
  }, []);

  const onPointerUp = useCallback(() => {
    if (dragRef.current) needsClearRef.current = true;
    draggingRef.current = false;
    dragRef.current = null;
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const handler = (e: WheelEvent) => {
      e.preventDefault();
      const canvas = canvasRef.current;
      if (!canvas) return;

      const W = canvas.width;
      const H = canvas.height;
      const rect = canvas.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * (W / rect.width);
      const my = (e.clientY - rect.top)  * (H / rect.height);

      const factor = e.deltaY > 0 ? 1 / 1.15 : 1.15;
      const oldZoom = pRef.current.zoom;
      const newZoom = Math.max(0.1, Math.min(20, oldZoom * factor));
      const f = newZoom / oldZoom;

      // Zoom-to-mouse: adjust pan so the IFS point under the cursor stays fixed.
      const nx = mx / (W / 2) - 1;
      const ny = my / (H / 2) - 1;
      const newPanX = nx * (1 - f) + pRef.current.panX * f;
      const newPanY = ny * (f - 1) + pRef.current.panY * f;

      needsClearRef.current = true;

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

      {/* ─── Info overlay ─────────────────────────────────────────────── */}
      {(() => {
        const info = PRESET_INFO[preset.id];
        return info ? (
          <div className={styles.infoOverlay}>
            <div className={styles.infoOverlayHeader}>
              <div className={styles.infoOverlayName}>
                <span className={styles.infoOverlayDot} />
                {preset.name}
              </div>
              <button
                className={styles.infoOverlayToggle}
                onClick={() => setInfoCollapsed(v => !v)}
                aria-label={infoCollapsed ? 'Expand' : 'Collapse'}
              >
                {infoCollapsed ? '▸' : '▾'}
              </button>
            </div>
            {!infoCollapsed && (
              <>
                <div className={styles.infoOverlayMeta}>
                  <span className={styles.categoryBadge}>{info.category}</span>
                  <span className={styles.notationBadge}>{info.maps} maps</span>
                </div>
                <ul className={styles.infoOverlayRules}>
                  {info.rules.map((rule, i) => (
                    <li key={i} className={styles.infoOverlayRule}>{rule}</li>
                  ))}
                </ul>
              </>
            )}
          </div>
        ) : null;
      })()}

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
