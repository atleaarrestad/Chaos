import { useState, useEffect, useRef, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Slider, Toggle, SelectControl,
  ControlPanel, ControlGroup, SimControls,
} from '@/components/Controls';
import { InfoDialog } from '@/components/InfoDialog';
import { useFullscreen } from '@/hooks/useFullscreen';
import { getNumParam, getStrParam, useShareUrl } from '@/hooks/useUrlParams';
import ExportDialog from '../../components/ExportDialog/ExportDialog';
import { exportImage } from '../../lib/exportImage';
import {
  detectWebGL,
  createWebGLRenderer,
  type WebGLBifurcationRenderer,
  type ColorSchemeId,
} from './bifurcation-webgl';
import styles from './Bifurcation.module.css';

// ─── Types ────────────────────────────────────────────────────────────────────

type ColorScheme = ColorSchemeId;

interface ZoomState { rMin: number; rMax: number; yMin: number; yMax: number; }

// ─── Constants ────────────────────────────────────────────────────────────────

const BG_R = 7, BG_G = 7, BG_B = 18; // #070712

interface Preset {
  label: string;
  rMin: number;
  rMax: number;
  desc: string;
}

const PRESETS: Preset[] = [
  { label: 'Full',     rMin: 2.5,    rMax: 4.0,    desc: 'r ∈ [2.5, 4], full bifurcation diagram'         },
  { label: 'Doubling', rMin: 2.8,    rMax: 3.6,    desc: 'r ∈ [2.8, 3.6], period-doubling cascade'        },
  { label: 'Chaos',    rMin: 3.5,    rMax: 4.0,    desc: 'r ∈ [3.5, 4], onset of chaos'                   },
  { label: '3-cycle',  rMin: 3.82,   rMax: 3.88,   desc: 'r ∈ [3.82, 3.88], period-3 window'              },
  { label: 'δ point',  rMin: 3.54,   rMax: 3.57,   desc: 'r ≈ 3.569, Feigenbaum accumulation point'       },
  { label: 'Deep',     rMin: 3.856,  rMax: 3.862,  desc: 'r ∈ [3.856, 3.862], deep self-similar structure'},
];

// ─── Axis helpers ─────────────────────────────────────────────────────────────

function niceTicks(min: number, max: number, target = 5): number[] {
  const range = max - min;
  if (range <= 0 || !isFinite(range)) return [];
  const raw = range / target;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = norm < 1.5 ? mag : norm < 3.5 ? 2 * mag : norm < 7.5 ? 5 * mag : 10 * mag;
  if (!isFinite(step) || step <= 0) return [];
  const first = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let i = 0; i < 20; i++) {
    const v = Math.round((first + i * step) / step) * step;
    if (v > max + step * 1e-9) break;
    ticks.push(v);
  }
  return ticks;
}

function fmtTick(v: number, step: number): string {
  const d = Math.max(0, Math.ceil(-Math.log10(step)));
  return v.toFixed(d);
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function Bifurcation() {
  // Canvas refs — both always in DOM, one visible at a time
  const containerRef = useRef<HTMLDivElement>(null);
  const glCanvasRef  = useRef<HTMLCanvasElement>(null);
  const cpuCanvasRef = useRef<HTMLCanvasElement>(null);
  const sidebarRef   = useRef<HTMLDivElement>(null);
  const layerRef     = useRef<HTMLDivElement>(null);

  const rafRef   = useRef<number>(0);
  const dirtyRef = useRef(true);
  const glRef    = useRef<WebGLBifurcationRenderer | null>(null);

  // URL params
  const [searchParams] = useSearchParams();
  const initialRMin = getNumParam(searchParams, 'r0', 2.5);
  const initialRMax = getNumParam(searchParams, 'r1', 4.0);
  const initialColorScheme = (() => {
    const v = getStrParam(searchParams, 'c', 'cyan');
    return (v === 'cyan' || v === 'heat' || v === 'plasma' || v === 'mono') ? v as ColorScheme : 'cyan';
  })();

  // ─── State ──────────────────────────────────────────────────────────────

  const [rMin,         setRMin]         = useState(initialRMin);
  const [rMax,         setRMax]         = useState(initialRMax);
  const [yMin,         setYMin]         = useState(0);
  const [yMax,         setYMax]         = useState(1);
  const [iterations,   setIterations]   = useState(300);
  const [burnin,       setBurnin]       = useState(200);
  const [colorScheme,  setColorScheme]  = useState<ColorScheme>(initialColorScheme);
  const [logScale,     setLogScale]     = useState(true);
  const [activePreset, setActivePreset] = useState<number | null>(0);
  const [showInfo, setShowInfo] = useState(false);
  const [showExport, setShowExport] = useState(false);
  const [gpuAvailable, setGpuAvailable] = useState(false);
  const [useGPU,       setUseGPU]       = useState(false);
  const [zoomHistory,  setZoomHistory]  = useState<ZoomState[]>([]);
  const [hoverR,       setHoverR]       = useState<number | null>(null);
  const [hoverX,       setHoverX]       = useState<number | null>(null);
  const [sidebarOpen,  setSidebarOpen]  = useState(true);
  const [copied, setCopied] = useState(false);
  const copiedTimeoutRef = useRef<number | null>(null);

  const { isFullscreen, toggleFullscreen } = useFullscreen(containerRef);
  const { shareUrl } = useShareUrl();

  // Drag-to-zoom selection rect (CSS pixels relative to the interaction layer)
  const dragStartRef = useRef<{ x: number; y: number } | null>(null);
  const [selRect, setSelRect] = useState<{ x: number; y: number; w: number; h: number } | null>(null);

  // Mirror params to ref so draw loop always reads fresh values without re-creating draw
  const pRef = useRef({ rMin, rMax, yMin, yMax, iterations, burnin, colorScheme, logScale, useGPU });
  useEffect(() => {
    pRef.current = { rMin, rMax, yMin, yMax, iterations, burnin, colorScheme, logScale, useGPU };
    dirtyRef.current = true;
  });

  // ─── WebGL init ──────────────────────────────────────────────────────────

  useEffect(() => {
    if (!detectWebGL()) return;
    setGpuAvailable(true);
    setUseGPU(true);
    const renderer = createWebGLRenderer(glCanvasRef.current!);
    if (!renderer) { setGpuAvailable(false); setUseGPU(false); return; }
    glRef.current = renderer;
    dirtyRef.current = true;
    return () => { glRef.current?.dispose(); glRef.current = null; };
  }, []);

  // ─── Draw ────────────────────────────────────────────────────────────────

  const draw = useCallback(() => {
    rafRef.current = requestAnimationFrame(draw);
    if (!dirtyRef.current) return;
    dirtyRef.current = false;

    const { rMin, rMax, yMin, yMax, iterations, burnin, colorScheme, logScale, useGPU } = pRef.current;

    if (useGPU && glRef.current) {
      // ── GPU path ──────────────────────────────────────────────────────────
      const canvas = glCanvasRef.current;
      if (!canvas || canvas.width === 0 || canvas.height === 0) return;
      glRef.current.render({ rMin, rMax, yMin, yMax, iterations, burnin, colorScheme, logScale });

    } else {
      // ── CPU path ──────────────────────────────────────────────────────────
      const canvas = cpuCanvasRef.current;
      if (!canvas) return;
      const W = canvas.width, H = canvas.height;
      if (W === 0 || H === 0) return;

      const counts = new Uint16Array(W * H);
      const yRange = yMax - yMin;

      for (let px = 0; px < W; px++) {
        const r = rMin + (px / (W - 1)) * (rMax - rMin);
        let x = 0.5;
        for (let i = 0; i < burnin; i++) x = r * x * (1 - x);
        for (let i = 0; i < iterations; i++) {
          x = r * x * (1 - x);
          if (x >= yMin && x <= yMax) {
            const xNorm = (x - yMin) / yRange;
            const py = H - 1 - Math.round(xNorm * (H - 1));
            if (py >= 0 && py < H) counts[py * W + px]++;
          }
        }
      }

      let maxCount = 0;
      for (let i = 0; i < counts.length; i++) {
        if (counts[i] > maxCount) maxCount = counts[i];
      }

      const ctx  = canvas.getContext('2d')!;
      const img  = ctx.createImageData(W, H);
      const d    = img.data;
      const logMax = maxCount > 0 ? Math.log1p(maxCount) : 1;

      for (let i = 0; i < W * H; i++) {
        const c   = counts[i];
        const idx = i * 4;

        if (c === 0) {
          d[idx] = BG_R; d[idx + 1] = BG_G; d[idx + 2] = BG_B; d[idx + 3] = 255;
          continue;
        }

        const t = logScale ? Math.log1p(c) / logMax : c / maxCount;

        switch (colorScheme) {
          case 'cyan': {
            const s = 0.15 + 0.85 * t;
            d[idx]     = Math.round(BG_R + (34  - BG_R) * s);
            d[idx + 1] = Math.round(BG_G + (211 - BG_G) * s);
            d[idx + 2] = Math.round(BG_B + (238 - BG_B) * s);
            d[idx + 3] = 255;
            break;
          }
          case 'heat': {
            if (t < 0.33) {
              const s = t / 0.33;
              d[idx] = Math.round(180 * s); d[idx + 1] = 0; d[idx + 2] = 0;
            } else if (t < 0.67) {
              const s = (t - 0.33) / 0.34;
              d[idx] = Math.round(180 + 75 * s); d[idx + 1] = Math.round(180 * s); d[idx + 2] = 0;
            } else {
              const s = (t - 0.67) / 0.33;
              d[idx] = 255; d[idx + 1] = Math.round(180 + 75 * s); d[idx + 2] = Math.round(240 * s);
            }
            d[idx + 3] = 255;
            break;
          }
          case 'plasma': {
            if (t < 0.5) {
              const s = t * 2;
              d[idx]     = Math.round(100 + 155 * s);
              d[idx + 1] = 0;
              d[idx + 2] = Math.round(200 * (1 - s));
            } else {
              const s = (t - 0.5) * 2;
              d[idx]     = 255;
              d[idx + 1] = Math.round(200 * s);
              d[idx + 2] = 0;
            }
            d[idx + 3] = 255;
            break;
          }
          case 'mono': {
            const v = Math.round(255 * t);
            d[idx] = v; d[idx + 1] = v; d[idx + 2] = v; d[idx + 3] = 255;
            break;
          }
        }
      }
      ctx.putImageData(img, 0, 0);
    }
  }, []);

  // ─── Resize observer — watches the container, resizes both canvases ───────

  useEffect(() => {
    const container = glCanvasRef.current?.parentElement;
    if (!container) return;
    const ro = new ResizeObserver(() => {
      const dpr  = Math.min(window.devicePixelRatio || 1, 2);
      const rect = container.getBoundingClientRect();
      const w = (rect.width  * dpr) | 0;
      const h = (rect.height * dpr) | 0;
      for (const canvas of [glCanvasRef.current, cpuCanvasRef.current]) {
        if (canvas) { canvas.width = w; canvas.height = h; }
      }
      dirtyRef.current = true;
    });
    ro.observe(container);
    return () => ro.disconnect();
  }, []);

  // ─── Animation loop lifecycle ─────────────────────────────────────────────

  useEffect(() => {
    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [draw]);

  // ─── Scroll-wheel zoom (needs passive:false to preventDefault) ────────────

  useEffect(() => {
    const el = layerRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if (sidebarRef.current?.contains(e.target as Node)) return;
      e.preventDefault();

      const rect   = el.getBoundingClientRect();
      const cx     = (e.clientX - rect.left)  / rect.width;
      const cy     = (e.clientY - rect.top)   / rect.height;
      const factor = e.deltaY > 0 ? 1.25 : 1 / 1.25;

      const { rMin, rMax, yMin, yMax } = pRef.current;
      const rRange = rMax - rMin;
      const yRange = yMax - yMin;

      // Pivot point under the cursor (CSS y=0 is top = yMax)
      const rAtCursor = rMin + cx * rRange;
      const yAtCursor = yMax - cy * yRange;

      const newRRange = rRange * factor;
      const newYRange = yRange * factor;

      setRMin(Math.max(0, rAtCursor - cx * newRRange));
      setRMax(Math.min(4, rAtCursor + (1 - cx) * newRRange));
      setYMin(Math.max(0, yAtCursor - (1 - cy) * newYRange));
      setYMax(Math.min(1, yAtCursor + cy * newYRange));
      setActivePreset(null);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  // ─── Drag-to-zoom mouse handlers ─────────────────────────────────────────

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    dragStartRef.current = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    setSelRect(null);
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    // Update hover readout
    const { rMin, rMax, yMin, yMax } = pRef.current;
    setHoverR(rMin + (mx / rect.width) * (rMax - rMin));
    setHoverX(yMax - (my / rect.height) * (yMax - yMin));

    if (!dragStartRef.current) return;
    const { x: x0, y: y0 } = dragStartRef.current;
    setSelRect({
      x: Math.min(x0, mx),
      y: Math.min(y0, my),
      w: Math.abs(mx - x0),
      h: Math.abs(my - y0),
    });
  }, []);

  const handleMouseUp = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const start = dragStartRef.current;
    dragStartRef.current = null;
    setSelRect(null);
    if (!start) return;

    const rect  = e.currentTarget.getBoundingClientRect();
    const endX  = e.clientX - rect.left;
    const endY  = e.clientY - rect.top;
    const W     = rect.width;
    const H     = rect.height;

    const xLeft  = Math.min(start.x, endX);
    const xRight = Math.max(start.x, endX);
    const yTop   = Math.min(start.y, endY);
    const yBot   = Math.max(start.y, endY);

    // Ignore tiny clicks
    if (xRight - xLeft < 4 && yBot - yTop < 4) return;

    const { rMin, rMax, yMin, yMax } = pRef.current;
    const rRange = rMax - rMin;
    const yRange = yMax - yMin;

    const newRMin = rMin + (xLeft  / W) * rRange;
    const newRMax = rMin + (xRight / W) * rRange;
    // CSS y=0 is top (= yMax), CSS y=H is bottom (= yMin)
    const newYMax = yMax - (yTop / H) * yRange;
    const newYMin = yMax - (yBot / H) * yRange;

    setZoomHistory(prev => [...prev, { rMin, rMax, yMin, yMax }]);
    setRMin(newRMin);
    setRMax(newRMax);
    setYMin(newYMin);
    setYMax(newYMax);
    setActivePreset(null);
  }, []);

  const handleMouseLeave = useCallback(() => {
    setHoverR(null);
    setHoverX(null);
    dragStartRef.current = null;
    setSelRect(null);
  }, []);

  // ─── Other handlers ───────────────────────────────────────────────────────

  const handleRMinChange = useCallback((v: number) => {
    setRMin(Math.min(v, rMax - 0.001));
    setActivePreset(null);
  }, [rMax]);

  const handleRMaxChange = useCallback((v: number) => {
    setRMax(Math.max(v, rMin + 0.001));
    setActivePreset(null);
  }, [rMin]);

  const zoomBack = useCallback(() => {
    setZoomHistory(prev => {
      if (prev.length === 0) return prev;
      const last = prev[prev.length - 1];
      setRMin(last.rMin); setRMax(last.rMax);
      setYMin(last.yMin); setYMax(last.yMax);
      setActivePreset(null);
      return prev.slice(0, -1);
    });
  }, []);

  const reset = useCallback(() => {
    setRMin(2.5); setRMax(4.0);
    setYMin(0);   setYMax(1);
    setIterations(300); setBurnin(200);
    setColorScheme('cyan'); setLogScale(true);
    setActivePreset(0);
    setZoomHistory([]);
  }, []);


  const flashCopied = useCallback(() => {
    setCopied(true);
    if (copiedTimeoutRef.current !== null) window.clearTimeout(copiedTimeoutRef.current);
    copiedTimeoutRef.current = window.setTimeout(() => setCopied(false), 2000);
  }, []);

  const handleShare = useCallback(() => {
    shareUrl({ r0: rMin, r1: rMax, c: colorScheme });
    flashCopied();
  }, [rMin, rMax, colorScheme, flashCopied, shareUrl]);

  useEffect(() => () => {
    if (copiedTimeoutRef.current !== null) window.clearTimeout(copiedTimeoutRef.current);
  }, []);

  // ─── Keyboard shortcuts ───────────────────────────────────────────────────

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.code === 'KeyR') { e.preventDefault(); reset(); }
      if (e.code === 'KeyF') { e.preventDefault(); toggleFullscreen(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [reset, toggleFullscreen]);

  // ─── Render ───────────────────────────────────────────────────────────────

  const rTicks = niceTicks(rMin, rMax, 4);
  const xTicks = niceTicks(yMin, yMax, 5);
  const rStep = rTicks.length > 1 ? rTicks[1] - rTicks[0] : rMax - rMin;
  const xStep = xTicks.length > 1 ? xTicks[1] - xTicks[0] : yMax - yMin;

  return (
    <div ref={containerRef} className={styles.container}>
      {/* GPU canvas */}
      <canvas
        ref={glCanvasRef}
        className={styles.canvas}
        style={{ visibility: useGPU && gpuAvailable ? 'visible' : 'hidden' }}
      />
      {/* CPU canvas */}
      <canvas
        ref={cpuCanvasRef}
        className={styles.canvas}
        style={{ visibility: !useGPU || !gpuAvailable ? 'visible' : 'hidden' }}
      />

      {/* Interaction overlay — sits above canvases, below sidebar */}
      <div
        ref={layerRef}
        className={`${styles.interactionLayer}${selRect ? ` ${styles.selecting}` : ''}`}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
      >
        {selRect && (
          <div
            className={styles.selectionRect}
            style={{ left: selRect.x, top: selRect.y, width: selRect.w, height: selRect.h }}
          />
        )}
      </div>

      {/* Axis overlay — tick marks and labels for r (x-axis) and x/population (y-axis) */}
      <svg className={styles.axisOverlay} aria-hidden="true">
        {rTicks.map(rv => {
          const xp = (rv - rMin) / (rMax - rMin) * 100;
          if (xp < 0.5 || xp > 99.5) return null;
          return (
            <g key={`r${rv}`}>
              <line x1={`${xp}%`} y1="94%" x2={`${xp}%`} y2="97%"
                    stroke="rgba(255,255,255,0.2)" strokeWidth="1" />
              <text x={`${xp}%`} y="93%"
                    textAnchor="middle" dominantBaseline="auto"
                    fill="rgba(255,255,255,0.5)" fontSize="10" fontFamily="monospace">
                {fmtTick(rv, rStep)}
              </text>
            </g>
          );
        })}
        {xTicks.map(xv => {
          const yp = (1 - (xv - yMin) / (yMax - yMin)) * 100;
          if (yp < 1 || yp > 92) return null;
          return (
            <g key={`x${xv}`}>
              <line x1="0" y1={`${yp}%`} x2="7" y2={`${yp}%`}
                    stroke="rgba(255,255,255,0.2)" strokeWidth="1" />
              <text x="10" y={`${yp}%`}
                    textAnchor="start" dominantBaseline="middle"
                    fill="rgba(255,255,255,0.5)" fontSize="10" fontFamily="monospace">
                {fmtTick(xv, xStep)}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Sidebar collapse toggle — always visible, slides with sidebar */}
      <button
        className={`${styles.sidebarToggle}${sidebarOpen ? '' : ` ${styles.sidebarToggleCollapsed}`}`}
        type="button"
        title={sidebarOpen ? 'Hide controls' : 'Show controls'}
        onClick={() => setSidebarOpen(v => !v)}
      >
        {sidebarOpen ? '›' : '‹'}
      </button>

      {/* ─── Right-hand config sidebar ──────────────────────────────────── */}
      <div
        ref={sidebarRef}
        className={`${styles.sidebar}${sidebarOpen ? '' : ` ${styles.sidebarHidden}`}`}
      >
        <div className={styles.sidebarPanels}>
          <ControlPanel title="Presets">
            <ControlGroup>
              <div className={styles.presetGrid}>
                {PRESETS.map((p, idx) => (
                  <button
                    key={p.label}
                    className={`${styles.presetBtn}${activePreset === idx ? ` ${styles.presetBtnActive}` : ''}`}
                    type="button"
                    title={p.desc}
                    onClick={() => {
                      setRMin(p.rMin); setRMax(p.rMax);
                      setYMin(0);      setYMax(1);
                      setActivePreset(idx);
                      setZoomHistory([]);
                    }}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            </ControlGroup>
          </ControlPanel>

          <ControlPanel title="Parameters">
            <ControlGroup>
              <Slider
                label="r min"
                value={rMin}
                onChange={handleRMinChange}
                min={0}
                max={4}
                step={0.001}
                format={v => v.toFixed(3)}
                manualInput
              />
              <Slider
                label="r max"
                value={rMax}
                onChange={handleRMaxChange}
                min={0}
                max={4}
                step={0.001}
                format={v => v.toFixed(3)}
                manualInput
              />
              <Slider
                label="Iterations"
                value={iterations}
                onChange={setIterations}
                min={50}
                max={1000}
                step={50}
              />
              <Slider
                label="Burn-in"
                value={burnin}
                onChange={setBurnin}
                min={50}
                max={500}
                step={50}
              />
            </ControlGroup>
          </ControlPanel>

          <ControlPanel title="Display" defaultOpen={false}>
            <ControlGroup>
              <SelectControl
                label="Color scheme"
                value={colorScheme}
                onChange={setColorScheme}
                options={[
                  { value: 'cyan'   as const, label: 'Cyan (default)' },
                  { value: 'heat'   as const, label: 'Heat map'       },
                  { value: 'plasma' as const, label: 'Plasma'         },
                  { value: 'mono'   as const, label: 'Monochrome'     },
                ]}
              />
              <Toggle label="Log density" value={logScale} onChange={setLogScale} />
              {gpuAvailable && (
                <Toggle label="GPU render" value={useGPU} onChange={setUseGPU} />
              )}
            </ControlGroup>
          </ControlPanel>

        </div>

        <div className={styles.sidebarActions}>
          {zoomHistory.length > 0 && (
            <button className={styles.backBtn} type="button" onClick={zoomBack}>
              ← Back
            </button>
          )}
          <SimControls onReset={reset} onExport={() => setShowExport(true)} />
        </div>
      </div>

      {showExport && (
        <ExportDialog
          onClose={() => setShowExport(false)}
          onDownload={({ width, height, format }) => {
            const canvas = (useGPU ? glCanvasRef : cpuCanvasRef).current;
            if (!canvas) return;
            exportImage(canvas, width, height, format, 'bifurcation');
            setShowExport(false);
          }}
        />
      )}

      {/* ─── HUD ────────────────────────────────────────────────────────── */}
      <div className={styles.hud}>
        <div className={styles.hudLeft}>
          <span className={styles.hudTitle}>Bifurcation</span>
          <span className={styles.hudSub}>
            r ∈ [{rMin.toFixed(3)}, {rMax.toFixed(3)}] · x ∈ [{yMin.toFixed(3)}, {yMax.toFixed(3)}]
          </span>
        </div>
        <div className={styles.hudRight}>
          {hoverR !== null && hoverX !== null && (
            <span className={styles.hudHover}>r={hoverR.toFixed(4)}  x={hoverX.toFixed(4)}</span>
          )}
          <span className={styles.hudHint}>
            {activePreset !== null ? PRESETS[activePreset].desc : 'drag to zoom · scroll to zoom · logistic map x → r·x·(1−x)'}
          </span>
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
          <button className={styles.infoBtn} onClick={() => setShowInfo(true)} title="About the bifurcation diagram">ⓘ</button>
        </div>
      </div>

      {showInfo && (
        <InfoDialog title="Bifurcation Diagram" onClose={() => setShowInfo(false)}>
          <p>
            Shows the long-term behaviour of the logistic map
            x<sub>n+1</sub> = r·x<sub>n</sub>·(1−x<sub>n</sub>) as <em>r</em> varies from 0 to 4.
          </p>
          <h3>Period doubling</h3>
          <p>
            Low <em>r</em>: population settles on a fixed point. As <em>r</em> increases the
            stable state splits into a 2-cycle, then 4, 8, 16… until around r ≈ 3.57 where
            behaviour becomes fully chaotic.
          </p>
          <h3>Feigenbaum constant</h3>
          <p>
            The ratio between successive bifurcation intervals converges to
            δ ≈ 4.669. This constant shows up in every smooth one-humped map, not just the logistic map.
          </p>
          <h3>Controls</h3>
          <ul>
            <li><strong>Drag a rectangle:</strong> zoom into any region</li>
            <li><strong>Scroll:</strong> zoom around the cursor</li>
            <li><strong>Back:</strong> undo the last zoom</li>
          </ul>
        </InfoDialog>
      )}
    </div>
  );
}
