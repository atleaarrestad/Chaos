import { useState, useEffect, useRef, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Slider, Toggle, SelectControl,
  ControlPanel, ControlGroup, SimControls,
} from '@/components/Controls';
import { InfoDialog } from '@/components/InfoDialog';
import { useFullscreen } from '@/hooks/useFullscreen';
import { getNumParam, getStrParam, useShareUrl } from '@/hooks/useUrlParams';
import { exportCanvasPng } from '@/lib/exportPng';
import styles from './Cardioid.module.css';

type ColorScheme = 'cardioid' | 'rainbow' | 'plasma' | 'mono';

// ─── Constants ────────────────────────────────────────────────────────────────

const BG  = '#0b0b18';
const TAU = Math.PI * 2;

const PRESETS = [
  { label: 'Cardioid',  k: 2,         desc: 'k = 2, one-cusped envelope' },
  { label: 'Nephroid',  k: 3,         desc: 'k = 3, two-cusped epicycloid' },
  { label: 'Trefoil',   k: 4,         desc: 'k = 4, three-cusped epicycloid' },
  { label: '½',         k: 0.5,       desc: 'k = ½, half-frequency pattern' },
  { label: 'φ',         k: 1.6180339, desc: 'k = φ, golden ratio (non-repeating)' },
  { label: 'e',         k: Math.E,    desc: 'k = e ≈ 2.718, Euler\'s number' },
] as const;

const COLOR_SCHEMES: ColorScheme[] = ['cardioid', 'rainbow', 'plasma', 'mono'];

function getInitialColorScheme(searchParams: URLSearchParams): ColorScheme {
  const value = getStrParam(searchParams, 'c', 'cardioid');
  return COLOR_SCHEMES.includes(value as ColorScheme) ? value as ColorScheme : 'cardioid';
}

// ─── Color helpers ────────────────────────────────────────────────────────────

function lineColor(scheme: ColorScheme, t: number, alpha: number): string {
  switch (scheme) {
    case 'cardioid': {
      const h = 25 + t * 35; // warm orange-yellow sweep
      return `hsla(${h}, 95%, 65%, ${alpha})`;
    }
    case 'rainbow':
      return `hsla(${Math.round(t * 360)}, 80%, 65%, ${alpha})`;
    case 'plasma': {
      const h = 260 - t * 200; // blue → violet → magenta → red
      return `hsla(${h}, 90%, 60%, ${alpha})`;
    }
    case 'mono':
      return `rgba(215, 228, 255, ${alpha})`;
  }
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function Cardioid() {
  const [searchParams] = useSearchParams();
  const initialFactor = getNumParam(searchParams, 'k', 2);
  const initialNumPoints = Math.round(getNumParam(searchParams, 'n', 200));
  const initialColorScheme = getInitialColorScheme(searchParams);

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const tPrevRef = useRef<number>(0);
  const kRef = useRef<number>(initialFactor);        // live animated factor
  const hudKRef = useRef<HTMLSpanElement>(null); // updated in-place each frame
  const copiedTimeoutRef = useRef<number | null>(null);

  const { isFullscreen, toggleFullscreen } = useFullscreen(containerRef);
  const { shareUrl } = useShareUrl();

  // ─── State ────────────────────────────────────────────────────────────────

  const [numPoints,   setNumPoints]   = useState(() => initialNumPoints);
  const [factor,      setFactor]      = useState(() => initialFactor);
  const [animate,     setAnimate]     = useState(true);
  const [animSpeed,   setAnimSpeed]   = useState(0.3);
  const [colorScheme, setColorScheme] = useState<ColorScheme>(() => initialColorScheme);
  const [lineOpacity, setLineOpacity] = useState(0.4);
  const [lineWidth,   setLineWidth]   = useState(1.0);
  const [showCircle,  setShowCircle]  = useState(true);
  const [showDots,    setShowDots]    = useState(false);
  const [activePreset, setActivePreset] = useState<number | null>(() => {
    const idx = PRESETS.findIndex((preset) => preset.k === initialFactor);
    return idx >= 0 ? idx : null;
  });
  const [showInfo, setShowInfo] = useState(false);
  const [copied, setCopied] = useState(false);

  // Mirror state to ref so animation loop always reads fresh values
  const pRef = useRef({
    numPoints, animate, animSpeed, colorScheme, lineOpacity, lineWidth, showCircle, showDots,
  });
  useEffect(() => {
    pRef.current = { numPoints, animate, animSpeed, colorScheme, lineOpacity, lineWidth, showCircle, showDots };
  });

  // ─── Factor slider ────────────────────────────────────────────────────────

  const handleFactorChange = useCallback((v: number) => {
    setFactor(v);
    kRef.current = v;
    setActivePreset(null);
  }, []);

  // ─── Draw loop ────────────────────────────────────────────────────────────

  const draw = useCallback((ts: number) => {
    const canvas = canvasRef.current;
    if (!canvas) { rafRef.current = requestAnimationFrame(draw); return; }

    const {
      numPoints, animate, animSpeed, colorScheme, lineOpacity, lineWidth, showCircle, showDots,
    } = pRef.current;

    const dt = tPrevRef.current > 0 ? (ts - tPrevRef.current) / 1000 : 0;
    tPrevRef.current = ts;

    if (animate && dt > 0) {
      kRef.current += animSpeed * dt;
      if (kRef.current > 100) kRef.current = 0.5;
    }
    const k = kRef.current;

    // Live HUD label — no React re-render
    if (hudKRef.current) hudKRef.current.textContent = k.toFixed(3);

    const ctx = canvas.getContext('2d')!;
    const W = canvas.width, H = canvas.height;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cx  = W / 2, cy = H / 2;
    const R   = Math.min(W, H) * 0.42;

    // Clear
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, W, H);

    // ── Reference circle ──────────────────────────────────────────────────
    if (showCircle) {
      ctx.beginPath();
      ctx.arc(cx, cy, R, 0, TAU);
      ctx.strokeStyle = 'rgba(255,255,255,0.10)';
      ctx.lineWidth = dpr;
      ctx.stroke();
    }

    // ── Point dots ────────────────────────────────────────────────────────
    if (showDots) {
      ctx.fillStyle = 'rgba(255,255,255,0.28)';
      for (let i = 0; i < numPoints; i++) {
        const angle = (i / numPoints) * TAU - Math.PI / 2;
        ctx.beginPath();
        ctx.arc(cx + R * Math.cos(angle), cy + R * Math.sin(angle), 1.5 * dpr, 0, TAU);
        ctx.fill();
      }
    }

    // ── Chords (times-table lines) ────────────────────────────────────────
    ctx.lineWidth = lineWidth * dpr;

    if (colorScheme === 'mono') {
      // All same color → one batched path
      ctx.strokeStyle = lineColor('mono', 0, lineOpacity);
      ctx.beginPath();
      for (let i = 0; i < numPoints; i++) {
        const a1 = (i     / numPoints) * TAU - Math.PI / 2;
        const a2 = (i * k / numPoints) * TAU - Math.PI / 2;
        ctx.moveTo(cx + R * Math.cos(a1), cy + R * Math.sin(a1));
        ctx.lineTo(cx + R * Math.cos(a2), cy + R * Math.sin(a2));
      }
      ctx.stroke();
    } else {
      // Group into hue buckets to reduce strokeStyle switches (all color schemes)
      const BUCKETS = 30;
      for (let b = 0; b < BUCKETS; b++) {
        ctx.strokeStyle = lineColor(colorScheme, b / BUCKETS, lineOpacity);
        ctx.beginPath();
        const lo = Math.round((b / BUCKETS) * numPoints);
        const hi = Math.round(((b + 1) / BUCKETS) * numPoints);
        for (let i = lo; i < hi; i++) {
          const a1 = (i     / numPoints) * TAU - Math.PI / 2;
          const a2 = (i * k / numPoints) * TAU - Math.PI / 2;
          ctx.moveTo(cx + R * Math.cos(a1), cy + R * Math.sin(a1));
          ctx.lineTo(cx + R * Math.cos(a2), cy + R * Math.sin(a2));
        }
        ctx.stroke();
      }
    }

    rafRef.current = requestAnimationFrame(draw);
  }, []);

  // ─── Resize observer ──────────────────────────────────────────────────────

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => {
      const dpr  = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      canvas.width  = (rect.width  * dpr) | 0;
      canvas.height = (rect.height * dpr) | 0;
    });
    ro.observe(canvas);
    return () => ro.disconnect();
  }, []);

  // ─── Animation loop lifecycle ─────────────────────────────────────────────

  useEffect(() => {
    tPrevRef.current = 0;
    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [draw]);

  // ─── Reset ────────────────────────────────────────────────────────────────

  const reset = useCallback(() => {
    kRef.current = 2;
    setFactor(2);
    setNumPoints(200);
    setAnimate(true);
    setAnimSpeed(0.3);
    setColorScheme('cardioid');
    setLineOpacity(0.4);
    setLineWidth(1.0);
    setShowCircle(true);
    setShowDots(false);
    setActivePreset(0);
  }, []);

  const exportPng = useCallback(() => {
    if (!canvasRef.current) return;
    exportCanvasPng(canvasRef.current, 'cardioid.png');
  }, []);

  const flashCopied = useCallback(() => {
    setCopied(true);
    if (copiedTimeoutRef.current !== null) window.clearTimeout(copiedTimeoutRef.current);
    copiedTimeoutRef.current = window.setTimeout(() => setCopied(false), 2000);
  }, []);

  const handleShare = useCallback(() => {
    shareUrl({ k: kRef.current, n: numPoints, c: colorScheme });
    flashCopied();
  }, [colorScheme, flashCopied, numPoints, shareUrl]);

  useEffect(() => () => {
    if (copiedTimeoutRef.current !== null) window.clearTimeout(copiedTimeoutRef.current);
  }, []);

  // ─── Keyboard shortcuts ───────────────────────────────────────────────────

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.code === 'Space') { e.preventDefault(); setAnimate(a => !a); }
      if (e.code === 'KeyR')  { e.preventDefault(); reset(); }
      if (e.code === 'KeyF')  { e.preventDefault(); toggleFullscreen(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [reset, toggleFullscreen]);

  // ─── Render ───────────────────────────────────────────────────────────────

  return (
    <div ref={containerRef} className={styles.container}>
      <canvas ref={canvasRef} className={styles.canvas} />

      {/* ─── Right-hand config sidebar ──────────────────────────────────── */}
      <div className={styles.sidebar}>
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
                    onClick={() => { kRef.current = p.k; setFactor(p.k); setAnimate(false); setActivePreset(idx); }}
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
                label="Factor (k)"
                value={factor}
                onChange={handleFactorChange}
                min={0.5}
                max={100}
                step={0.001}
                format={v => v.toFixed(3)}
                manualInput
              />
              <Slider
                label="Points (N)"
                value={numPoints}
                onChange={setNumPoints}
                min={50}
                max={500}
                step={10}
              />
            </ControlGroup>
          </ControlPanel>

          <ControlPanel title="Animation">
            <ControlGroup>
              <Slider
                label="Speed"
                value={animSpeed}
                onChange={setAnimSpeed}
                min={0.05}
                max={5}
                step={0.05}
                format={v => `${v.toFixed(2)} k/s`}
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
                  { value: 'cardioid' as const, label: 'Cardioid (orange)' },
                  { value: 'rainbow'  as const, label: 'Rainbow' },
                  { value: 'plasma'   as const, label: 'Plasma' },
                  { value: 'mono'     as const, label: 'Monochrome' },
                ]}
              />
              <Slider
                label="Opacity"
                value={lineOpacity}
                onChange={setLineOpacity}
                min={0.05}
                max={1}
                step={0.05}
                format={v => `${Math.round(v * 100)}%`}
              />
              <Slider
                label="Line width"
                value={lineWidth}
                onChange={setLineWidth}
                min={0.3}
                max={3}
                step={0.1}
                format={v => `${v.toFixed(1)}px`}
              />
              <Toggle label="Circle" value={showCircle} onChange={setShowCircle} />
              <Toggle label="Point dots" value={showDots} onChange={setShowDots} />
            </ControlGroup>
          </ControlPanel>

        </div>

        <div className={styles.sidebarActions}>
          <SimControls
            running={animate}
            onToggle={() => setAnimate(a => !a)}
            onReset={reset}
            onExport={exportPng}
          />
        </div>
      </div>

      {/* ─── HUD ────────────────────────────────────────────────────────── */}
      <div className={styles.hud}>
        <div className={styles.hudLeft}>
          <span className={styles.hudTitle}>Cardioid</span>
          <span className={styles.hudSub}>
            k = <span ref={hudKRef}>{factor.toFixed(3)}</span>
          </span>
        </div>
        <div className={styles.hudRight}>
          <span className={styles.hudHint}>
            {activePreset !== null ? PRESETS[activePreset].desc : 'times-table visualization'}
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
          <button className={styles.infoBtn} onClick={() => setShowInfo(true)} title="About the cardioid">ⓘ</button>
        </div>
      </div>

      {showInfo && (
        <InfoDialog title="Cardioid & Times Tables" onClose={() => setShowInfo(false)}>
          <p>
            Points are evenly spaced around a circle. For multiplier <em>k</em>, each point
            <em> p</em> is connected by a line to point <em>p × k (mod n)</em>. The lines
            form an envelope curve.
          </p>
          <h3>The cardioid</h3>
          <p>
            At <em>k = 2</em> the envelope is a perfect cardioid, the same heart-shaped curve
            that forms the main bulb of the Mandelbrot set. It's the same underlying math.
          </p>
          <h3>Other multipliers</h3>
          <p>
            k = 3 gives a nephroid, k = 4 gives a three-cusped curve. Each integer multiplier
            produces a different epicycloid.
          </p>
          <h3>Controls</h3>
          <ul>
            <li><strong>Points:</strong> number of points on the circle</li>
            <li><strong>Multiplier:</strong> the factor <em>k</em></li>
            <li><strong>Animate:</strong> sweep <em>k</em> to watch the shapes morph</li>
          </ul>
        </InfoDialog>
      )}
    </div>
  );
}
