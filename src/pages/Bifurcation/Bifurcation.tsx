import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Slider, Toggle, SelectControl,
  ControlPanel, ControlGroup,
} from '@/components/Controls';
import styles from './Bifurcation.module.css';

// ─── Types ────────────────────────────────────────────────────────────────────

type ColorScheme = 'cyan' | 'heat' | 'plasma' | 'mono';

// ─── Constants ────────────────────────────────────────────────────────────────

const BG_R = 7, BG_G = 7, BG_B = 18; // #070712

interface Preset {
  label: string;
  rMin: number;
  rMax: number;
  desc: string;
}

const PRESETS: Preset[] = [
  { label: 'Full',     rMin: 2.5,    rMax: 4.0,    desc: 'r ∈ [2.5, 4] — full bifurcation diagram'         },
  { label: 'Doubling', rMin: 2.8,    rMax: 3.6,    desc: 'r ∈ [2.8, 3.6] — period-doubling cascade'        },
  { label: 'Chaos',    rMin: 3.5,    rMax: 4.0,    desc: 'r ∈ [3.5, 4] — onset of chaos'                   },
  { label: '3-cycle',  rMin: 3.82,   rMax: 3.88,   desc: 'r ∈ [3.82, 3.88] — period-3 window'              },
  { label: 'δ point',  rMin: 3.54,   rMax: 3.57,   desc: 'r ≈ 3.569 — Feigenbaum accumulation point'       },
  { label: 'Deep',     rMin: 3.856,  rMax: 3.862,  desc: 'r ∈ [3.856, 3.862] — deep self-similar structure'},
];

// ─── Component ────────────────────────────────────────────────────────────────

export default function Bifurcation() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef    = useRef<number>(0);
  const dirtyRef  = useRef(true);

  // ─── State ──────────────────────────────────────────────────────────────

  const [rMin,         setRMin]         = useState(2.5);
  const [rMax,         setRMax]         = useState(4.0);
  const [iterations,   setIterations]   = useState(300);
  const [burnin,       setBurnin]       = useState(200);
  const [colorScheme,  setColorScheme]  = useState<ColorScheme>('cyan');
  const [logScale,     setLogScale]     = useState(true);
  const [activePreset, setActivePreset] = useState<number | null>(0);

  // Mirror to ref so draw loop always reads fresh values without re-creating draw
  const pRef = useRef({ rMin, rMax, iterations, burnin, colorScheme, logScale });
  useEffect(() => {
    pRef.current = { rMin, rMax, iterations, burnin, colorScheme, logScale };
    dirtyRef.current = true;
  });

  // ─── Draw ────────────────────────────────────────────────────────────────

  const draw = useCallback(() => {
    rafRef.current = requestAnimationFrame(draw);

    const canvas = canvasRef.current;
    if (!canvas || !dirtyRef.current) return;
    dirtyRef.current = false;

    const { rMin, rMax, iterations, burnin, colorScheme, logScale } = pRef.current;

    const W = canvas.width;
    const H = canvas.height;
    if (W === 0 || H === 0) return;

    // ── Accumulate hit counts ──────────────────────────────────────────────
    const counts = new Uint16Array(W * H);

    for (let px = 0; px < W; px++) {
      const r = rMin + (px / (W - 1)) * (rMax - rMin);
      let x = 0.5;
      for (let i = 0; i < burnin; i++) x = r * x * (1 - x);
      for (let i = 0; i < iterations; i++) {
        x = r * x * (1 - x);
        const py = H - 1 - Math.round(x * (H - 1));
        if (py >= 0 && py < H) counts[py * W + px]++;
      }
    }

    // ── Find max for normalisation ─────────────────────────────────────────
    let maxCount = 0;
    for (let i = 0; i < counts.length; i++) {
      if (counts[i] > maxCount) maxCount = counts[i];
    }

    // ── Render to ImageData ────────────────────────────────────────────────
    const ctx = canvas.getContext('2d')!;
    const img = ctx.createImageData(W, H);
    const d   = img.data;

    const logMax = maxCount > 0 ? Math.log1p(maxCount) : 1;

    for (let i = 0; i < W * H; i++) {
      const c   = counts[i];
      const idx = i * 4;

      if (c === 0) {
        d[idx] = BG_R; d[idx + 1] = BG_G; d[idx + 2] = BG_B; d[idx + 3] = 255;
        continue;
      }

      const t = logScale
        ? Math.log1p(c) / logMax
        : c / maxCount;

      switch (colorScheme) {
        case 'cyan': {
          // Dark background → dim cyan → bright white-cyan
          const s = 0.15 + 0.85 * t;
          d[idx]     = Math.round(BG_R + (34  - BG_R) * s);
          d[idx + 1] = Math.round(BG_G + (211 - BG_G) * s);
          d[idx + 2] = Math.round(BG_B + (238 - BG_B) * s);
          d[idx + 3] = 255;
          break;
        }
        case 'heat': {
          // Black → red → yellow → white
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
          // Dark purple → magenta → orange → bright yellow
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
  }, []);

  // ─── Resize observer ─────────────────────────────────────────────────────

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => {
      const dpr  = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      canvas.width  = (rect.width  * dpr) | 0;
      canvas.height = (rect.height * dpr) | 0;
      dirtyRef.current = true;
    });
    ro.observe(canvas);
    return () => ro.disconnect();
  }, []);

  // ─── Animation loop lifecycle ─────────────────────────────────────────────

  useEffect(() => {
    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [draw]);

  // ─── Handlers ────────────────────────────────────────────────────────────

  const handleRMinChange = useCallback((v: number) => {
    setRMin(Math.min(v, rMax - 0.001));
    setActivePreset(null);
  }, [rMax]);

  const handleRMaxChange = useCallback((v: number) => {
    setRMax(Math.max(v, rMin + 0.001));
    setActivePreset(null);
  }, [rMin]);

  const reset = useCallback(() => {
    setRMin(2.5);
    setRMax(4.0);
    setIterations(300);
    setBurnin(200);
    setColorScheme('cyan');
    setLogScale(true);
    setActivePreset(0);
  }, []);

  // ─── Render ───────────────────────────────────────────────────────────────

  return (
    <div className={styles.container}>
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
                    onClick={() => {
                      setRMin(p.rMin);
                      setRMax(p.rMax);
                      setActivePreset(idx);
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
            </ControlGroup>
          </ControlPanel>
        </div>

        <div className={styles.sidebarActions}>
          <button className={styles.resetBtn} type="button" onClick={reset}>
            Reset
          </button>
        </div>
      </div>

      {/* ─── HUD ────────────────────────────────────────────────────────── */}
      <div className={styles.hud}>
        <div className={styles.hudLeft}>
          <span className={styles.hudTitle}>Bifurcation</span>
          <span className={styles.hudSub}>
            r ∈ [{rMin.toFixed(3)}, {rMax.toFixed(3)}]
          </span>
        </div>
        <div className={styles.hudRight}>
          <span className={styles.hudHint}>
            {activePreset !== null ? PRESETS[activePreset].desc : 'logistic map  x → r·x·(1−x)'}
          </span>
        </div>
      </div>
    </div>
  );
}
