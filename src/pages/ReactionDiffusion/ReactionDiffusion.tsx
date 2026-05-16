import { useCallback, useEffect, useRef, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  ControlGroup, ControlPanel, SimControls, Slider, Toggle,
} from '@/components/Controls';
import { InfoDialog } from '@/components/InfoDialog';
import { useFullscreen } from '@/hooks/useFullscreen';
import { getNumParam } from '@/hooks/useUrlParams';
import ExportDialog from '../../components/ExportDialog/ExportDialog';
import { exportImage } from '../../lib/exportImage';
import { ReactionDiffusionGPU, SIM_W, SIM_H } from './reaction-diffusion-gpu';
import styles from './ReactionDiffusion.module.css';

// ─── Simulation constants ─────────────────────────────────────────────────────

const BG = '#030d0d';

// ─── Presets ──────────────────────────────────────────────────────────────────

interface Preset {
  label: string;
  desc:  string;
  f: number;
  k: number;
  Du: number;
  Dv: number;
  scattered: boolean;
}

const PRESETS: Preset[] = [
  {
    label: 'Spots',
    desc:  'Isolated spots / pearls form and stabilise - a Turing instability in its simplest form.',
    f: 0.035, k: 0.065, Du: 0.2097, Dv: 0.105, scattered: false,
  },
  {
    label: 'Fingerprints',
    desc:  'Dense interlocking stripes reminiscent of fingerprint ridges or zebra-fish markings.',
    f: 0.037, k: 0.060, Du: 0.2097, Dv: 0.105, scattered: false,
  },
  {
    label: 'Coral',
    desc:  'Branching coral-like tendrils grow outward from each seed, splitting as they expand.',
    f: 0.062, k: 0.061, Du: 0.2097, Dv: 0.105, scattered: true,
  },
  {
    label: 'Mitosis',
    desc:  'Each spot pinches in two and the daughter spots repeat - a model of cell division.',
    f: 0.028, k: 0.053, Du: 0.2097, Dv: 0.105, scattered: false,
  },
  {
    label: 'Maze',
    desc:  'Thin, winding walls fill the space with an intricate corridor-like maze.',
    f: 0.029, k: 0.057, Du: 0.2097, Dv: 0.105, scattered: true,
  },
  {
    label: 'Worms',
    desc:  'Long, slowly writhing worm-like stripes meander across the field.',
    f: 0.058, k: 0.065, Du: 0.2097, Dv: 0.105, scattered: true,
  },
];

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Seed a circular blob of activator V (≈1) in the grid at (cx,cy) with radius r. */
function seedBlob(
  u: Float32Array, v: Float32Array,
  cx: number, cy: number, r: number,
  W: number, H: number,
) {
  const r2 = r * r;
  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      if (dx * dx + dy * dy > r2) continue;
      const x = ((cx + dx) % W + W) % W;
      const y = ((cy + dy) % H + H) % H;
      const i = y * W + x;
      u[i] = 0.5 + (Math.random() - 0.5) * 0.1;
      v[i] = 0.25 + Math.random() * 0.05;
    }
  }
}

function buildInitial(preset: Preset): { u: Float32Array; v: Float32Array } {
  const u = new Float32Array(SIM_W * SIM_H).fill(1);
  const v = new Float32Array(SIM_W * SIM_H);

  const cx = SIM_W >> 1;
  const cy = SIM_H >> 1;

  if (preset.scattered) {
    // Scatter many small seeds
    const n = 40 + Math.round(Math.random() * 20);
    for (let i = 0; i < n; i++) {
      seedBlob(u, v,
        4 + Math.floor(Math.random() * (SIM_W - 8)),
        4 + Math.floor(Math.random() * (SIM_H - 8)),
        3 + Math.floor(Math.random() * 4),
        SIM_W, SIM_H,
      );
    }
  } else {
    // Central seed plus a few random ones
    seedBlob(u, v, cx,     cy,     8, SIM_W, SIM_H);
    seedBlob(u, v, cx - 50, cy + 30, 6, SIM_W, SIM_H);
    seedBlob(u, v, cx + 60, cy - 40, 6, SIM_W, SIM_H);
    seedBlob(u, v, cx - 70, cy - 50, 5, SIM_W, SIM_H);
    seedBlob(u, v, cx + 40, cy + 70, 5, SIM_W, SIM_H);
  }

  return { u, v };
}

// ─── Live params ref type ─────────────────────────────────────────────────────

interface LiveParams {
  running:    boolean;
  f:          number;
  k:          number;
  Du:         number;
  Dv:         number;
  stepsFrame: number;
  brushSize:  number;
  showGrid:   boolean;
  eraseMode:  boolean;
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function ReactionDiffusion() {
  const [searchParams, setSearchParams] = useSearchParams();

  const [activePreset,  setActivePreset]  = useState(() => getNumParam(searchParams, 'preset', 0) | 0);
  const [running,       setRunning]       = useState(true);
  const [f,             setF]             = useState(() => getNumParam(searchParams, 'f',     PRESETS[0].f));
  const [k,             setK]             = useState(() => getNumParam(searchParams, 'k',     PRESETS[0].k));
  const [Du,            setDu]            = useState(() => getNumParam(searchParams, 'du',    PRESETS[0].Du));
  const [Dv,            setDv]            = useState(() => getNumParam(searchParams, 'dv',    PRESETS[0].Dv));
  const [stepsFrame,    setStepsFrame]    = useState(() => getNumParam(searchParams, 'steps', 8) | 0);
  const [brushSize,     setBrushSize]     = useState(6);
  const [showInfo,      setShowInfo]      = useState(false);
  const [showExport,    setShowExport]    = useState(false);
  const [generation,    setGeneration]    = useState(0);
  const [showGrid,      setShowGrid]      = useState(false);
  const [gpuError,      setGpuError]      = useState<string | null>(null);
  const [eraseMode,     setEraseMode]     = useState(false);

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef    = useRef<HTMLCanvasElement>(null);
  const gpuRef       = useRef<ReactionDiffusionGPU | null>(null);

  const rafRef     = useRef(0);
  const genRef     = useRef(0);
  const pRef       = useRef<LiveParams>({ running, f, k, Du, Dv, stepsFrame, brushSize, showGrid, eraseMode });
  const isPainting = useRef(false);
  const isErasing  = useRef(false);
  const initPresetRef  = useRef(activePreset);

  const { isFullscreen, toggleFullscreen } = useFullscreen(containerRef);

  // Sync live params ref
  useEffect(() => {
    pRef.current = { running, f, k, Du, Dv, stepsFrame, brushSize, showGrid, eraseMode };
  }, [running, f, k, Du, Dv, stepsFrame, brushSize, showGrid, eraseMode]);

  // Persist params to URL (replace so no extra history entries)
  useEffect(() => {
    setSearchParams(sp => {
      sp.set('f',      f.toFixed(4));
      sp.set('k',      k.toFixed(4));
      sp.set('du',     Du.toFixed(4));
      sp.set('dv',     Dv.toFixed(4));
      sp.set('steps',  String(stepsFrame));
      sp.set('preset', String(activePreset));
      return sp;
    }, { replace: true });
  }, [f, k, Du, Dv, stepsFrame, activePreset, setSearchParams]);

  // ── RAF loop (initialises GPU + runs simulation) ────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let gpu: ReactionDiffusionGPU;
    try {
      const safeIdx = Math.max(0, Math.min(initPresetRef.current, PRESETS.length - 1));
      const { u: initU, v: initV } = buildInitial(PRESETS[safeIdx]);
      gpu = new ReactionDiffusionGPU(initU, initV);
      gpuRef.current = gpu;
    } catch (e) {
      setGpuError((e as Error).message);
      return;
    }

    let lastGenUpdate = 0;

    function frame() {
      rafRef.current = requestAnimationFrame(frame);
      const p = pRef.current;

      if (p.running) {
        gpu.step({ f: p.f, k: p.k, Du: p.Du, Dv: p.Dv }, p.stepsFrame);
        genRef.current += p.stepsFrame;
      }

      gpu.render();

      const ctx = canvas!.getContext('2d');
      if (!ctx) return;
      const W = canvas!.width, H = canvas!.height;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(gpu.canvas, 0, 0, W, H);

      // Grid overlay
      if (p.showGrid) {
        const cw = W / SIM_W;
        const ch = H / SIM_H;
        ctx.strokeStyle = 'rgba(255,255,255,0.04)';
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        for (let x = 0; x <= SIM_W; x++) { ctx.moveTo(x * cw, 0); ctx.lineTo(x * cw, H); }
        for (let y = 0; y <= SIM_H; y++) { ctx.moveTo(0, y * ch); ctx.lineTo(W, y * ch); }
        ctx.stroke();
      }

      if (genRef.current - lastGenUpdate >= 50) {
        lastGenUpdate = genRef.current;
        setGeneration(genRef.current);
      }
    }

    rafRef.current = requestAnimationFrame(frame);
    return () => {
      cancelAnimationFrame(rafRef.current);
      gpu.dispose();
      gpuRef.current = null;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Reset / preset ──────────────────────────────────────────────────────────
  const reset = useCallback((presetIdx = activePreset) => {
    const gpu = gpuRef.current;
    if (!gpu) return;
    const safeIdx = presetIdx >= 0 ? presetIdx : 0;
    const { u: initU, v: initV } = buildInitial(PRESETS[safeIdx]);
    gpu.reset(initU, initV);
    genRef.current = 0;
    setGeneration(0);
  }, [activePreset]);

  const goToPreset = useCallback((idx: number) => {
    const gpu = gpuRef.current;
    if (!gpu) return;
    const preset = PRESETS[idx];
    setActivePreset(idx);
    setF(preset.f);
    setK(preset.k);
    setDu(preset.Du);
    setDv(preset.Dv);
    pRef.current = { ...pRef.current, f: preset.f, k: preset.k, Du: preset.Du, Dv: preset.Dv };
    const { u: initU, v: initV } = buildInitial(preset);
    gpu.reset(initU, initV);
    genRef.current = 0;
    setGeneration(0);
  }, []);

  // ── Canvas resize ───────────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width  = Math.round(canvas.offsetWidth  * dpr);
      canvas.height = Math.round(canvas.offsetHeight * dpr);
    });
    ro.observe(canvas);
    return () => ro.disconnect();
  }, []);

  // ── Keyboard shortcuts ──────────────────────────────────────────────────────
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;
      if (e.code === 'Space') { e.preventDefault(); setRunning(r => !r); }
      if (e.code === 'KeyR')  { reset(); }
      if (e.code === 'KeyF')  { toggleFullscreen(); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [reset, toggleFullscreen]);

  // ── Painting interaction ────────────────────────────────────────────────────
  function canvasToSim(canvas: HTMLCanvasElement, clientX: number, clientY: number) {
    const rect = canvas.getBoundingClientRect();
    const rx = (clientX - rect.left) / rect.width;
    const ry = (clientY - rect.top)  / rect.height;
    return {
      sx: Math.round(rx * SIM_W),
      sy: Math.round(ry * SIM_H),
    };
  }

  function doPaint(canvas: HTMLCanvasElement, clientX: number, clientY: number) {
    const gpu = gpuRef.current;
    if (!gpu) return;
    const { sx, sy } = canvasToSim(canvas, clientX, clientY);
    if (isErasing.current) {
      gpu.eraseAt(sx, sy, pRef.current.brushSize);
    } else {
      gpu.seedAt(sx, sy, pRef.current.brushSize);
    }
  }

  function onPointerDown(e: React.PointerEvent<HTMLCanvasElement>) {
    e.preventDefault();
    isPainting.current = true;
    // Right-click inverts the current mode; left-click uses the mode toggle
    isErasing.current = e.button === 2 ? !pRef.current.eraseMode : pRef.current.eraseMode;
    (e.target as HTMLCanvasElement).setPointerCapture(e.pointerId);
    doPaint(e.currentTarget, e.clientX, e.clientY);
  }

  function onPointerMove(e: React.PointerEvent<HTMLCanvasElement>) {
    if (!isPainting.current) return;
    doPaint(e.currentTarget, e.clientX, e.clientY);
  }

  function onPointerUp() {
    isPainting.current = false;
  }

  // ── Export ──────────────────────────────────────────────────────────────────
  function handleExport({ width, height, format }: { width: number; height: number; format: import('../../lib/exportImage').ExportFormat }) {
    const canvas = canvasRef.current;
    if (canvas) exportImage(canvas, width, height, format, 'reaction-diffusion');
  }

  return (
    <div ref={containerRef} className={styles.container} style={{ background: BG }}>
      {/* ── GPU error fallback ──────────────────────────────────────────────── */}
      {gpuError && (
        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '2rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: 'var(--text-sm)' }}>
          <p>GPU simulation unavailable:<br /><span style={{ color: 'var(--col-reaction)', fontFamily: 'var(--font-mono)' }}>{gpuError}</span></p>
        </div>
      )}
      {/* ── Canvas ─────────────────────────────────────────────────────────── */}
      <canvas
        ref={canvasRef}
        className={styles.canvas}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
        onContextMenu={e => e.preventDefault()}
      />

      {/* ── Sidebar ────────────────────────────────────────────────────────── */}
      <div className={styles.sidebar}>
        <div className={styles.sidebarPanels}>

          <ControlPanel title="Preset">
            <ControlGroup>
              <div className={styles.presetGrid}>
                {PRESETS.map((preset, idx) => (
                  <button
                    key={preset.label}
                    className={[styles.presetBtn, activePreset === idx ? styles.presetBtnActive : ''].join(' ')}
                    onClick={() => goToPreset(idx)}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
              <p className={styles.presetDesc}>{PRESETS[activePreset]?.desc ?? 'Custom parameters.'}</p>
            </ControlGroup>
          </ControlPanel>

          <ControlPanel title="Parameters">
            <ControlGroup>
              <Slider label="Feed rate (f)" value={f}  onChange={v => { setF(v); setActivePreset(-1); }} min={0.01}  max={0.10}  step={0.001} format={v => v.toFixed(3)} />
              <Slider label="Kill rate (k)" value={k}  onChange={v => { setK(v); setActivePreset(-1); }} min={0.04}  max={0.072} step={0.001} format={v => v.toFixed(3)} />
              <Slider label="Diffusion U"   value={Du} onChange={v => setDu(v)}                         min={0.10}  max={0.30}  step={0.005} format={v => v.toFixed(3)} />
              <Slider label="Diffusion V"   value={Dv} onChange={v => setDv(v)}                         min={0.04}  max={0.16}  step={0.005} format={v => v.toFixed(3)} />
            </ControlGroup>
          </ControlPanel>

          <ControlPanel title="Display">
            <ControlGroup>
              <Slider label="Steps / frame" value={stepsFrame} onChange={v => setStepsFrame(v)} min={1} max={48} step={1} />
              <Slider label="Brush size"    value={brushSize}  onChange={v => setBrushSize(v)}  min={2} max={20} step={1} />
              <Toggle label="Grid overlay"  value={showGrid}   onChange={setShowGrid} />
              <Toggle label="Erase mode"    value={eraseMode}  onChange={setEraseMode} />
            </ControlGroup>
          </ControlPanel>

        </div>

        <div className={styles.sidebarActions}>
          <SimControls
            running={running}
            onToggle={() => setRunning(r => !r)}
            onReset={() => reset(activePreset < 0 ? 0 : activePreset)}
            onExport={() => setShowExport(true)}
          />
        </div>
      </div>

      {/* ── HUD ────────────────────────────────────────────────────────────── */}
      <div className={styles.hud}>
        <div className={styles.hudLeft}>
          <span className={styles.hudTitle}>Reaction Diffusion</span>
          <span className={styles.hudSub}>gen {generation.toLocaleString()}</span>
        </div>
        <div className={styles.hudRight}>
          <span className={styles.hudHint}>
            {eraseMode ? 'drag to erase · right-click to seed' : 'drag to seed · right-click to erase'}
          </span>
          <button className={styles.infoBtn} onClick={() => setShowInfo(true)} title="About">ℹ</button>
          <button className={styles.hudBtn}  onClick={toggleFullscreen} title="Fullscreen (F)">
            {isFullscreen ? '⤡' : '⤢'}
          </button>
        </div>
      </div>

      {/* ── Dialogs ────────────────────────────────────────────────────────── */}
      {showInfo && (
        <InfoDialog title="Reaction Diffusion (Gray-Scott)" onClose={() => setShowInfo(false)}>
          <p>
            Two chemicals, U (inhibitor) and V (activator), diffuse and react across a 2D grid.
            Their interaction is governed by the <strong>Gray-Scott equations</strong>:
          </p>
          <p style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)', lineHeight: 1.8 }}>
            ∂U/∂t = Dᵤ∇²U − UV² + f(1−U)<br />
            ∂V/∂t = D꜀∇²V + UV² − (f+k)V
          </p>
          <p>
            <strong>f</strong> (feed) controls how fast U is replenished.
            <strong> k</strong> (kill) controls how fast V decays.
            Changing just these two values produces spots, stripes, coral, mazes, and self-replicating blobs.
          </p>
          <p>
            This is a numerical model of <strong>Turing instability</strong>: a uniform state
            spontaneously breaks symmetry into spatial patterns, explaining animal coat markings
            and seashell pigmentation.
          </p>
          <p>
            <strong>Click or drag</strong> on the canvas to inject new activator seeds.
          </p>
        </InfoDialog>
      )}
      {showExport && (
        <ExportDialog
          onClose={() => setShowExport(false)}
          onDownload={handleExport}
        />
      )}
    </div>
  );
}
