import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ControlGroup, ControlPanel, SimControls, Slider, Toggle,
} from '@/components/Controls';
import { InfoDialog } from '@/components/InfoDialog';
import { useFullscreen } from '@/hooks/useFullscreen';
import ExportDialog from '../../components/ExportDialog/ExportDialog';
import { exportImage } from '../../lib/exportImage';
import styles from './ReactionDiffusion.module.css';

// ─── Simulation constants ─────────────────────────────────────────────────────

const SIM_W = 256;
const SIM_H = 256;
const BG    = '#030d0d';

// ─── Presets ──────────────────────────────────────────────────────────────────

interface Preset {
  label: string;
  desc:  string;
  f: number;
  k: number;
  Du: number;
  Dv: number;
}

const PRESETS: Preset[] = [
  {
    label: 'Spots',
    desc:  'Isolated spots / pearls form and stabilise - a Turing instability in its simplest form.',
    f: 0.035, k: 0.065, Du: 0.2097, Dv: 0.105,
  },
  {
    label: 'Fingerprints',
    desc:  'Dense interlocking stripes reminiscent of fingerprint ridges or zebra-fish markings.',
    f: 0.037, k: 0.060, Du: 0.2097, Dv: 0.105,
  },
  {
    label: 'Coral',
    desc:  'Branching coral-like tendrils grow outward from each seed, splitting as they expand.',
    f: 0.062, k: 0.061, Du: 0.2097, Dv: 0.105,
  },
  {
    label: 'Mitosis',
    desc:  'Each spot pinches in two and the daughter spots repeat - a model of cell division.',
    f: 0.028, k: 0.053, Du: 0.2097, Dv: 0.105,
  },
  {
    label: 'Maze',
    desc:  'Thin, winding walls fill the space with an intricate corridor-like maze.',
    f: 0.029, k: 0.057, Du: 0.2097, Dv: 0.105,
  },
  {
    label: 'Worms',
    desc:  'Long, slowly writhing worm-like stripes meander across the field.',
    f: 0.058, k: 0.065, Du: 0.2097, Dv: 0.105,
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

  if (preset.label === 'Coral' || preset.label === 'Maze' || preset.label === 'Worms') {
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

// ─── Gray-Scott step ──────────────────────────────────────────────────────────

function gsStep(
  u: Float32Array, v: Float32Array,
  nu: Float32Array, nv: Float32Array,
  f: number, k: number, Du: number, Dv: number,
  W: number, H: number,
) {
  for (let y = 0; y < H; y++) {
    const yW  = y * W;
    const ynW = (y === 0 ? H - 1 : y - 1) * W;
    const ysW = (y === H - 1 ? 0 : y + 1) * W;

    for (let x = 0; x < W; x++) {
      const i  = yW + x;
      const xL = x === 0     ? W - 1 : x - 1;
      const xR = x === W - 1 ? 0     : x + 1;

      const ui = u[i];
      const vi = v[i];

      const lapU = u[yW + xL] + u[yW + xR] + u[ynW + x] + u[ysW + x] - 4 * ui;
      const lapV = v[yW + xL] + v[yW + xR] + v[ynW + x] + v[ysW + x] - 4 * vi;

      const uvv = ui * vi * vi;

      let nu_ = ui + Du * lapU - uvv + f * (1 - ui);
      let nv_ = vi + Dv * lapV + uvv - (f + k) * vi;

      if (nu_ < 0) nu_ = 0; else if (nu_ > 1) nu_ = 1;
      if (nv_ < 0) nv_ = 0; else if (nv_ > 1) nv_ = 1;

      nu[i] = nu_;
      nv[i] = nv_;
    }
  }
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
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function ReactionDiffusion() {
  const [activePreset,  setActivePreset]  = useState(0);
  const [running,       setRunning]       = useState(true);
  const [f,             setF]             = useState(PRESETS[0].f);
  const [k,             setK]             = useState(PRESETS[0].k);
  const [Du,            setDu]            = useState(PRESETS[0].Du);
  const [Dv,            setDv]            = useState(PRESETS[0].Dv);
  const [stepsFrame,    setStepsFrame]    = useState(8);
  const [brushSize,     setBrushSize]     = useState(6);
  const [showInfo,      setShowInfo]      = useState(false);
  const [showExport,    setShowExport]    = useState(false);
  const [generation,    setGeneration]    = useState(0);
  const [showGrid,      setShowGrid]      = useState(false);

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef    = useRef<HTMLCanvasElement>(null);
  const offRef       = useRef<OffscreenCanvas | null>(null);
  const offCtxRef    = useRef<OffscreenCanvasRenderingContext2D | null>(null);
  const imgDataRef   = useRef<ImageData | null>(null);

  const uRef  = useRef(new Float32Array(SIM_W * SIM_H).fill(1));
  const vRef  = useRef(new Float32Array(SIM_W * SIM_H));
  const nuRef = useRef(new Float32Array(SIM_W * SIM_H));
  const nvRef = useRef(new Float32Array(SIM_W * SIM_H));

  const rafRef   = useRef(0);
  const genRef   = useRef(0);
  const pRef     = useRef<LiveParams>({ running, f, k, Du, Dv, stepsFrame, brushSize, showGrid });
  const isPainting = useRef(false);

  const { isFullscreen, toggleFullscreen } = useFullscreen(containerRef);

  // Sync live params ref
  useEffect(() => {
    pRef.current = { running, f, k, Du, Dv, stepsFrame, brushSize, showGrid };
  }, [running, f, k, Du, Dv, stepsFrame, brushSize, showGrid]);

  // ── Initialise offscreen canvas + image data ────────────────────────────────
  useEffect(() => {
    const off = new OffscreenCanvas(SIM_W, SIM_H);
    offRef.current = off;
    offCtxRef.current = off.getContext('2d')!;
    imgDataRef.current = new ImageData(SIM_W, SIM_H);
  }, []);

  // ── Reset to a preset ───────────────────────────────────────────────────────
  const reset = useCallback((presetIdx = activePreset) => {
    const preset = PRESETS[presetIdx];
    const { u: initU, v: initV } = buildInitial(preset);
    uRef.current.set(initU);
    vRef.current.set(initV);
    nuRef.current.fill(0);
    nvRef.current.fill(0);
    genRef.current = 0;
    setGeneration(0);
  }, [activePreset]);

  const goToPreset = useCallback((idx: number) => {
    const preset = PRESETS[idx];
    setActivePreset(idx);
    setF(preset.f);
    setK(preset.k);
    setDu(preset.Du);
    setDv(preset.Dv);
    pRef.current = { ...pRef.current, f: preset.f, k: preset.k, Du: preset.Du, Dv: preset.Dv };
    const { u: initU, v: initV } = buildInitial(preset);
    uRef.current.set(initU);
    vRef.current.set(initV);
    nuRef.current.fill(0);
    nvRef.current.fill(0);
    genRef.current = 0;
    setGeneration(0);
  }, []);

  // ── RAF loop ─────────────────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let lastGenUpdate = 0;

    function frame() {
      rafRef.current = requestAnimationFrame(frame);
      const p = pRef.current;

      if (p.running) {
        const steps = p.stepsFrame;
        for (let s = 0; s < steps; s++) {
          gsStep(
            uRef.current, vRef.current,
            nuRef.current, nvRef.current,
            p.f, p.k, p.Du, p.Dv,
            SIM_W, SIM_H,
          );
          // Swap buffers
          const tmpU = uRef.current; uRef.current = nuRef.current; nuRef.current = tmpU;
          const tmpV = vRef.current; vRef.current = nvRef.current; nvRef.current = tmpV;
          genRef.current++;
        }
      }

      // Render
      const off = offRef.current;
      const offCtx = offCtxRef.current;
      const imgData = imgDataRef.current;
      if (!off || !offCtx || !imgData) return;

      const { data } = imgData;
      const v = vRef.current;
      for (let i = 0; i < SIM_W * SIM_H; i++) {
        const t = v[i];
        const idx = i << 2;
        data[idx]     = Math.round(3  + t * 20);
        data[idx + 1] = Math.round(13 + t * 195);
        data[idx + 2] = Math.round(13 + t * 178);
        data[idx + 3] = 255;
      }
      offCtx.putImageData(imgData, 0, 0);

      const ctx = canvas!.getContext('2d');
      if (!ctx) return;
      const W = canvas!.width, H = canvas!.height;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(off, 0, 0, W, H);

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

      // Throttle gen counter update
      if (genRef.current - lastGenUpdate >= 50) {
        lastGenUpdate = genRef.current;
        setGeneration(genRef.current);
      }
    }

    // Initial seed
    const preset = PRESETS[0];
    const { u: initU, v: initV } = buildInitial(preset);
    uRef.current.set(initU);
    vRef.current.set(initV);

    rafRef.current = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(rafRef.current);
  // eslint-disable-next-line react-hooks/exhaustive-deps
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

  function paint(canvas: HTMLCanvasElement, clientX: number, clientY: number) {
    const { sx, sy } = canvasToSim(canvas, clientX, clientY);
    seedBlob(uRef.current, vRef.current, sx, sy, pRef.current.brushSize, SIM_W, SIM_H);
  }

  function onPointerDown(e: React.PointerEvent<HTMLCanvasElement>) {
    isPainting.current = true;
    (e.target as HTMLCanvasElement).setPointerCapture(e.pointerId);
    paint(e.currentTarget, e.clientX, e.clientY);
  }

  function onPointerMove(e: React.PointerEvent<HTMLCanvasElement>) {
    if (!isPainting.current) return;
    paint(e.currentTarget, e.clientX, e.clientY);
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
      {/* ── Canvas ─────────────────────────────────────────────────────────── */}
      <canvas
        ref={canvasRef}
        className={styles.canvas}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
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
              <Slider label="Steps / frame" value={stepsFrame} onChange={v => setStepsFrame(v)} min={1} max={24} step={1} />
              <Slider label="Brush size"    value={brushSize}  onChange={v => setBrushSize(v)}  min={2} max={20} step={1} />
              <Toggle label="Grid overlay"  value={showGrid}   onChange={setShowGrid} />
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
          <span className={styles.hudHint}>click / drag to seed</span>
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
