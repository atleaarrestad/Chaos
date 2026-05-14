import { useState, useEffect, useRef, useCallback } from 'react';
import { Info } from 'lucide-react';
import {
  Slider, Toggle, SelectControl,
  ControlPanel, ControlGroup,
} from '@/components/Controls';
import { detectWebGL2 } from '@/lib/gpu/context';
import styles from './DoublePendulum.module.css';
import { DoublePendulumGPU, type DoublePendulumParams, type ColorMode } from './double-pendulum-gpu';

// ─── Constants ────────────────────────────────────────────────────────────────

const STEPS_PER_FRAME = 4; // must match GPU shader MAX_STEPS usage
const RAD = Math.PI / 180;
const DEG = 180 / Math.PI;

const DEFAULT_THETA1 = 50 * RAD;   // gentle swing, below flip threshold
const DEFAULT_THETA2 = -20 * RAD;  // slight offset on second rod
const DEFAULT_OMEGA1 = 0;
const DEFAULT_OMEGA2 = 0;
const DEFAULT_G      = 9.81;
const DEFAULT_DT     = 0.001;
const DEFAULT_SPEED  = 1;          // real-time by default
const DEFAULT_SPREAD = 0.01;
const DEFAULT_TRAIL  = 15.0;  // seconds of trail

const MAX_PHASE_PTS  = 5_000;
const MAX_TRAIL_PTS  = 72_000;  // (x,y,t) per point — covers 600s at 120Hz
const TRAIL_BUCKETS  = 40;      // alpha groups drawn per frame
const BG_COLOR = '#050510';

// ─── Presets ──────────────────────────────────────────────────────────────────

const PRESETS = [
  {
    label: 'Gentle',
    desc: 'Calm arcs, low energy',
    theta1: 50 * RAD, theta2: -20 * RAD, omega1: 0, omega2: 0,
    g: 9.81, dt: 0.001, speed: 1, trailSecs: 4,
  },
  {
    label: 'Classic',
    desc: 'Textbook chaos starting position',
    theta1: 120 * RAD, theta2: 120 * RAD, omega1: 0, omega2: 0,
    g: 9.81, dt: 0.001, speed: 1, trailSecs: 6,
  },
  {
    label: 'Near Tip',
    desc: 'Balanced near top — falls dramatically',
    theta1: 175 * RAD, theta2: 5 * RAD, omega1: 0, omega2: 0,
    g: 9.81, dt: 0.001, speed: 1, trailSecs: 8,
  },
  {
    label: 'Mirror',
    desc: 'Symmetric arms, beautiful folded trails',
    theta1: 110 * RAD, theta2: -110 * RAD, omega1: 0, omega2: 0,
    g: 9.81, dt: 0.001, speed: 1, trailSecs: 6,
  },
  {
    label: 'Windmill',
    desc: 'Angular momentum sends arm spinning',
    theta1: 20 * RAD, theta2: 20 * RAD, omega1: 6, omega2: 0,
    g: 9.81, dt: 0.001, speed: 1, trailSecs: 5,
  },
  {
    label: 'Freefall',
    desc: 'Straight down, pure velocity — maximum chaos',
    theta1: 0, theta2: 0, omega1: 8, omega2: 0,
    g: 9.81, dt: 0.001, speed: 1, trailSecs: 4,
  },
] as const;

// ─── CPU RK4 (reference pendulum) ─────────────────────────────────────────────

function cpuRK4(
  th1: number, om1: number, th2: number, om2: number,
  dt: number, g: number,
): [number, number, number, number] {
  const deriv = (th1: number, om1: number, th2: number, om2: number) => {
    const d = th1 - th2, sd = Math.sin(d), cd = Math.cos(d);
    const denom = 3 - Math.cos(2 * d);
    const a1 = (-3*g*Math.sin(th1) - g*Math.sin(th1-2*th2) - 2*sd*(om2*om2+om1*om1*cd)) / denom;
    const a2 = (2*sd*(2*om1*om1 + 2*g*Math.cos(th1) + om2*om2*cd)) / denom;
    return [om1, a1, om2, a2] as const;
  };
  const h = dt / 2;
  const [k1a,k1b,k1c,k1d] = deriv(th1, om1, th2, om2);
  const [k2a,k2b,k2c,k2d] = deriv(th1+k1a*h, om1+k1b*h, th2+k1c*h, om2+k1d*h);
  const [k3a,k3b,k3c,k3d] = deriv(th1+k2a*h, om1+k2b*h, th2+k2c*h, om2+k2d*h);
  const [k4a,k4b,k4c,k4d] = deriv(th1+k3a*dt, om1+k3b*dt, th2+k3c*dt, om2+k3d*dt);
  const s = dt / 6;
  return [
    th1 + (k1a+2*k2a+2*k3a+k4a)*s,
    om1 + (k1b+2*k2b+2*k3b+k4b)*s,
    th2 + (k1c+2*k2c+2*k3c+k4c)*s,
    om2 + (k1d+2*k2d+2*k3d+k4d)*s,
  ];
}

// ─── Params live-ref type ─────────────────────────────────────────────────────

interface LiveParams {
  theta1: number; omega1: number; theta2: number; omega2: number;
  g: number; dt: number; speed: number; spread: number;
  trailSecs: number; pointSize: number; colorMode: ColorMode;
  running: boolean; showPendulum: boolean; showPhase: boolean; showEnsemble: boolean;
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function DoublePendulum() {
  // ── Physics params ──────────────────────────────────────────────────────────
  const [theta1,  setTheta1]  = useState(DEFAULT_THETA1);
  const [theta2,  setTheta2]  = useState(DEFAULT_THETA2);
  const [omega1,  setOmega1]  = useState(DEFAULT_OMEGA1);
  const [omega2,  setOmega2]  = useState(DEFAULT_OMEGA2);
  const [g,       setG]       = useState(DEFAULT_G);
  const [spread,  setSpread]  = useState(DEFAULT_SPREAD);

  // ── Animation params ────────────────────────────────────────────────────────
  const [dt,        setDt]        = useState(DEFAULT_DT);
  const [speed,     setSpeed]     = useState(DEFAULT_SPEED);
  const [trailSecs, setTrailSecs] = useState(DEFAULT_TRAIL);
  const [pointSize, setPointSize] = useState(2.0);
  const [running,   setRunning]   = useState(true);

  // ── Display params ──────────────────────────────────────────────────────────
  const [colorMode,     setColorMode]     = useState<ColorMode>('heat');
  const [showPendulum,  setShowPendulum]  = useState(true);
  const [showPhase,     setShowPhase]     = useState(true);
  const [showEnsemble,  setShowEnsemble]  = useState(false);
  const [gpuAvailable,  setGpuAvailable]  = useState(false);

  // ── Refs ────────────────────────────────────────────────────────────────────
  const canvasRef      = useRef<HTMLCanvasElement>(null);
  const phasePanelRef  = useRef<HTMLCanvasElement>(null);
  const sidebarRef     = useRef<HTMLDivElement>(null);
  const rafRef         = useRef(0);
  const gpuRef         = useRef<DoublePendulumGPU | null>(null);

  /** Off-screen canvas for accumulated trail (GPU ensemble scatter). */
  const trailCanvasRef = useRef<HTMLCanvasElement>(document.createElement('canvas'));

  /** Ring buffer of recent bob2 positions: (x, y, timestamp_sec) triples. */
  const trailBufRef   = useRef(new Float32Array(MAX_TRAIL_PTS * 3));
  const trailHeadRef  = useRef(0);
  const trailCountRef = useRef(0);

  /** CPU reference pendulum: exact initial conditions (no spread). */
  const refRef = useRef({ th1: DEFAULT_THETA1, om1: DEFAULT_OMEGA1, th2: DEFAULT_THETA2, om2: DEFAULT_OMEGA2 });

  /** Ring buffer for (θ₁_wrapped, ω₁) phase portrait. */
  const phaseBufRef = useRef(new Float32Array(MAX_PHASE_PTS * 2));
  const phaseHeadRef  = useRef(0);
  const phaseTotalRef = useRef(0);
  const phaseMaxOmRef = useRef(8);  // auto-scale ω axis

  /** Mutable params read by RAF loop to avoid stale closures. */
  const pRef = useRef<LiveParams>({
    theta1, theta2, omega1, omega2, g, dt, speed, spread,
    trailSecs, pointSize, colorMode, running, showPendulum, showPhase, showEnsemble,
  });

  useEffect(() => {
    pRef.current = {
      theta1, theta2, omega1, omega2, g, dt, speed, spread,
      trailSecs, pointSize, colorMode, running, showPendulum, showPhase, showEnsemble,
    };
  }, [theta1, theta2, omega1, omega2, g, dt, speed, spread,
      trailSecs, pointSize, colorMode, running, showPendulum, showPhase, showEnsemble]);

  // ── Helpers ─────────────────────────────────────────────────────────────────

  const gpuParams = useCallback((): DoublePendulumParams => {
    const p = pRef.current;
    return { theta1: p.theta1, omega1: p.omega1, theta2: p.theta2, omega2: p.omega2,
             g: p.g, dt: p.dt, spread: p.spread, pointSize: p.pointSize, colorMode: p.colorMode };
  }, []);

  const resetSimulation = useCallback(() => {
    const p = pRef.current;
    gpuRef.current?.reset(gpuParams());
    refRef.current = { th1: p.theta1, om1: p.omega1, th2: p.theta2, om2: p.omega2 };
    phaseHeadRef.current  = 0;
    phaseTotalRef.current = 0;
    phaseMaxOmRef.current = 8;
    trailHeadRef.current  = 0;
    trailCountRef.current = 0;
    // Clear trail canvas
    const tc = trailCanvasRef.current;
    const tCtx = tc.getContext('2d');
    if (tCtx) {
      tCtx.fillStyle = BG_COLOR;
      tCtx.fillRect(0, 0, tc.width, tc.height);
    }
  }, [gpuParams]);

  const goToPreset = useCallback((preset: typeof PRESETS[number]) => {
    setTheta1(preset.theta1);
    setTheta2(preset.theta2);
    setOmega1(preset.omega1);
    setOmega2(preset.omega2);
    setG(preset.g);
    setDt(preset.dt);
    setSpeed(preset.speed);
    setTrailSecs(preset.trailSecs);
    // refRef and trail are reset by the useEffect that watches these state changes
  }, []);


  useEffect(() => {
    const available = detectWebGL2();
    setGpuAvailable(available);
    if (!available) return;
    try {
      gpuRef.current = new DoublePendulumGPU(gpuParams());
    } catch (err) {
      console.warn('DoublePendulumGPU init failed:', err);
      setGpuAvailable(false);
    }
    return () => {
      gpuRef.current?.dispose();
      gpuRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Reset when physics initial conditions change
  useEffect(() => {
    if (gpuRef.current) resetSimulation();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theta1, theta2, omega1, omega2, g, spread]);

  // ── Draw helpers ─────────────────────────────────────────────────────────────

  /** Draw rods and bobs of the CPU reference pendulum. */
  function drawPendulumGeometry(
    ctx: CanvasRenderingContext2D,
    W: number, H: number,
    th1: number, th2: number,
  ) {
    const scale  = Math.min(W, H) / 4.4;
    const px     = W / 2;
    const py     = H / 2;

    const b1x = px + Math.sin(th1) * scale;
    const b1y = py + Math.cos(th1) * scale;
    const b2x = b1x + Math.sin(th2) * scale;
    const b2y = b1y + Math.cos(th2) * scale;

    ctx.save();
    ctx.lineCap  = 'round';
    ctx.lineJoin = 'round';

    // ── Shadow / glow for rods ──────────────────────────────────────────────
    ctx.shadowColor = '#4ade80';
    ctx.shadowBlur  = 14;

    // Rod 1
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(b1x, b1y);
    ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.lineWidth   = 2.5;
    ctx.stroke();

    // Rod 2
    ctx.beginPath();
    ctx.moveTo(b1x, b1y);
    ctx.lineTo(b2x, b2y);
    ctx.stroke();

    ctx.shadowBlur = 0;

    // ── Pivot ────────────────────────────────────────────────────────────────
    const pivotR = 4.5;
    ctx.beginPath();
    ctx.arc(px, py, pivotR, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(8, 18, 26, 0.95)';
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = 'rgba(200, 230, 215, 0.65)';
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(px - pivotR * 0.3, py - pivotR * 0.35, pivotR * 0.28, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.fill();

    // ── Bob 1 (joint) ────────────────────────────────────────────────────────
    const r1 = Math.max(5, scale * 0.05);
    ctx.shadowColor = 'rgba(74,222,128,0.55)';
    ctx.shadowBlur  = 10;
    ctx.beginPath();
    ctx.arc(b1x, b1y, r1, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(22, 62, 44, 0.97)';
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = 'rgba(74,222,128,0.85)';
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(b1x - r1 * 0.28, b1y - r1 * 0.32, r1 * 0.25, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.fill();

    // ── Bob 2 (end bob) ──────────────────────────────────────────────────────
    const r2 = Math.max(6, scale * 0.062);
    ctx.shadowColor = 'rgba(74,222,128,0.65)';
    ctx.shadowBlur  = 12;
    ctx.beginPath();
    ctx.arc(b2x, b2y, r2, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(22, 62, 44, 0.97)';
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = 'rgba(74,222,128,0.92)';
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(b2x - r2 * 0.28, b2y - r2 * 0.32, r2 * 0.25, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.fill();

    ctx.restore();
  }

  /** Draw (θ₁_wrapped, ω₁) phase portrait into the panel canvas. */
  function drawPhasePortrait(pc: HTMLCanvasElement) {
    const pCtx = pc.getContext('2d');
    if (!pCtx) return;
    const W = pc.width, H = pc.height;

    pCtx.fillStyle = '#040410';
    pCtx.fillRect(0, 0, W, H);

    const total   = phaseTotalRef.current;
    const maxOm   = phaseMaxOmRef.current;
    const toX = (th: number) => W / 2 + (th / Math.PI) * (W / 2 - 4);
    const toY = (om: number) => H / 2 - (om / maxOm) * (H / 2 - 4);

    // Axis grid
    pCtx.strokeStyle = 'rgba(255,255,255,0.08)';
    pCtx.lineWidth   = 1;
    pCtx.beginPath();
    pCtx.moveTo(0, H / 2); pCtx.lineTo(W, H / 2);
    pCtx.moveTo(W / 2, 0); pCtx.lineTo(W / 2, H);
    pCtx.stroke();

    // ±π vertical guidelines
    pCtx.strokeStyle = 'rgba(255,255,255,0.04)';
    [toX(-Math.PI), toX(Math.PI)].forEach(x => {
      pCtx.beginPath(); pCtx.moveTo(x, 0); pCtx.lineTo(x, H); pCtx.stroke();
    });

    if (total === 0) {
      pCtx.fillStyle   = 'rgba(160,200,160,0.5)';
      pCtx.font        = `${Math.round(Math.min(W, H) * 0.07)}px system-ui, sans-serif`;
      pCtx.textAlign   = 'center';
      pCtx.textBaseline = 'middle';
      pCtx.fillText('running…', W / 2, H / 2);
      return;
    }

    const count = Math.min(total, MAX_PHASE_PTS);
    const head  = phaseHeadRef.current;

    pCtx.strokeStyle = 'rgba(74,222,128,0.7)';
    pCtx.lineWidth   = 0.8;
    pCtx.beginPath();
    for (let i = 0; i < count; i++) {
      const si = ((head - count + i) % MAX_PHASE_PTS + MAX_PHASE_PTS) % MAX_PHASE_PTS;
      const tx = phaseBufRef.current[si * 2];
      const om = phaseBufRef.current[si * 2 + 1];
      const cx = toX(tx), cy = toY(om);
      if (i === 0) pCtx.moveTo(cx, cy);
      else         pCtx.lineTo(cx, cy);
    }
    pCtx.stroke();

    // Current point (bright dot)
    const latestSi = ((head - 1) % MAX_PHASE_PTS + MAX_PHASE_PTS) % MAX_PHASE_PTS;
    const ltx = phaseBufRef.current[latestSi * 2];
    const lom = phaseBufRef.current[latestSi * 2 + 1];
    pCtx.beginPath();
    pCtx.arc(toX(ltx), toY(lom), 3, 0, Math.PI * 2);
    pCtx.fillStyle = '#fff';
    pCtx.fill();
  }

  // ── Main draw loop ───────────────────────────────────────────────────────────

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) { rafRef.current = requestAnimationFrame(draw); return; }

    const ctx = canvas.getContext('2d')!;
    const W = canvas.width, H = canvas.height;
    const p = pRef.current;
    const gpu = gpuRef.current;

    // ── Keep trail canvas in sync with main canvas size ─────────────────────
    const tc = trailCanvasRef.current;
    if (tc.width !== W || tc.height !== H) {
      tc.width  = W;
      tc.height = H;
      const tCtx2 = tc.getContext('2d');
      if (tCtx2) { tCtx2.fillStyle = BG_COLOR; tCtx2.fillRect(0, 0, W, H); }
    }

    // ── Advance simulation ───────────────────────────────────────────────────
    if (p.running) {
      // Always advance CPU reference pendulum
      let { th1, om1, th2, om2 } = refRef.current;
      for (let s = 0; s < p.speed * STEPS_PER_FRAME; s++) {
        [th1, om1, th2, om2] = cpuRK4(th1, om1, th2, om2, p.dt, p.g);
        if (!isFinite(th1) || !isFinite(om1)) { th1 = p.theta1; om1 = 0; th2 = p.theta2; om2 = 0; break; }
      }
      refRef.current = { th1, om1, th2, om2 };

      // Store phase-portrait point (wrap θ₁ to [-π, π])
      const twoPi  = 2 * Math.PI;
      const th1w   = ((th1 % twoPi) + twoPi) % twoPi;
      const th1wp  = th1w > Math.PI ? th1w - twoPi : th1w;
      const idx    = phaseHeadRef.current;
      phaseBufRef.current[idx * 2]     = th1wp;
      phaseBufRef.current[idx * 2 + 1] = om1;
      phaseHeadRef.current = (idx + 1) % MAX_PHASE_PTS;
      phaseTotalRef.current++;
      if (Math.abs(om1) > phaseMaxOmRef.current) phaseMaxOmRef.current = Math.abs(om1) * 1.15;

      // ── Update trail ──────────────────────────────────────────────────────
      const tCtx = tc.getContext('2d')!;

      if (p.showEnsemble && gpu) {
        // GPU ensemble: use the old per-frame fade (many points, less precision needed)
        const perFrame = Math.pow(0.5, 1 / Math.max(p.trailSecs * 60, 1));
        tCtx.fillStyle = `rgba(5,5,16,${(1 - perFrame).toFixed(5)})`;
        tCtx.fillRect(0, 0, W, H);
        // Advance + scatter
        const gp = gpuParams();
        for (let i = 0; i < p.speed; i++) gpu.step(gp);
        gpu.renderScatter(gp);
        const sq = Math.min(W, H);
        const ox = (W - sq) / 2;
        const oy = (H - sq) / 2;
        tCtx.drawImage(gpu.canvas, ox, oy, sq, sq);
      } else {
        // Single-pendulum trail: ring buffer → redraw each frame with explicit alpha.
        // This avoids the 8-bit canvas precision bug where tiny per-frame alphas round to 0.
        const scale = Math.min(W, H) / 4.4;
        const b1x = W / 2 + Math.sin(th1) * scale;
        const b1y = H / 2 + Math.cos(th1) * scale;
        const b2x = b1x + Math.sin(th2) * scale;
        const b2y = b1y + Math.cos(th2) * scale;

        // Store current position with wall-clock timestamp (seconds)
        const now  = performance.now() / 1000;
        const head = trailHeadRef.current;
        trailBufRef.current[head * 3]     = b2x;
        trailBufRef.current[head * 3 + 1] = b2y;
        trailBufRef.current[head * 3 + 2] = now;
        trailHeadRef.current  = (head + 1) % MAX_TRAIL_PTS;
        trailCountRef.current = Math.min(trailCountRef.current + 1, MAX_TRAIL_PTS);

        // Clear and redraw trail from buffer using actual elapsed time for ages.
        // This is frame-rate independent — works correctly at 60Hz, 120Hz, 144Hz etc.
        tCtx.fillStyle = BG_COLOR;
        tCtx.fillRect(0, 0, W, H);

        const count = trailCountRef.current;
        const H2    = trailHeadRef.current; // next-write slot = oldest slot when full

        // Find the oldest index (in ring-buffer order) that is within trailSecs
        let validStart = 0;
        for (let i = 0; i < count; i++) {
          const fi = ((H2 - count + i + MAX_TRAIL_PTS) % MAX_TRAIL_PTS) * 3;
          if (now - trailBufRef.current[fi + 2] <= p.trailSecs) { validStart = i; break; }
          validStart = i + 1;
        }
        const validCount = count - validStart;

        if (validCount > 1) {
          tCtx.lineWidth = 1.5;
          tCtx.lineCap   = 'round';
          tCtx.lineJoin  = 'round';
          const segsPerBucket = Math.max(1, Math.ceil(validCount / TRAIL_BUCKETS));

          for (let b = 0; b < TRAIL_BUCKETS; b++) {
            const segStart = validStart + b * segsPerBucket;
            const segEnd   = Math.min(segStart + segsPerBucket, count - 1);
            if (segStart >= segEnd) break;

            // t = 0 (oldest) → 1 (newest); linear alpha so older segments stay visible
            const t     = (b + 0.5) / TRAIL_BUCKETS;
            const alpha = (t * 0.92).toFixed(3);

            tCtx.beginPath();
            for (let i = segStart; i <= segEnd; i++) {
              const fi = ((H2 - count + i + MAX_TRAIL_PTS) % MAX_TRAIL_PTS) * 3;
              const px = trailBufRef.current[fi];
              const py = trailBufRef.current[fi + 1];
              if (i === segStart) tCtx.moveTo(px, py);
              else tCtx.lineTo(px, py);
            }
            tCtx.strokeStyle = `rgba(74,222,128,${alpha})`;
            tCtx.stroke();
          }
        }
      }
    }

    // ── Paint to main canvas ─────────────────────────────────────────────────
    ctx.fillStyle = BG_COLOR;
    ctx.fillRect(0, 0, W, H);
    ctx.drawImage(tc, 0, 0);

    if (p.showPendulum) {
      const { th1, th2 } = refRef.current;
      drawPendulumGeometry(ctx, W, H, th1, th2);
    }

    // ── Phase portrait panel ─────────────────────────────────────────────────
    if (p.showPhase) {
      const pc = phasePanelRef.current;
      if (pc && pc.clientWidth > 0) {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const pw = (pc.clientWidth  * dpr) | 0;
        const ph = (pc.clientHeight * dpr) | 0;
        if (pc.width !== pw || pc.height !== ph) { pc.width = pw; pc.height = ph; }
        drawPhasePortrait(pc);
      }
    }

    rafRef.current = requestAnimationFrame(draw);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Lifecycle ────────────────────────────────────────────────────────────────

  useEffect(() => {
    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [draw]);

  // DPR-aware canvas resize
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width  = (canvas.getBoundingClientRect().width  * dpr) | 0;
      canvas.height = (canvas.getBoundingClientRect().height * dpr) | 0;
    });
    ro.observe(canvas);
    return () => ro.disconnect();
  }, []);

  // ── Render ────────────────────────────────────────────────────────────────────

  const theta1Deg = Math.round(theta1 * DEG);
  const theta2Deg = Math.round(theta2 * DEG);

  return (
    <div className={styles.container}>
      <canvas ref={canvasRef} className={styles.canvas} />

      {/* ── Floating sidebar ───────────────────────────────────────────────── */}
      <div ref={sidebarRef} className={styles.sidebar}>

        <ControlPanel title="Presets">
          <ControlGroup>
            <div className={styles.snapGrid}>
              {PRESETS.map(p => (
                <button
                  key={p.label}
                  className={styles.snapBtn}
                  type="button"
                  title={p.desc}
                  onClick={() => goToPreset(p)}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </ControlGroup>
        </ControlPanel>

        <ControlPanel title="Physics">
          <ControlGroup>
            <Slider
              label="Gravity g"
              value={g} onChange={setG}
              min={1} max={20} step={0.1}
              format={v => v.toFixed(1)} unit="m/s²"
            />
            <Slider
              label="θ₁"
              value={theta1} onChange={setTheta1}
              min={-Math.PI} max={Math.PI} step={0.01}
              format={v => `${Math.round(v * DEG)}°`}
            />
            <Slider
              label="θ₂"
              value={theta2} onChange={setTheta2}
              min={-Math.PI} max={Math.PI} step={0.01}
              format={v => `${Math.round(v * DEG)}°`}
            />
            <Slider
              label="ω₁"
              value={omega1} onChange={setOmega1}
              min={-12} max={12} step={0.1}
              format={v => v.toFixed(1)} unit="rad/s"
            />
            <Slider
              label="ω₂"
              value={omega2} onChange={setOmega2}
              min={-12} max={12} step={0.1}
              format={v => v.toFixed(1)} unit="rad/s"
            />
          </ControlGroup>
        </ControlPanel>

        <ControlPanel title="Animation">
          <ControlGroup>
            <Toggle label="Running" value={running} onChange={setRunning} />
            <Slider
              label="Speed"
              value={speed} onChange={setSpeed}
              min={1} max={8} step={1} unit="×"
            />
            <Slider
              label="dt"
              value={dt} onChange={setDt}
              min={0.0005} max={0.005} step={0.0005}
              format={v => v.toFixed(4)}
            />
            <Slider
              label="Trail length"
              value={trailSecs} onChange={setTrailSecs}
              min={1} max={300} step={1}
              format={v => `${Math.round(v)}s`}
            />
            <Slider
              label="Point size"
              value={pointSize} onChange={setPointSize}
              min={1} max={5} step={0.5}
              format={v => v.toFixed(1)}
            />
          </ControlGroup>
        </ControlPanel>

        <ControlPanel title="Display">
          <ControlGroup>
            <Toggle label="Pendulum" value={showPendulum} onChange={setShowPendulum}
              description="Show reference pendulum geometry" />
            <Toggle label="Phase portrait" value={showPhase} onChange={setShowPhase}
              description="Show θ₁ vs ω₁ phase space" />
            <Toggle label="Chaos ensemble" value={showEnsemble} onChange={setShowEnsemble}
              description={gpuAvailable
                ? 'Show 16 384 pendulums with slightly different starting angles to visualise how chaos causes them to diverge'
                : 'WebGL2 required – not available in this browser'} />
            {showEnsemble && gpuAvailable && (<>
              <SelectControl
                label="Color"
                value={colorMode}
                onChange={v => setColorMode(v as ColorMode)}
                options={[
                  { value: 'heat',    label: 'Heat (blue→red)' },
                  { value: 'rainbow', label: 'Rainbow' },
                  { value: 'green',   label: 'Green' },
                ]}
              />
              <Slider
                label="Spread"
                value={spread} onChange={setSpread}
                min={0.001} max={1.0} step={0.001}
                format={v => v.toFixed(3)} unit="rad"
              />
              <Slider
                label="Point size"
                value={pointSize} onChange={setPointSize}
                min={1} max={5} step={0.5}
                format={v => v.toFixed(1)}
              />
            </>)}
          </ControlGroup>
        </ControlPanel>

        <button className={styles.resetBtn} type="button" onClick={resetSimulation}>
          Reset Simulation
        </button>
      </div>

      {/* ── Phase portrait panel ─────────────────────────────────────────────── */}
      {showPhase && (
        <div className={styles.panelStack}>
          <div className={styles.analysisPanel}>
            <div className={styles.panelHeader}>
              <span className={styles.panelTitle}>Phase Portrait</span>
              <div className={styles.infoBtnWrapper}>
                <button className={styles.infoBtn} type="button">
                  <Info size={13} />
                </button>
                <div className={styles.infoTooltip}>
                  θ₁ (horizontal, −π to π) vs ω₁ (vertical, angular velocity of rod 1).
                  The reference pendulum traces its trajectory through phase space —
                  chaotic motion fills the accessible region irregularly,
                  while periodic motion forms closed loops.
                </div>
              </div>
            </div>
            <div className={styles.plotWrapper}>
              <canvas ref={phasePanelRef} className={styles.plotCanvas} />
              <span className={`${styles.axisLabel} ${styles.axisLabelH}`}>θ₁</span>
              <span className={`${styles.axisLabel} ${styles.axisLabelV}`}>ω₁</span>
            </div>
          </div>
        </div>
      )}

      {/* ── HUD ──────────────────────────────────────────────────────────────── */}
      <div className={styles.hud}>
        <div className={styles.hudLeft}>
          <span className={styles.hudTitle}>Double Pendulum</span>
          <span className={styles.hudSub}>
            {theta1Deg}° / {theta2Deg}°
          </span>
        </div>
        <div className={styles.hudRight}>
          {showEnsemble && gpuAvailable && (
            <span className={styles.hudHint}>
              16 384 pendulums
            </span>
          )}
          <span className={styles.hudHint}>
            {running ? 'running' : 'paused'}
          </span>
        </div>
      </div>
    </div>
  );
}
