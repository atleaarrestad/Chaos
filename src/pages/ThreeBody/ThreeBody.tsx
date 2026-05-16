import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Slider, Toggle,
  ControlPanel, ControlGroup,
} from '@/components/Controls';
import { InfoDialog } from '@/components/InfoDialog';
import styles from './ThreeBody.module.css';

// ─── Constants ────────────────────────────────────────────────────────────────

const BG            = '#050510';
const BODY_COLORS   = ['#f43f5e', '#38bdf8', '#fbbf24'] as const;
const BODY_RGB      = [[244, 63, 94], [56, 189, 248], [251, 191, 36]] as const;
const MAX_TRAIL     = 10_000;   // ring-buffer size (x,y pairs) per body
const EPS           = 0.008;   // gravitational softening radius (prevents div-by-zero)
const BASE_STEPS    = 30;      // physics sub-steps per animation frame
const TRAIL_BUCKETS = 30;      // alpha gradient segments for trail rendering

// ─── Types ────────────────────────────────────────────────────────────────────

interface Body { x: number; y: number; vx: number; vy: number; mass: number; }
type Bodies = [Body, Body, Body];

// ─── Physics ──────────────────────────────────────────────────────────────────

function computeAccels(b: Bodies, G: number): [[number, number], [number, number], [number, number]] {
  const a: [[number, number], [number, number], [number, number]] = [[0, 0], [0, 0], [0, 0]];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      if (i === j) continue;
      const dx = b[j].x - b[i].x;
      const dy = b[j].y - b[i].y;
      const r2 = dx * dx + dy * dy + EPS * EPS;
      const r  = Math.sqrt(r2);
      const f  = G * b[j].mass / (r2 * r);
      a[i][0] += f * dx;
      a[i][1] += f * dy;
    }
  }
  return a;
}

function rk4Step(bodies: Bodies, dt: number, G: number): Bodies {
  const h2 = dt / 2, h6 = dt / 6;

  const a1 = computeAccels(bodies, G);

  const s2 = bodies.map((b, i) => ({
    ...b,
    x: b.x + h2 * b.vx, y: b.y + h2 * b.vy,
    vx: b.vx + h2 * a1[i][0], vy: b.vy + h2 * a1[i][1],
  })) as Bodies;
  const a2 = computeAccels(s2, G);

  const s3 = bodies.map((b, i) => ({
    ...b,
    x: b.x + h2 * s2[i].vx, y: b.y + h2 * s2[i].vy,
    vx: b.vx + h2 * a2[i][0], vy: b.vy + h2 * a2[i][1],
  })) as Bodies;
  const a3 = computeAccels(s3, G);

  const s4 = bodies.map((b, i) => ({
    ...b,
    x: b.x + dt * s3[i].vx, y: b.y + dt * s3[i].vy,
    vx: b.vx + dt * a3[i][0], vy: b.vy + dt * a3[i][1],
  })) as Bodies;
  const a4 = computeAccels(s4, G);

  return bodies.map((b, i) => ({
    ...b,
    x:  b.x  + h6 * (b.vx     + 2 * s2[i].vx  + 2 * s3[i].vx  + s4[i].vx),
    y:  b.y  + h6 * (b.vy     + 2 * s2[i].vy   + 2 * s3[i].vy   + s4[i].vy),
    vx: b.vx + h6 * (a1[i][0] + 2 * a2[i][0] + 2 * a3[i][0] + a4[i][0]),
    vy: b.vy + h6 * (a1[i][1] + 2 * a2[i][1] + 2 * a3[i][1] + a4[i][1]),
  })) as Bodies;
}

function totalEnergy(bodies: Bodies, G: number): number {
  let ke = 0, pe = 0;
  for (const b of bodies) ke += 0.5 * b.mass * (b.vx * b.vx + b.vy * b.vy);
  for (let i = 0; i < 3; i++) {
    for (let j = i + 1; j < 3; j++) {
      const dx = bodies[j].x - bodies[i].x;
      const dy = bodies[j].y - bodies[i].y;
      const r  = Math.sqrt(dx * dx + dy * dy + EPS * EPS);
      pe -= G * bodies[i].mass * bodies[j].mass / r;
    }
  }
  return ke + pe;
}

// ─── Presets ──────────────────────────────────────────────────────────────────

// Lagrange equilateral triangle: ω² = G·m / (R³·√3) for G=m=R=1 → ω = 3^(−¼)
const _W  = Math.pow(3, -0.25); // angular velocity ≈ 0.7598
const _S3 = Math.sqrt(3) / 2;   // sin 60°

interface Preset {
  label: string;
  desc: string;
  G: number;
  dt: number;
  bodies: Bodies;
}

const PRESETS: Preset[] = [
  {
    label: 'Figure-8',
    desc: 'Choreographic orbit — all three bodies chase each other through the same figure-eight curve (Chenciner & Montgomery, 2000)',
    G: 1, dt: 0.001,
    bodies: [
      { x: -0.97000436, y:  0.24308753, vx:  0.46620368, vy:  0.43236573, mass: 1 },
      { x:  0,          y:  0,          vx: -0.93240737, vy: -0.86473146, mass: 1 },
      { x:  0.97000436, y: -0.24308753, vx:  0.46620368, vy:  0.43236573, mass: 1 },
    ],
  },
  {
    label: 'Lagrange',
    desc: 'Three equal masses at vertices of a rotating equilateral triangle — a perfectly stable configuration (Lagrange, 1772)',
    G: 1, dt: 0.002,
    bodies: [
      { x:  1,    y:  0,    vx:  0,       vy:  _W,    mass: 1 },
      { x: -0.5,  y:  _S3,  vx: -_W * _S3, vy: -_W / 2, mass: 1 },
      { x: -0.5,  y: -_S3,  vx:  _W * _S3, vy: -_W / 2, mass: 1 },
    ],
  },
  {
    label: 'Pythagorean',
    desc: 'Masses 3–4–5 at rest in a right triangle — quickly becomes chaotic with near-collisions and ejections (Szebehely & Peters, 1967)',
    G: 1, dt: 0.0005,
    bodies: [
      { x:  1, y:  3, vx: 0, vy: 0, mass: 3 },
      { x: -2, y: -1, vx: 0, vy: 0, mass: 4 },
      { x:  1, y: -1, vx: 0, vy: 0, mass: 5 },
    ],
  },
  {
    label: 'Binary',
    desc: 'A tight binary pair with a distant intruder — the intruder falls in and disrupts the orbital dance',
    G: 1, dt: 0.002,
    bodies: [
      { x: -0.5, y: 0, vx: 0, vy: -0.7071, mass: 1 },
      { x:  0.5, y: 0, vx: 0, vy:  0.7071, mass: 1 },
      { x:  5,   y: 0, vx: 0, vy:  0,      mass: 1 },
    ],
  },
  {
    label: 'Chaos',
    desc: 'Three unequal masses in a configuration that quickly produces an unpredictable gravitational dance',
    G: 1, dt: 0.001,
    bodies: [
      { x: -1,  y:  0.5,  vx:  0.30,  vy: -0.20,  mass: 1.5 },
      { x:  1,  y:  0.3,  vx: -0.10,  vy:  0.20,  mass: 2.0 },
      { x:  0,  y: -1.2,  vx: -0.25,  vy: -0.10,  mass: 1.0 },
    ],
  },
];

// ─── Mutable params read by the RAF loop ──────────────────────────────────────

interface LiveParams {
  running: boolean;
  speed: number;
  dt: number;
  trailLen: number;
  showTrails: boolean;
  showVectors: boolean;
  G: number;
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function ThreeBody() {
  const [running,      setRunning]      = useState(true);
  const [speed,        setSpeed]        = useState(1);
  const [dt,           setDt]           = useState(PRESETS[0].dt);
  const [trailLen,     setTrailLen]     = useState(3000);
  const [showTrails,   setShowTrails]   = useState(true);
  const [showVectors,  setShowVectors]  = useState(false);
  const [G,            setG]            = useState(PRESETS[0].G);
  const [activePreset, setActivePreset] = useState<number | null>(0);
  const [showInfo,     setShowInfo]     = useState(false);

  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const rafRef      = useRef(0);
  const energyRef   = useRef<HTMLSpanElement>(null);
  const timeHudRef  = useRef<HTMLSpanElement>(null);

  const bodiesRef   = useRef<Bodies>(JSON.parse(JSON.stringify(PRESETS[0].bodies)));
  const trailsRef   = useRef<Float32Array[]>([
    new Float32Array(MAX_TRAIL * 2),
    new Float32Array(MAX_TRAIL * 2),
    new Float32Array(MAX_TRAIL * 2),
  ]);
  const trailHeads  = useRef([0, 0, 0]);
  const trailCounts = useRef([0, 0, 0]);
  const viewScale     = useRef(200.0);
  const smoothRefDist = useRef(1.0);   // slow-EMA of median body distance — prevents bouncy zoom
  const simTime     = useRef(0);
  const initEnergy  = useRef<number | null>(null);

  // Stable ref for RAF loop access to latest live params
  const pRef = useRef<LiveParams>({ running, speed, dt, trailLen, showTrails, showVectors, G });
  useEffect(() => {
    pRef.current = { running, speed, dt, trailLen, showTrails, showVectors, G };
  }, [running, speed, dt, trailLen, showTrails, showVectors, G]);

  // Stable ref so resetSimulation callback stays stable
  const activePresetRef = useRef<number | null>(0);
  useEffect(() => { activePresetRef.current = activePreset; }, [activePreset]);

  // ── Trail / sim helpers ────────────────────────────────────────────────────

  // Median body distance from CoM — used to seed the camera on preset load
  const medianDistFromCoM = (bodies: Body[]) => {
    let totalMass = 0, cx = 0, cy = 0;
    for (const b of bodies) { totalMass += b.mass; cx += b.x * b.mass; cy += b.y * b.mass; }
    cx /= totalMass; cy /= totalMass;
    const dists = bodies.map(b => Math.sqrt((b.x - cx) ** 2 + (b.y - cy) ** 2));
    dists.sort((a, b) => a - b);
    return Math.max(dists[1], 0.4);
  };

  const clearTrails = (initRefDist = 1.0) => {
    for (let i = 0; i < 3; i++) {
      trailsRef.current[i].fill(0);
      trailHeads.current[i]  = 0;
      trailCounts.current[i] = 0;
    }
    viewScale.current     = 200;
    smoothRefDist.current = initRefDist;
    simTime.current    = 0;
    initEnergy.current = null;
  };

  // ── Preset navigation ──────────────────────────────────────────────────────

  const goToPreset = useCallback((idx: number) => {
    const preset = PRESETS[idx];
    setActivePreset(idx);
    setDt(preset.dt);
    setG(preset.G);
    pRef.current = { ...pRef.current, dt: preset.dt, G: preset.G };
    bodiesRef.current = JSON.parse(JSON.stringify(preset.bodies));
    clearTrails(medianDistFromCoM(bodiesRef.current));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Reset ──────────────────────────────────────────────────────────────────

  const resetSimulation = useCallback(() => {
    const idx = activePresetRef.current;
    if (idx !== null) {
      const preset = PRESETS[idx];
      bodiesRef.current = JSON.parse(JSON.stringify(preset.bodies));
      pRef.current = { ...pRef.current, dt: preset.dt, G: preset.G };
    }
    clearTrails(medianDistFromCoM(bodiesRef.current));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Main RAF draw loop ─────────────────────────────────────────────────────

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) { rafRef.current = requestAnimationFrame(draw); return; }

    const ctx = canvas.getContext('2d')!;
    const W = canvas.width, H = canvas.height;
    const p = pRef.current;

    // ── Advance physics ────────────────────────────────────────────────────────
    if (p.running) {
      const steps = p.speed * BASE_STEPS;
      for (let s = 0; s < steps; s++) {
        bodiesRef.current = rk4Step(bodiesRef.current, p.dt, p.G);
        simTime.current  += p.dt;
      }

      // Record one trail point per body per frame
      for (let i = 0; i < 3; i++) {
        const b    = bodiesRef.current[i];
        const head = trailHeads.current[i];
        trailsRef.current[i][head * 2]     = b.x;
        trailsRef.current[i][head * 2 + 1] = b.y;
        trailHeads.current[i]  = (head + 1) % MAX_TRAIL;
        trailCounts.current[i] = Math.min(trailCounts.current[i] + 1, MAX_TRAIL);
      }

      // Energy HUD (imperative update — no re-render needed)
      const E = totalEnergy(bodiesRef.current, p.G);
      if (initEnergy.current === null) initEnergy.current = E;
      if (energyRef.current) {
        const E0 = initEnergy.current;
        const driftPct = E0 !== 0
          ? ((E - E0) / Math.abs(E0) * 100).toFixed(3)
          : '—';
        energyRef.current.textContent = `E = ${E.toFixed(3)}  drift ${driftPct}%`;
      }
      if (timeHudRef.current) {
        timeHudRef.current.textContent = `t = ${simTime.current.toFixed(1)}`;
      }
    }

    const bodies = bodiesRef.current;

    // ── Camera: CoM + adaptive scale ──────────────────────────────────────────
    let totalMass = 0, cx = 0, cy = 0;
    for (const b of bodies) { totalMass += b.mass; cx += b.x * b.mass; cy += b.y * b.mass; }
    cx /= totalMass; cy /= totalMass;

    // Sort body distances; use median distance so an escaping body doesn't collapse the view.
    // Use a running max with very slow decay rather than EMA: the camera zooms out immediately
    // when bodies expand but shrinks back only over many seconds. This makes periodic orbits
    // (e.g. figure-8) produce a completely stable view — the max is hit once and held.
    const dists = bodies.map(b => Math.sqrt((b.x - cx) ** 2 + (b.y - cy) ** 2));
    dists.sort((a, b) => a - b);
    const rawRefDist = Math.max(dists[1], 0.4);
    smoothRefDist.current = Math.max(rawRefDist, smoothRefDist.current * 0.9997);

    const targetScale = Math.min(W, H) * 0.42 / smoothRefDist.current;
    const clampedTarget = Math.max(10, Math.min(1200, targetScale));
    viewScale.current += (clampedTarget - viewScale.current) * 0.06;
    const scale = viewScale.current;

    const toX = (x: number) => W / 2 + (x - cx) * scale;
    const toY = (y: number) => H / 2 - (y - cy) * scale;

    // ── Clear ─────────────────────────────────────────────────────────────────
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, W, H);

    // ── Trails ────────────────────────────────────────────────────────────────
    if (p.showTrails) {
      ctx.lineCap  = 'round';
      ctx.lineJoin = 'round';

      for (let bi = 0; bi < 3; bi++) {
        const trail = trailsRef.current[bi];
        const head  = trailHeads.current[bi];
        const total = trailCounts.current[bi];
        const count = Math.min(total, p.trailLen);
        if (count < 2) continue;

        const [r, g, bc] = BODY_RGB[bi];
        const bucketSize  = Math.max(1, Math.ceil(count / TRAIL_BUCKETS));

        for (let bucket = 0; bucket < TRAIL_BUCKETS; bucket++) {
          const segStart = bucket * bucketSize;
          const segEnd   = Math.min(segStart + bucketSize, count - 1);
          if (segEnd <= segStart) break;

          const t = (bucket + 0.5) / TRAIL_BUCKETS;
          ctx.strokeStyle = `rgba(${r},${g},${bc},${(t * 0.88).toFixed(3)})`;
          ctx.lineWidth   = 1.0 + t * 0.8;

          ctx.beginPath();
          for (let i = segStart; i <= segEnd; i++) {
            const si = ((head - count + i + MAX_TRAIL) % MAX_TRAIL) * 2;
            const px = toX(trail[si]);
            const py = toY(trail[si + 1]);
            if (i === segStart) ctx.moveTo(px, py);
            else                ctx.lineTo(px, py);
          }
          ctx.stroke();
        }
      }
    }

    // ── Velocity vectors ───────────────────────────────────────────────────────
    if (p.showVectors) {
      ctx.lineWidth   = 1.5;
      ctx.lineCap     = 'round';
      ctx.globalAlpha = 0.8;
      for (let bi = 0; bi < 3; bi++) {
        const b  = bodies[bi];
        const px = toX(b.x), py = toY(b.y);
        const ex = toX(b.x + b.vx * 0.3);
        const ey = toY(b.y + b.vy * 0.3);
        ctx.strokeStyle = BODY_COLORS[bi];
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(ex, ey);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    // ── Bodies ────────────────────────────────────────────────────────────────
    for (let bi = 0; bi < 3; bi++) {
      const b   = bodies[bi];
      const px  = toX(b.x), py = toY(b.y);
      const r   = Math.max(5, 3 + b.mass * 1.8);
      const [cr, cg, cb] = BODY_RGB[bi];

      // Soft radial glow
      const grd = ctx.createRadialGradient(px, py, 0, px, py, r * 4);
      grd.addColorStop(0,   `rgba(${cr},${cg},${cb},0.30)`);
      grd.addColorStop(0.4, `rgba(${cr},${cg},${cb},0.08)`);
      grd.addColorStop(1,   `rgba(${cr},${cg},${cb},0)`);
      ctx.beginPath();
      ctx.arc(px, py, r * 4, 0, Math.PI * 2);
      ctx.fillStyle = grd;
      ctx.fill();

      // Core body with shadow glow
      ctx.shadowColor = BODY_COLORS[bi];
      ctx.shadowBlur  = 18;
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fillStyle = BODY_COLORS[bi];
      ctx.fill();
      ctx.shadowBlur = 0;

      // Specular highlight
      ctx.beginPath();
      ctx.arc(px - r * 0.3, py - r * 0.32, r * 0.25, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,0.55)';
      ctx.fill();
    }

    rafRef.current = requestAnimationFrame(draw);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

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

  return (
    <div className={styles.container}>
      <canvas ref={canvasRef} className={styles.canvas} />

      {/* ── Floating sidebar ──────────────────────────────────────────────────── */}
      <div className={styles.sidebar}>
        <div className={styles.sidebarPanels}>

          <ControlPanel title="Presets">
            <ControlGroup>
              <div className={styles.snapGrid}>
                {PRESETS.map((preset, idx) => (
                  <button
                    key={preset.label}
                    className={`${styles.snapBtn}${activePreset === idx ? ` ${styles.snapBtnActive}` : ''}`}
                    type="button"
                    title={preset.desc}
                    onClick={() => goToPreset(idx)}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
              <div className={styles.legend}>
                {BODY_COLORS.map((col, i) => (
                  <span key={i} className={styles.legendItem}>
                    <span className={styles.legendDot} style={{ background: col }} />
                    Body {i + 1}
                  </span>
                ))}
              </div>
            </ControlGroup>
          </ControlPanel>

          <ControlPanel title="Physics">
            <ControlGroup>
              <Slider
                label="G"
                value={G}
                onChange={v => { setG(v); setActivePreset(null); }}
                min={0.1} max={5} step={0.1}
                format={v => v.toFixed(1)}
              />
            </ControlGroup>
          </ControlPanel>

          <ControlPanel title="Animation">
            <ControlGroup>
              <Toggle label="Running" value={running} onChange={setRunning} />
              <Slider
                label="Speed"
                value={speed} onChange={setSpeed}
                min={0.25} max={8} step={0.25}
                format={v => `${v}×`}
              />
              <Slider
                label="dt"
                value={dt}
                onChange={v => { setDt(v); setActivePreset(null); }}
                min={0.0001} max={0.005} step={0.0001}
                format={v => v.toFixed(4)}
              />
              <Slider
                label="Trail length"
                value={trailLen} onChange={setTrailLen}
                min={200} max={MAX_TRAIL} step={200}
                format={v => v.toLocaleString()}
              />
            </ControlGroup>
          </ControlPanel>

          <ControlPanel title="Display">
            <ControlGroup>
              <Toggle
                label="Trails"
                value={showTrails} onChange={setShowTrails}
                description="Show orbital trail for each body"
              />
              <Toggle
                label="Velocity vectors"
                value={showVectors} onChange={setShowVectors}
                description="Show velocity arrow on each body"
              />
            </ControlGroup>
          </ControlPanel>

        </div>

        <div className={styles.sidebarActions}>
          <button className={styles.resetBtn} type="button" onClick={resetSimulation}>
            Reset Simulation
          </button>
        </div>
      </div>

      {/* ── HUD ───────────────────────────────────────────────────────────────── */}
      <div className={styles.hud}>
        <div className={styles.hudLeft}>
          <span className={styles.hudTitle}>Three Body Problem</span>
          <span className={styles.hudSub} ref={timeHudRef}>t = 0.0</span>
        </div>
        <div className={styles.hudRight}>
          <span className={styles.hudHint} ref={energyRef} />
          <span className={styles.hudHint}>{running ? 'running' : 'paused'}</span>
          <button
            className={styles.infoBtn}
            onClick={() => setShowInfo(true)}
            title="About the three body problem"
          >
            ⓘ
          </button>
        </div>
      </div>

      {showInfo && (
        <InfoDialog title="Three Body Problem" onClose={() => setShowInfo(false)}>
          <p>
            Three massive bodies attracting each other through gravity. While the
            two-body problem has an exact analytical solution (Kepler's laws), adding
            a third body makes it generically unsolvable in closed form — and for most
            initial conditions, chaotic.
          </p>
          <h3>Presets</h3>
          <ul>
            <li>
              <strong>Figure-8:</strong> A remarkable periodic solution proved in 2000
              by Chenciner &amp; Montgomery. All three equal masses trace the same
              figure-eight curve in sequence, never colliding.
            </li>
            <li>
              <strong>Lagrange:</strong> A stable configuration where three equal masses
              sit at the vertices of a rotating equilateral triangle, discovered by
              Joseph-Louis Lagrange in 1772.
            </li>
            <li>
              <strong>Pythagorean:</strong> Masses 3–4–5 at rest in a right triangle.
              Quickly becomes chaotic, typically ending with one body ejected.
            </li>
            <li>
              <strong>Binary:</strong> A tight binary pair with a distant intruder
              falling in, perturbing and eventually disrupting the orbit.
            </li>
            <li>
              <strong>Chaos:</strong> Three unequal masses in a configuration that
              leads to an unpredictable gravitational dance.
            </li>
          </ul>
          <h3>Energy drift</h3>
          <p>
            The HUD shows the percentage change in total mechanical energy since
            the simulation started. A smaller drift means better numerical accuracy.
            Use a smaller dt for more precision at the cost of speed.
          </p>
          <h3>Controls</h3>
          <ul>
            <li><strong>Speed:</strong> multiplies the number of physics steps per frame</li>
            <li><strong>dt:</strong> time step per physics sub-step</li>
            <li><strong>Trail length:</strong> how many recorded frames of history are drawn</li>
          </ul>
        </InfoDialog>
      )}
    </div>
  );
}
