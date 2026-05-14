import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Slider, Toggle, SelectControl,
  ControlPanel, ControlGroup,
} from '@/components/Controls';
import styles from './Attractors.module.css';

// ─── Types ───────────────────────────────────────────────────────────────────

type Vec3    = readonly [number, number, number];
type MutVec3 = [number, number, number];
type DerivFn = (x: number, y: number, z: number, p: number[]) => Vec3;
type ColorScheme = 'velocity' | 'heat' | 'plasma' | 'neon';

const N_PARTICLES = 6;
const MAX_BUF     = 8_000;

// ─── Ring buffer ─────────────────────────────────────────────────────────────

interface RingBuf {
  xyz:   Float32Array;
  head:  number;
  total: number;
}

const makeRing = (): RingBuf => ({
  xyz: new Float32Array(MAX_BUF * 3),
  head: 0, total: 0,
});

function ringPush(rb: RingBuf, x: number, y: number, z: number): void {
  const i = rb.head * 3;
  rb.xyz[i] = x; rb.xyz[i + 1] = y; rb.xyz[i + 2] = z;
  rb.head = (rb.head + 1) % MAX_BUF;
  rb.total++;
}

function ringGet(rb: RingBuf, i: number, visible: number): Vec3 {
  const slot = ((rb.head - visible + i) % MAX_BUF + MAX_BUF) % MAX_BUF;
  const b = slot * 3;
  return [rb.xyz[b], rb.xyz[b + 1], rb.xyz[b + 2]];
}

// ─── RK4 ─────────────────────────────────────────────────────────────────────

function rk4(
  x: number, y: number, z: number,
  fn: DerivFn, p: number[], dt: number,
): MutVec3 {
  const [k1x, k1y, k1z] = fn(x, y, z, p);
  const h = dt * 0.5;
  const [k2x, k2y, k2z] = fn(x + k1x * h, y + k1y * h, z + k1z * h, p);
  const [k3x, k3y, k3z] = fn(x + k2x * h, y + k2y * h, z + k2z * h, p);
  const [k4x, k4y, k4z] = fn(x + k3x * dt, y + k3y * dt, z + k3z * dt, p);
  const s = dt / 6;
  return [
    x + (k1x + 2 * k2x + 2 * k3x + k4x) * s,
    y + (k1y + 2 * k2y + 2 * k3y + k4y) * s,
    z + (k1z + 2 * k2z + 2 * k3z + k4z) * s,
  ];
}

// ─── Color schemes ────────────────────────────────────────────────────────────

const cl = (v: number) => Math.max(0, Math.min(255, v | 0));

const COLOR_SCHEMES: Record<ColorScheme, (t: number) => [number, number, number]> = {
  velocity: t => {
    if (t < 0.20) { const s = t / 0.20;          return [cl(10),             cl(20 + 180 * s), cl(160 + 60 * s)];  }
    if (t < 0.40) { const s = (t - 0.20) / 0.20; return [cl(10 + 20 * s),   cl(200 - 20 * s), cl(220 - 190 * s)]; }
    if (t < 0.60) { const s = (t - 0.40) / 0.20; return [cl(30 + 180 * s),  cl(180 + 30 * s), cl(30)];            }
    if (t < 0.80) { const s = (t - 0.60) / 0.20; return [cl(210 + 45 * s),  cl(210 - 70 * s), cl(20 - 10 * s)];  }
    const s = (t - 0.80) / 0.20;                  return [cl(255),           cl(140 - 100 * s), cl(10)];
  },
  heat: t => {
    const v = t * 3;
    return [cl(v * 255), cl(Math.max(0, v - 1) * 255), cl(Math.max(0, v - 2) * 255)];
  },
  plasma: t => [cl(13 + 220 * t), cl(8 + 240 * t * t), cl(135 - 70 * t)],
  neon:   t => [cl(80 * t * t),   cl(30 + 210 * t),    cl(60 + 160 * t * (1 - 0.4 * t))],
};

// ─── Attractor catalogue ─────────────────────────────────────────────────────

interface ParamDef {
  key: string; label: string;
  min: number; max: number; step: number; default: number;
}

interface AttractorType {
  id:          string;
  name:        string;
  equations:   string[];
  description: string;
  fn:          DerivFn;
  dt:          number;
  warmupSteps: number;
  initPos:     Vec3;
  center:      Vec3;
  /** World units spanning the full canvas height (controls zoom-to-fit). */
  viewUnits:   number;
  /** Typical peak speed in world-units/s, used to normalise velocity colour. */
  maxSpeed:    number;
  defaultView: { rx: number; ry: number };
  params:      ParamDef[];
  accentColor: string;
  accentRgb:   string;
}

const ATTRACTOR_CATALOGUE: AttractorType[] = [
  {
    id: 'rossler', name: 'Rössler',
    equations: ['dx/dt = −y − z', 'dy/dt = x + ay', 'dz/dt = b + z(x − c)'],
    description: 'A single-scroll attractor simpler than Lorenz. Parameter c drives period-doubling bifurcations leading to chaos.',
    fn: (x, y, z, [a, b, c]) => [-(y + z), x + a * y, b + z * (x - c)],
    dt: 0.01, warmupSteps: 5000, initPos: [0.1, 0, 0],
    center: [0, 0, 12], viewUnits: 36, maxSpeed: 30,
    defaultView: { rx: 0.3, ry: 0.5 },
    params: [
      { key: 'a', label: 'a', min: 0.1,  max: 0.5,  step: 0.01,   default: 0.2      },
      { key: 'b', label: 'b', min: 0.1,  max: 0.5,  step: 0.01,   default: 0.2      },
      { key: 'c', label: 'c', min: 4.0,  max: 12.0, step: 0.01,   default: 5.7      },
    ],
    accentColor: '#f59e0b', accentRgb: '245,158,11',
  },
  {
    id: 'halvorsen', name: 'Halvorsen',
    equations: ['dx/dt = −ax − 4y − 4z − y²', 'dy/dt = −ay − 4z − 4x − z²', 'dz/dt = −az − 4x − 4y − x²'],
    description: 'Cyclically symmetric — all three variables are interchangeable under a 120° rotation. Parameter a modulates between periodic and chaotic regimes.',
    fn: (x, y, z, [a]) => [
      -a * x - 4 * y - 4 * z - y * y,
      -a * y - 4 * z - 4 * x - z * z,
      -a * z - 4 * x - 4 * y - x * x,
    ],
    dt: 0.005, warmupSteps: 10000, initPos: [1, 0, 0],
    center: [0, 0, 0], viewUnits: 22, maxSpeed: 80,
    defaultView: { rx: 0.5, ry: 0.8 },
    params: [
      { key: 'a', label: 'a', min: 0.8, max: 2.0, step: 0.01, default: 1.4 },
    ],
    accentColor: '#22d3ee', accentRgb: '34,211,238',
  },
  {
    id: 'thomas', name: 'Thomas',
    equations: ['dx/dt = sin(y) − bx', 'dy/dt = sin(z) − by', 'dz/dt = sin(x) − bz'],
    description: 'René Thomas\'s cyclically symmetric system. Near b ≈ 0.208 the orbit sits at the edge of chaos. Lower b produces multi-lobe wandering.',
    fn: (x, y, z, [b]) => [Math.sin(y) - b * x, Math.sin(z) - b * y, Math.sin(x) - b * z],
    dt: 0.05, warmupSteps: 2000, initPos: [0.1, 0, 0],
    center: [0, 0, 0], viewUnits: 11, maxSpeed: 2,
    defaultView: { rx: 0.4, ry: 0.6 },
    params: [
      { key: 'b', label: 'b', min: 0.1, max: 0.35, step: 0.001, default: 0.208186 },
    ],
    accentColor: '#4ade80', accentRgb: '74,222,128',
  },
  {
    id: 'aizawa', name: 'Aizawa',
    equations: ['dx/dt = (z−b)x − dy', 'dy/dt = dx + (z−b)y', 'dz/dt = c + az − z³/3 − r²(1+ez) + fz·x³'],
    description: 'Orbits spiral around a toroidal manifold, producing delicate layered scrolls. One of the most visually distinctive low-dimensional attractors.',
    fn: (x, y, z, [a, b, c, d, e, f]) => [
      (z - b) * x - d * y,
      d * x + (z - b) * y,
      c + a * z - (z * z * z) / 3 - (x * x + y * y) * (1 + e * z) + f * z * x * x * x,
    ],
    dt: 0.01, warmupSteps: 5000, initPos: [0.1, 0, 0.1],
    center: [0, 0, 0.2], viewUnits: 4, maxSpeed: 5,
    defaultView: { rx: 1.1, ry: 0.3 },
    params: [
      { key: 'a', label: 'a', min: 0.7, max: 1.0,  step: 0.01, default: 0.95 },
      { key: 'b', label: 'b', min: 0.5, max: 0.9,  step: 0.01, default: 0.7  },
      { key: 'c', label: 'c', min: 0.4, max: 0.8,  step: 0.01, default: 0.6  },
      { key: 'd', label: 'd', min: 2.5, max: 4.5,  step: 0.1,  default: 3.5  },
      { key: 'e', label: 'e', min: 0.1, max: 0.4,  step: 0.01, default: 0.25 },
      { key: 'f', label: 'f', min: 0.0, max: 0.2,  step: 0.01, default: 0.1  },
    ],
    accentColor: '#f472b6', accentRgb: '244,114,182',
  },
  {
    id: 'dadras', name: 'Dadras',
    equations: ['dx/dt = y − px + qyz', 'dy/dt = ry − xz + z', 'dz/dt = sxy − tz'],
    description: 'Discovered by Dadras & Momeni. Two lobes connected by narrow chaotic bridges. Highly sensitive to p, which controls the expansion rate.',
    fn: (x, y, z, [p, q, r, s, t]) => [
      y - p * x + q * y * z,
      r * y - x * z + z,
      s * x * y - t * z,
    ],
    dt: 0.01, warmupSteps: 5000, initPos: [0, 0.3, 0.3],
    center: [0, 0, 0], viewUnits: 20, maxSpeed: 25,
    defaultView: { rx: 0.4, ry: 0.3 },
    params: [
      { key: 'p', label: 'p', min: 1.0, max: 5.0,  step: 0.1, default: 3.0 },
      { key: 'q', label: 'q', min: 1.0, max: 4.0,  step: 0.1, default: 2.7 },
      { key: 'r', label: 'r', min: 0.5, max: 3.0,  step: 0.1, default: 1.7 },
      { key: 's', label: 's', min: 1.0, max: 4.0,  step: 0.1, default: 2.0 },
      { key: 't', label: 't', min: 5.0, max: 15.0, step: 0.1, default: 9.0 },
    ],
    accentColor: '#a855f7', accentRgb: '168,85,247',
  },
];

const DEFAULT_ID = 'rossler';

// ─── Particle helpers ─────────────────────────────────────────────────────────

interface Particle { pos: MutVec3; rb: RingBuf; }

function buildParticles(def: AttractorType, pv: number[]): Particle[] {
  let [px, py, pz] = def.initPos as MutVec3;
  for (let i = 0; i < def.warmupSteps; i++) {
    const next = rk4(px, py, pz, def.fn, pv, def.dt);
    if (!isFinite(next[0]) || !isFinite(next[1]) || !isFinite(next[2])) break;
    [px, py, pz] = next;
  }
  const spread = def.viewUnits * 0.004;
  return Array.from({ length: N_PARTICLES }, (_, i) => {
    const angle = (i / N_PARTICLES) * Math.PI * 2;
    return {
      pos: [
        px + spread * Math.cos(angle),
        py + spread * Math.sin(angle),
        pz + spread * 0.3 * Math.sin(angle * 3),
      ] as MutVec3,
      rb: makeRing(),
    };
  });
}

// ─── Loop-params snapshot (avoids stale closures in rAF) ─────────────────────

interface LoopParams {
  def:          AttractorType;
  pv:           number[];
  speed:        number;
  trailLength:  number;
  colorScheme:  ColorScheme;
  running:      boolean;
  autoRotate:   boolean;
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function Attractors() {
  const defDef = ATTRACTOR_CATALOGUE.find(a => a.id === DEFAULT_ID)!;

  const [attractorId,  setAttractorId]  = useState(DEFAULT_ID);
  const [paramValues,  setParamValues]  = useState<Record<string, number>>(
    () => Object.fromEntries(defDef.params.map(p => [p.key, p.default])),
  );
  const [speed,        setSpeed]        = useState(5);
  const [trailLength,  setTrailLength]  = useState(6_000);
  const [colorScheme,  setColorScheme]  = useState<ColorScheme>('velocity');
  const [running,      setRunning]      = useState(true);
  const [autoRotate,   setAutoRotate]   = useState(true);

  const canvasRef    = useRef<HTMLCanvasElement>(null);
  const sidebarRef   = useRef<HTMLDivElement>(null);
  const rafRef       = useRef(0);
  const rotRef       = useRef({ x: defDef.defaultView.rx, y: defDef.defaultView.ry });
  const dragRef      = useRef<{ x: number; y: number } | null>(null);
  const zoomRef      = useRef(1);
  const particlesRef = useRef<Particle[]>([]);
  const pRef         = useRef<LoopParams>({
    def: defDef,
    pv:  defDef.params.map(p => p.default),
    speed, trailLength, colorScheme, running, autoRotate,
  });

  // Keep pRef in sync with all reactive state
  useEffect(() => {
    const def = ATTRACTOR_CATALOGUE.find(a => a.id === attractorId)!;
    const pv  = def.params.map(p => paramValues[p.key] ?? p.default);
    pRef.current = { def, pv, speed, trailLength, colorScheme, running, autoRotate };
  }, [attractorId, paramValues, speed, trailLength, colorScheme, running, autoRotate]);

  // Re-initialise particles when the attractor type changes
  useEffect(() => {
    const def = ATTRACTOR_CATALOGUE.find(a => a.id === attractorId)!;
    const pv  = def.params.map(p => paramValues[p.key] ?? p.default);
    particlesRef.current = buildParticles(def, pv);
    rotRef.current = { x: def.defaultView.rx, y: def.defaultView.ry };
    zoomRef.current = 1;
    // paramValues intentionally excluded: we only re-init on type switch
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [attractorId]);

  const reset = useCallback(() => {
    const def = ATTRACTOR_CATALOGUE.find(a => a.id === pRef.current.def.id)!;
    const pv  = def.params.map(p => paramValues[p.key] ?? p.default);
    particlesRef.current = buildParticles(def, pv);
    rotRef.current = { x: def.defaultView.rx, y: def.defaultView.ry };
    zoomRef.current = 1;
  }, [paramValues]);

  const handleAttractorChange = useCallback((id: string) => {
    const def = ATTRACTOR_CATALOGUE.find(a => a.id === id)!;
    setParamValues(Object.fromEntries(def.params.map(p => [p.key, p.default])));
    setAttractorId(id);
  }, []);

  const handleParamChange = useCallback((key: string, value: number) => {
    setParamValues(prev => ({ ...prev, [key]: value }));
  }, []);

  // ─── Animation loop ──────────────────────────────────────────────────────────

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) { rafRef.current = requestAnimationFrame(draw); return; }

    const ctx = canvas.getContext('2d')!;
    const W = canvas.width, H = canvas.height;
    const { def, pv, speed, trailLength, colorScheme, running, autoRotate } = pRef.current;
    const particles = particlesRef.current;

    if (autoRotate && !dragRef.current) {
      rotRef.current.y += 0.004;
    }

    const { x: rotX, y: rotY } = rotRef.current;

    // Advance all particles
    if (running) {
      for (const p of particles) {
        let [px, py, pz] = p.pos;
        for (let i = 0; i < speed; i++) {
          const next = rk4(px, py, pz, def.fn, pv, def.dt);
          if (!isFinite(next[0]) || !isFinite(next[1]) || !isFinite(next[2])) {
            // Diverged — reset this particle to a nearby warmup point
            [px, py, pz] = [...def.initPos] as MutVec3;
            p.rb = makeRing();
            break;
          }
          ringPush(p.rb, next[0], next[1], next[2]);
          [px, py, pz] = next;
        }
        p.pos = [px, py, pz];
      }
    }

    // Clear
    ctx.fillStyle = '#080812';
    ctx.fillRect(0, 0, W, H);

    const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
    const cosX = Math.cos(rotX), sinX = Math.sin(rotX);

    const proj = (x: number, y: number, z: number): [number, number] => {
      const x1 = x * cosY + z * sinY;
      const z1 = -x * sinY + z * cosY;
      return [x1, y * cosX - z1 * sinX];
    };

    const scale = Math.min(W, H) / def.viewUnits * zoomRef.current;
    const [ox, oy] = proj(def.center[0], def.center[1], def.center[2]);
    const cx = W * 0.5 - ox * scale;
    const cy = H * 0.5 + oy * scale;

    // ─── Draw trails ─────────────────────────────────────────────────────────

    ctx.save();
    ctx.lineJoin = 'round';
    ctx.lineCap  = 'round';

    for (const p of particles) {
      const rb      = p.rb;
      const visible = Math.min(rb.total, trailLength, MAX_BUF);
      if (visible < 2) continue;

      const BATCH = Math.max(8, Math.ceil(visible / 200));

      for (let b = 0; b < visible - 1; b += BATCH) {
        const bEnd = Math.min(b + BATCH, visible - 1);
        const t    = (b + bEnd) * 0.5 / visible;

        let r: number, g: number, bl: number;
        if (colorScheme === 'velocity') {
          const midI        = Math.floor((b + bEnd) / 2);
          const [mx, my, mz] = ringGet(rb, midI, visible);
          const [nx, ny, nz] = ringGet(rb, Math.min(midI + 1, visible - 1), visible);
          const spd = Math.sqrt((nx - mx) ** 2 + (ny - my) ** 2 + (nz - mz) ** 2) / def.dt;
          const raw = Math.min(1, Math.log1p(spd) / Math.log1p(def.maxSpeed));
          const vt  = Math.pow(raw, 2.5);
          [r, g, bl] = COLOR_SCHEMES.velocity(vt);
        } else {
          [r, g, bl] = COLOR_SCHEMES[colorScheme](t);
        }

        ctx.globalAlpha = 0.1 + 0.9 * t;
        ctx.strokeStyle = `rgb(${r},${g},${bl})`;
        ctx.lineWidth   = 0.35 + 1.15 * t;

        ctx.beginPath();
        for (let i = b; i <= bEnd; i++) {
          const [x, y, z] = ringGet(rb, i, visible);
          const [sx, sy]  = proj(x, y, z);
          if (i === b) ctx.moveTo(cx + sx * scale, cy - sy * scale);
          else         ctx.lineTo(cx + sx * scale, cy - sy * scale);
        }
        ctx.stroke();
      }
    }

    ctx.restore();

    // ─── Head dots ───────────────────────────────────────────────────────────

    const glowR = Math.max(5, Math.min(W, H) * 0.013 * zoomRef.current);
    for (const p of particles) {
      const [hx3, hy3, hz3] = p.pos;
      const [hsx, hsy] = proj(hx3, hy3, hz3);
      const hx = cx + hsx * scale;
      const hy = cy - hsy * scale;

      const grd = ctx.createRadialGradient(hx, hy, 0, hx, hy, glowR);
      grd.addColorStop(0,   'rgba(255,255,255,0.9)');
      grd.addColorStop(0.3, `rgba(${def.accentRgb},0.6)`);
      grd.addColorStop(1,   `rgba(${def.accentRgb},0)`);
      ctx.fillStyle = grd;
      ctx.beginPath();
      ctx.arc(hx, hy, glowR, 0, Math.PI * 2);
      ctx.fill();
    }

    // ─── Velocity legend bar ─────────────────────────────────────────────────

    if (colorScheme === 'velocity') {
      const barW = Math.round(W * 0.26);
      const barH = Math.round(H * 0.016);
      const barX = Math.round((W - barW) / 2);
      const barY = H - Math.round(H * 0.05);
      const fs   = Math.max(10, Math.round(H * 0.02));

      const grad = ctx.createLinearGradient(barX, 0, barX + barW, 0);
      for (let i = 0; i <= 20; i++) {
        const t = i / 20;
        const [r, g, b] = COLOR_SCHEMES.velocity(t);
        grad.addColorStop(t, `rgb(${r},${g},${b})`);
      }
      ctx.save();
      ctx.globalAlpha = 0.8;
      ctx.fillStyle   = grad;
      ctx.beginPath();
      ctx.roundRect(barX, barY, barW, barH, barH / 2);
      ctx.fill();
      ctx.globalAlpha  = 1;
      ctx.fillStyle    = 'rgba(255,255,255,0.65)';
      ctx.font         = `${fs}px sans-serif`;
      ctx.textBaseline = 'bottom';
      ctx.textAlign    = 'left';
      ctx.fillText('slow', barX, barY - 3);
      ctx.textAlign = 'right';
      ctx.fillText('fast', barX + barW, barY - 3);
      ctx.restore();
    }

    rafRef.current = requestAnimationFrame(draw);
  }, []);

  // Start animation loop on mount
  useEffect(() => {
    const def = ATTRACTOR_CATALOGUE.find(a => a.id === DEFAULT_ID)!;
    particlesRef.current = buildParticles(def, def.params.map(p => p.default));
    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [draw]);

  // DPR-aware canvas resize
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

  // Drag-to-rotate + scroll-to-zoom
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const onDown = (e: MouseEvent) => {
      dragRef.current = { x: e.clientX, y: e.clientY };
      canvas.style.cursor = 'grabbing';
    };
    const onMove = (e: MouseEvent) => {
      if (!dragRef.current) return;
      rotRef.current.y += (e.clientX - dragRef.current.x) * 0.005;
      rotRef.current.x += (e.clientY - dragRef.current.y) * 0.005;
      dragRef.current = { x: e.clientX, y: e.clientY };
    };
    const onUp = () => { dragRef.current = null; canvas.style.cursor = 'grab'; };

    canvas.style.cursor = 'grab';
    canvas.addEventListener('mousedown', onDown);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup',   onUp);

    const onWheel = (e: WheelEvent) => {
      if (sidebarRef.current?.contains(e.target as Node)) return;
      const r = canvas.getBoundingClientRect();
      if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) return;
      e.preventDefault();
      const delta = e.deltaMode === 1 ? e.deltaY * 20 : e.deltaMode === 2 ? e.deltaY * 400 : e.deltaY;
      zoomRef.current = Math.max(0.2, Math.min(8, zoomRef.current * (1 - delta * 0.001)));
    };
    window.addEventListener('wheel', onWheel, { passive: false });

    return () => {
      canvas.removeEventListener('mousedown', onDown);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup',   onUp);
      window.removeEventListener('wheel',     onWheel);
    };
  }, []);

  const currentDef = ATTRACTOR_CATALOGUE.find(a => a.id === attractorId)!;

  return (
    <div className={styles.container}>
      <canvas ref={canvasRef} className={styles.canvas} />

      {/* ── Sidebar ──────────────────────────────────────────────────────── */}
      <div ref={sidebarRef} className={styles.sidebar}>

        <ControlPanel title="Attractor">
          <ControlGroup>
            <SelectControl
              label="Type"
              value={attractorId}
              onChange={handleAttractorChange}
              options={ATTRACTOR_CATALOGUE.map(a => ({ value: a.id, label: a.name }))}
            />
          </ControlGroup>

          <div
            className={styles.equationBlock}
            style={{ '--eq-accent': currentDef.accentColor } as React.CSSProperties}
          >
            {currentDef.equations.map((eq, i) => (
              <span key={i} className={styles.equationLine}>{eq}</span>
            ))}
          </div>

          <ControlGroup>
            {currentDef.params.map(p => (
              <Slider
                key={`${attractorId}-${p.key}`}
                label={p.label}
                value={paramValues[p.key] ?? p.default}
                onChange={v => handleParamChange(p.key, v)}
                min={p.min}
                max={p.max}
                step={p.step}
                format={v => v.toFixed(Math.max(0, Math.ceil(-Math.log10(p.step))))}
                manualInput
              />
            ))}
          </ControlGroup>
        </ControlPanel>

        <ControlPanel title="Rendering">
          <ControlGroup>
            <SelectControl
              label="Color"
              value={colorScheme}
              onChange={v => setColorScheme(v as ColorScheme)}
              options={[
                { value: 'velocity', label: 'Velocity' },
                { value: 'heat',    label: 'Heat'     },
                { value: 'plasma',  label: 'Plasma'   },
                { value: 'neon',    label: 'Neon'     },
              ]}
            />
            <Slider
              label="Trail"
              value={trailLength}
              onChange={setTrailLength}
              min={500}
              max={MAX_BUF}
              step={500}
              format={v => (v / 1000).toFixed(1)}
              unit="k pts"
            />
            <Slider
              label="Speed"
              value={speed}
              onChange={setSpeed}
              min={1}
              max={30}
              step={1}
              unit="steps/frame"
            />
          </ControlGroup>
        </ControlPanel>

        <ControlPanel title="Controls">
          <ControlGroup>
            <Toggle label="Running"     value={running}    onChange={setRunning}    />
            <Toggle label="Auto-rotate" value={autoRotate} onChange={setAutoRotate} />
          </ControlGroup>
          <ControlGroup>
            <button className={styles.resetBtn} type="button" onClick={reset}>
              Reset
            </button>
          </ControlGroup>
        </ControlPanel>

      </div>

      {/* ── HUD ──────────────────────────────────────────────────────────── */}
      <div className={styles.hud}>
        <div className={styles.hudLeft}>
          <span
            className={styles.hudTitle}
            style={{ color: currentDef.accentColor }}
          >
            {currentDef.name}
          </span>
          <span className={styles.hudCategory}>Strange Attractor</span>
        </div>
        <span className={styles.hudHint}>drag to rotate · scroll to zoom</span>
      </div>
    </div>
  );
}
