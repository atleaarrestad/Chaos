import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Slider, Toggle, SelectControl,
  ControlPanel, ControlGroup,
} from '@/components/Controls';
import styles from './Lorenz.module.css';

// ─── Types ───────────────────────────────────────────────────────────────────

type ColorScheme = 'lorenz' | 'heat' | 'plasma' | 'neon' | 'velocity';

// ─── Attractor ───────────────────────────────────────────────────────────────

interface AttractorDef {
  id: string;
  sigma: number;
  rho: number;
  beta: number;
  color: string; // 'r,g,b'
}

interface AttractorRuntime {
  id: string;
  pos: [number, number, number];
  rb: RingBuf;
}

const ATTRACTOR_PALETTE = [
  '129,140,248',  // indigo  (primary)
  '251,191,36',   // amber
  '251,113,133',  // rose
  '52,211,153',   // emerald
  '251,146,60',   // orange
  '167,139,250',  // violet
  '34,211,238',   // cyan
];

const DEFAULT_ATTRACTOR: AttractorDef = {
  id: 'a0', sigma: 10, rho: 28, beta: 8 / 3, color: ATTRACTOR_PALETTE[0],
};

const MAX_SPEED = 200; // typical top speed for classic Lorenz params

// ─── Ring buffer ─────────────────────────────────────────────────────────────

const MAX_BUF = 10_000;

interface RingBuf {
  xyz: Float32Array;
  head: number;   // next write slot
  total: number;  // total points ever pushed
}

const makeRing = (): RingBuf => ({
  xyz: new Float32Array(MAX_BUF * 3),
  head: 0,
  total: 0,
});

function ringPush(rb: RingBuf, x: number, y: number, z: number): void {
  const i = rb.head * 3;
  rb.xyz[i] = x; rb.xyz[i + 1] = y; rb.xyz[i + 2] = z;
  rb.head = (rb.head + 1) % MAX_BUF;
  rb.total++;
}

/** Return logical point i (0 = oldest visible, visible-1 = newest). */
function ringGet(rb: RingBuf, i: number, visible: number): [number, number, number] {
  const slot = ((rb.head - visible + i) % MAX_BUF + MAX_BUF) % MAX_BUF;
  const b = slot * 3;
  return [rb.xyz[b], rb.xyz[b + 1], rb.xyz[b + 2]];
}

// ─── RK4 integration ─────────────────────────────────────────────────────────

function rk4(
  x: number, y: number, z: number,
  sigma: number, rho: number, beta: number, dt: number,
): [number, number, number] {
  const f = (x: number, y: number, z: number): [number, number, number] => [
    sigma * (y - x),
    x * (rho - z) - y,
    x * y - beta * z,
  ];
  const h = dt * 0.5;
  const [k1x, k1y, k1z] = f(x, y, z);
  const [k2x, k2y, k2z] = f(x + k1x * h, y + k1y * h, z + k1z * h);
  const [k3x, k3y, k3z] = f(x + k2x * h, y + k2y * h, z + k2z * h);
  const [k4x, k4y, k4z] = f(x + k3x * dt, y + k3y * dt, z + k3z * dt);
  const s = dt / 6;
  return [
    x + (k1x + 2 * k2x + 2 * k3x + k4x) * s,
    y + (k1y + 2 * k2y + 2 * k3y + k4y) * s,
    z + (k1z + 2 * k2z + 2 * k3z + k4z) * s,
  ];
}

// ─── Color schemes ────────────────────────────────────────────────────────────

type RGB = readonly [number, number, number];

const c = (v: number): number => Math.max(0, Math.min(255, v | 0));

const COLORS: Record<ColorScheme, (t: number) => RGB> = {
  // Dark indigo → bright indigo-white (matching --col-lorenz)
  lorenz: t => [c(30 + 200 * t * t), c(30 + 180 * t * t), c(100 + 155 * t)],

  // Black → red → orange → yellow → white
  heat: t => {
    const v = t * 3;
    return [c(v * 255), c(Math.max(0, v - 1) * 255), c(Math.max(0, v - 2) * 255)];
  },

  // Deep purple → magenta → cyan (plasma-like)
  plasma: t => [c(13 + 220 * t), c(8 + 240 * t * t), c(135 - 70 * t)],

  // Black → dark teal → bright mint-green
  neon: t => [c(80 * t * t), c(30 + 210 * t), c(60 + 160 * t * (1 - 0.4 * t))],

  // Slow = deep blue → cyan → green → yellow → fast = white
  velocity: t => {
    // deep blue → cyan → green → yellow → orange → red
    if (t < 0.20) { const s = t / 0.20;           return [c(10),              c(20 + 180 * s),  c(160 + 60 * s)]; }
    if (t < 0.40) { const s = (t - 0.20) / 0.20;  return [c(10 + 20 * s),    c(200 - 20 * s),  c(220 - 190 * s)]; }
    if (t < 0.60) { const s = (t - 0.40) / 0.20;  return [c(30 + 180 * s),   c(180 + 30 * s),  c(30)]; }
    if (t < 0.80) { const s = (t - 0.60) / 0.20;  return [c(210 + 45 * s),   c(210 - 70 * s),  c(30 - 10 * s)]; }
    const s = (t - 0.80) / 0.20;                   return [c(255),            c(140 - 100 * s), c(20)]; },
};

// ─── Component ───────────────────────────────────────────────────────────────

interface Params {
  attractors: AttractorDef[];
  dt: number;
  speed: number; trailLength: number; colorScheme: ColorScheme;
  running: boolean; showAxes: boolean; autoRotate: boolean;
}

// ─── Bounding box for the 3D room ────────────────────────────────────────────

const BOX = {
  x: [-25, 25] as [number, number],
  y: [-30, 30] as [number, number],
  z: [  0, 50] as [number, number],
  step: 10,
};

export default function Lorenz() {
  const [attractors,   setAttractors]  = useState<AttractorDef[]>([{ ...DEFAULT_ATTRACTOR }]);
  const [dt,          setDt]          = useState(0.002);
  const [speed,       setSpeed]       = useState(4);
  const [trailLength, setTrailLength] = useState(10_000);
  const [colorScheme, setColorScheme] = useState<ColorScheme>('velocity');
  const [running,     setRunning]     = useState(true);
  const [showAxes,    setShowAxes]    = useState(false);
  const [autoRotate,  setAutoRotate]  = useState(false);

  const canvasRef          = useRef<HTMLCanvasElement>(null);
  const sidebarRef         = useRef<HTMLDivElement>(null);
  const rafRef             = useRef(0);
  const attractorStatesRef = useRef<AttractorRuntime[]>([{ id: 'a0', pos: [0.1, 0, 0], rb: makeRing() }]);
  const rotRef             = useRef({ x: 0.4, y: 0.5 });
  const dragRef            = useRef<{ x: number; y: number } | null>(null);
  const zoomRef            = useRef(1);

  // Mutable params ref — the rAF loop reads this to avoid stale closures.
  const pRef = useRef<Params>({ attractors, dt, speed, trailLength, colorScheme, running, showAxes, autoRotate });

  useEffect(() => {
    pRef.current = { attractors, dt, speed, trailLength, colorScheme, running, showAxes, autoRotate };
  }, [attractors, dt, speed, trailLength, colorScheme, running, showAxes, autoRotate]);

  const reset = useCallback(() => {
    attractorStatesRef.current.forEach(s => { s.pos = [0.1, 0, 0]; s.rb = makeRing(); });
  }, []);

  const addAttractor = useCallback(() => {
    setAttractors(prev => {
      const id = `a${Date.now()}`;
      const color = ATTRACTOR_PALETTE[prev.length % ATTRACTOR_PALETTE.length];
      attractorStatesRef.current.push({ id, pos: [0.1, 0, 0], rb: makeRing() });
      return [...prev, { id, sigma: 10, rho: 28, beta: 8 / 3, color }];
    });
  }, []);

  const removeAttractor = useCallback((id: string) => {
    setAttractors(prev => prev.filter(a => a.id !== id));
    attractorStatesRef.current = attractorStatesRef.current.filter(s => s.id !== id);
  }, []);

  const updateAttractor = useCallback((id: string, field: 'sigma' | 'rho' | 'beta', value: number) => {
    setAttractors(prev => prev.map(a => a.id === id ? { ...a, [field]: value } : a));
  }, []);


  // rotY=yaw, rotX=pitch. Derivation from proj():
  //   (0, 0)       → sees XY plane  (top-down along Z)
  //   (π/2, 0)     → sees XZ plane  (front, classic butterfly)
  //   (0, π/2)     → sees YZ plane  (side, along X)
  const SNAPS = [
    { label: 'Top\u00A0(Z)', rx: 0,          ry: 0          },
    { label: 'Front\u00A0(Y)', rx: Math.PI / 2, ry: 0          },
    { label: 'Side\u00A0(X)', rx: 0,          ry: Math.PI / 2 },
  ] as const;

  const snapTo = useCallback((rx: number, ry: number) => {
    rotRef.current = { x: rx, y: ry };
  }, []);

  // ─── Animation loop ───────────────────────────────────────────────────────

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) { rafRef.current = requestAnimationFrame(draw); return; }

    const ctx = canvas.getContext('2d')!;
    const W = canvas.width, H = canvas.height;
    const { attractors, dt, speed, trailLength, colorScheme, running, showAxes, autoRotate } = pRef.current;
    const states = attractorStatesRef.current;

    // Auto-rotate: nudge yaw each frame when not dragging
    if (autoRotate && !dragRef.current) {
      rotRef.current.y += 0.004;
    }

    const { x: rotX, y: rotY } = rotRef.current;

    // Advance all attractors
    if (running) {
      for (const state of states) {
        const def = attractors.find(a => a.id === state.id);
        if (!def) continue;
        let [px, py, pz] = state.pos;
        for (let i = 0; i < speed; i++) {
          [px, py, pz] = rk4(px, py, pz, def.sigma, def.rho, def.beta, dt);
          if (!isFinite(px) || !isFinite(py) || !isFinite(pz)) {
            [px, py, pz] = [0.1, 0, 0];
            state.rb = makeRing();
            break;
          }
          ringPush(state.rb, px, py, pz);
        }
        state.pos = [px, py, pz];
      }
    }

    // Clear canvas
    ctx.fillStyle = '#080812';
    ctx.fillRect(0, 0, W, H);

    // Pre-compute rotation trig
    const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
    const cosX = Math.cos(rotX), sinX = Math.sin(rotX);

    // Orthographic projection: rotate around Y then X
    const proj = (x: number, y: number, z: number): [number, number] => {
      const x1 = x * cosY + z * sinY;
      const z1 = -x * sinY + z * cosY;
      return [x1, y * cosX - z1 * sinX];
    };

    const scale = Math.min(W, H) / 80 * zoomRef.current;

    // Centre the view on the attractor's midpoint (0, 0, 27)
    const [ox, oy] = proj(0, 0, 27);
    const cx = W * 0.5 - ox * scale;
    const cy = H * 0.5 + oy * scale;

    // ─── 3D room: back-wall panels + grid + box edges + tick labels ─────────
    if (showAxes) {
      const { x: BX, y: BY, z: BZ, step: STEP } = BOX;

      const toS = (wx: number, wy: number, wz: number): [number, number] => {
        const [sx, sy] = proj(wx, wy, wz);
        return [cx + sx * scale, cy - sy * scale];
      };

      // Signed depth of a face normal under the current rotation.
      // < 0  →  face points away from viewer  →  back face (draw it).
      const faceDepth = (nx: number, ny: number, nz: number): number => {
        const z1 = -nx * sinY + nz * cosY;
        return ny * sinX + z1 * cosX;
      };

      const FACES: { n: [number, number, number]; c: [number, number, number][] }[] = [
        { n: [ 1, 0, 0], c: [[BX[1],BY[0],BZ[0]], [BX[1],BY[1],BZ[0]], [BX[1],BY[1],BZ[1]], [BX[1],BY[0],BZ[1]]] },
        { n: [-1, 0, 0], c: [[BX[0],BY[1],BZ[0]], [BX[0],BY[0],BZ[0]], [BX[0],BY[0],BZ[1]], [BX[0],BY[1],BZ[1]]] },
        { n: [ 0, 1, 0], c: [[BX[1],BY[1],BZ[0]], [BX[0],BY[1],BZ[0]], [BX[0],BY[1],BZ[1]], [BX[1],BY[1],BZ[1]]] },
        { n: [ 0,-1, 0], c: [[BX[0],BY[0],BZ[0]], [BX[1],BY[0],BZ[0]], [BX[1],BY[0],BZ[1]], [BX[0],BY[0],BZ[1]]] },
        { n: [ 0, 0, 1], c: [[BX[0],BY[0],BZ[1]], [BX[1],BY[0],BZ[1]], [BX[1],BY[1],BZ[1]], [BX[0],BY[1],BZ[1]]] },
        { n: [ 0, 0,-1], c: [[BX[0],BY[0],BZ[0]], [BX[1],BY[0],BZ[0]], [BX[1],BY[1],BZ[0]], [BX[0],BY[1],BZ[0]]] },
      ];

      ctx.save();
      ctx.lineJoin = 'round';

      // ── Back face panels + grid ─────────────────────────────────────────────
      for (const face of FACES) {
        if (faceDepth(face.n[0], face.n[1], face.n[2]) >= 0) continue;

        const sc = face.c.map(p => toS(p[0], p[1], p[2]));

        // Panel fill
        ctx.beginPath();
        ctx.moveTo(sc[0][0], sc[0][1]);
        for (let i = 1; i < sc.length; i++) ctx.lineTo(sc[i][0], sc[i][1]);
        ctx.closePath();
        ctx.fillStyle = 'rgba(110,120,160,0.06)';
        ctx.fill();

        // Panel edge
        ctx.strokeStyle = 'rgba(160,170,220,0.28)';
        ctx.lineWidth = 1;
        ctx.setLineDash([]);
        ctx.stroke();

        // Grid lines
        ctx.strokeStyle = 'rgba(160,170,220,0.1)';
        ctx.lineWidth = 0.5;
        const [nx, ny] = face.n;

        if (Math.abs(face.n[2]) > 0.5) {
          // Floor / ceiling — grid on X–Y
          const fz = face.n[2] > 0 ? BZ[1] : BZ[0];
          for (let gx = Math.ceil(BX[0] / STEP) * STEP; gx <= BX[1]; gx += STEP) {
            const a = toS(gx, BY[0], fz), b = toS(gx, BY[1], fz);
            ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
          }
          for (let gy = Math.ceil(BY[0] / STEP) * STEP; gy <= BY[1]; gy += STEP) {
            const a = toS(BX[0], gy, fz), b = toS(BX[1], gy, fz);
            ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
          }
        } else if (Math.abs(nx) > 0.5) {
          // Left / right wall — grid on Y–Z
          const fx = nx > 0 ? BX[1] : BX[0];
          for (let gy = Math.ceil(BY[0] / STEP) * STEP; gy <= BY[1]; gy += STEP) {
            const a = toS(fx, gy, BZ[0]), b = toS(fx, gy, BZ[1]);
            ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
          }
          for (let gz = BZ[0]; gz <= BZ[1]; gz += STEP) {
            const a = toS(fx, BY[0], gz), b = toS(fx, BY[1], gz);
            ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
          }
        } else {
          // Front / back wall — grid on X–Z
          const fy = ny > 0 ? BY[1] : BY[0];
          for (let gx = Math.ceil(BX[0] / STEP) * STEP; gx <= BX[1]; gx += STEP) {
            const a = toS(gx, fy, BZ[0]), b = toS(gx, fy, BZ[1]);
            ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
          }
          for (let gz = BZ[0]; gz <= BZ[1]; gz += STEP) {
            const a = toS(BX[0], fy, gz), b = toS(BX[1], fy, gz);
            ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
          }
        }
      }

      // ── Tick labels ─────────────────────────────────────────────────────────
      // Pick outer edge for each axis based on which back faces exist
      const negYBack = faceDepth(0, -1, 0) < 0;
      const negXBack = faceDepth(-1, 0, 0) < 0;
      const posXBack = faceDepth( 1, 0, 0) < 0;
      const OFS = 4; // world-unit offset to place labels outside the box

      const xEdgeY = negYBack ? BY[0] : BY[1];
      const yEdgeX = negXBack ? BX[0] : BX[1];
      const zEdgeX = posXBack ? BX[1] : BX[0];
      const zEdgeY = negYBack ? BY[0] : BY[1];

      // scale already encodes canvas size + DPR, so font sized by scale stays proportional.
      const fs    = Math.max(14, scale * 1.8);
      const fsLbl = Math.max(18, scale * 2.2);

      ctx.font = `${fs}px var(--font-sans, system-ui)`;
      ctx.fillStyle = 'rgba(180,190,230,0.75)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      for (let x = -20; x <= 20; x += 10) {
        const p = toS(x, xEdgeY + (negYBack ? -OFS : OFS), BZ[0]);
        ctx.fillText(String(x), p[0], p[1]);
      }
      for (let y = -20; y <= 20; y += 10) {
        const p = toS(yEdgeX + (negXBack ? -OFS : OFS), y, BZ[0]);
        ctx.fillText(String(y), p[0], p[1]);
      }
      for (let z = 0; z <= 50; z += 10) {
        const p = toS(zEdgeX + (posXBack ? OFS : -OFS), zEdgeY + (negYBack ? -OFS * 0.5 : OFS * 0.5), z);
        ctx.fillText(String(z), p[0], p[1]);
      }

      // Axis labels
      ctx.font = `bold ${fsLbl}px var(--font-sans, system-ui)`;
      ctx.fillStyle = 'rgba(210,220,250,0.9)';
      const xLbl = toS(0, xEdgeY + (negYBack ? -OFS * 4 : OFS * 4), BZ[0]);
      ctx.fillText('X Axis', xLbl[0], xLbl[1]);
      const yLbl = toS(yEdgeX + (negXBack ? -OFS * 4 : OFS * 4), 0, BZ[0]);
      ctx.fillText('Y Axis', yLbl[0], yLbl[1]);
      const zLbl = toS(zEdgeX + (posXBack ? OFS * 3.5 : -OFS * 3.5), zEdgeY + (negYBack ? -OFS * 1.5 : OFS * 1.5), 25);
      ctx.fillText('Z Axis', zLbl[0], zLbl[1]);

      ctx.restore();
    }

    // ─── Draw all attractor trails ────────────────────────────────────────
    const isVelocity = colorScheme === 'velocity';

    ctx.save();
    ctx.lineJoin = 'round';
    ctx.lineCap  = 'round';

    for (let ai = 0; ai < states.length; ai++) {
      const state = states[ai];
      const def   = attractors.find(a => a.id === state.id);
      if (!def) continue;

      const rb      = state.rb;
      const visible = Math.min(rb.total, trailLength, MAX_BUF);
      if (visible < 2) continue;

      const BATCH = Math.max(10, Math.ceil(visible / 150));

      for (let b = 0; b < visible - 1; b += BATCH) {
        const bEnd = Math.min(b + BATCH, visible - 1);
        const t    = (b + bEnd) * 0.5 / visible;

        let r: number, g: number, bl: number;
        if (isVelocity) {
          const midI    = Math.floor((b + bEnd) / 2);
          const [mx, my, mz] = ringGet(rb, midI, visible);
          const [nx, ny, nz] = ringGet(rb, Math.min(midI + 1, visible - 1), visible);
          const spd = Math.sqrt((nx - mx) ** 2 + (ny - my) ** 2 + (nz - mz) ** 2) / dt;
          // log + power curve: compresses the high end so typical speeds
          // land in the blue/green range and only peak transitions go red
          const raw = Math.min(1, Math.log1p(spd) / Math.log1p(MAX_SPEED));
          const vt  = Math.pow(raw, 3);
          [r, g, bl] = COLORS.velocity(vt);
        } else if (ai === 0) {
          [r, g, bl] = COLORS[colorScheme](t);
        } else {
          // Additional attractors: use their palette color, dimmed at tail
          const [pr, pg, pb] = def.color.split(',').map(Number);
          r = c(pr * (0.15 + 0.85 * t));
          g = c(pg * (0.15 + 0.85 * t));
          bl = c(pb * (0.15 + 0.85 * t));
        }

        ctx.globalAlpha = 0.15 + 0.85 * t;
        ctx.strokeStyle = `rgb(${r},${g},${bl})`;
        ctx.lineWidth   = 0.4 + 1.2 * t;

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

    // ─── Head dots for all attractors ─────────────────────────────────────
    const glowR = Math.max(6, 10 * scale / 8);
    for (let ai = 0; ai < states.length; ai++) {
      const state = states[ai];
      const def   = attractors.find(a => a.id === state.id);
      if (!def) continue;

      const [hx3, hy3, hz3] = state.pos;
      const [hsx, hsy] = proj(hx3, hy3, hz3);
      const hx = cx + hsx * scale, hy = cy - hsy * scale;
      const r  = ai === 0 ? glowR : glowR * 0.8;

      const grd = ctx.createRadialGradient(hx, hy, 0, hx, hy, r);
      grd.addColorStop(0,   'rgba(255,255,255,0.95)');
      grd.addColorStop(0.3, ai === 0 ? 'rgba(180,200,255,0.55)' : `rgba(${def.color},0.6)`);
      grd.addColorStop(1,   ai === 0 ? 'rgba(129,140,248,0)'    : `rgba(${def.color},0)`);
      ctx.fillStyle = grd;
      ctx.beginPath();
      ctx.arc(hx, hy, r, 0, Math.PI * 2);
      ctx.fill();
    }

    // ─── Velocity legend bar ───────────────────────────────────────────────
    if (colorScheme === 'velocity') {
      const barW  = Math.round(W * 0.28);
      const barH  = Math.round(H * 0.018);
      const barX  = Math.round((W - barW) / 2);
      const barY  = H - Math.round(H * 0.055);
      const fs    = Math.max(10, Math.round(H * 0.022));

      const grad = ctx.createLinearGradient(barX, 0, barX + barW, 0);
      for (let i = 0; i <= 20; i++) {
        const t = i / 20;
        const [r, g, b] = COLORS.velocity(t);
        grad.addColorStop(t, `rgb(${r},${g},${b})`);
      }

      ctx.save();
      ctx.globalAlpha = 0.85;
      ctx.fillStyle   = grad;
      ctx.beginPath();
      ctx.roundRect(barX, barY, barW, barH, barH / 2);
      ctx.fill();

      ctx.globalAlpha  = 1;
      ctx.fillStyle    = 'rgba(255,255,255,0.7)';
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

  // Start animation loop (once on mount, cleaned up on unmount)
  useEffect(() => {
    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [draw]);

  // Resize canvas to match layout (DPR-aware)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      canvas.width  = (rect.width  * dpr) | 0;
      canvas.height = (rect.height * dpr) | 0;
    });
    ro.observe(canvas);
    return () => ro.disconnect();
  }, []);

  // Drag-to-rotate
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
    const onUp = () => {
      if (!dragRef.current) return;
      dragRef.current = null;
      canvas.style.cursor = 'grab';
    };

    canvas.style.cursor = 'grab';
    canvas.addEventListener('mousedown', onDown);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);

    const onWheel = (e: WheelEvent) => {
      const c = canvasRef.current;
      if (!c) return;
      // Let the sidebar handle its own scrolling
      if (sidebarRef.current?.contains(e.target as Node)) return;
      const r = c.getBoundingClientRect();
      if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) return;
      e.preventDefault();
      const delta = e.deltaMode === 1 ? e.deltaY * 20
                  : e.deltaMode === 2 ? e.deltaY * 400
                  : e.deltaY;
      zoomRef.current = Math.max(0.2, Math.min(8, zoomRef.current * (1 - delta * 0.001)));
    };
    window.addEventListener('wheel', onWheel, { passive: false });

    return () => {
      canvas.removeEventListener('mousedown', onDown);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      window.removeEventListener('wheel', onWheel);
    };
  }, []);

  // ─── Render ───────────────────────────────────────────────────────────────

  return (
    <div className={styles.container}>
      <canvas ref={canvasRef} className={styles.canvas} />

      <div ref={sidebarRef} className={styles.sidebar}>
        <ControlPanel title="Attractors">
          <div className={styles.attractorList}>
            {attractors.map((a, idx) => (
            <div key={a.id} className={styles.attractorItem}>
              <div className={styles.attractorHeader}>
                <span className={styles.attractorDot} style={{ background: `rgb(${a.color})` }} />
                <span className={styles.attractorLabel}>Attractor {idx + 1}</span>
                {attractors.length > 1 && (
                  <button className={styles.removeBtn} type="button" onClick={() => removeAttractor(a.id)}>✕</button>
                )}
              </div>
              <div className={styles.attractorControls}>
                <Slider label="σ" value={a.sigma} onChange={v => updateAttractor(a.id, 'sigma', v)} min={0} max={30} step={0.1} manualInput />
                <Slider label="ρ" value={a.rho}   onChange={v => updateAttractor(a.id, 'rho',   v)} min={0} max={80} step={0.5} manualInput />
                <Slider label="β" value={a.beta}  onChange={v => updateAttractor(a.id, 'beta',  v)} min={0} max={10} step={0.001} format={v => v.toFixed(3)} manualInput />
              </div>
            </div>
          ))}
          </div>
          <ControlGroup>
            <button className={styles.addBtn} type="button" onClick={addAttractor}>
              + Add Attractor
            </button>
          </ControlGroup>
        </ControlPanel>

        <ControlPanel title="Animation">
          <ControlGroup>
            <Toggle label="Running" value={running} onChange={setRunning} />
            <Toggle
              label="Auto-rotate"
              value={autoRotate}
              onChange={setAutoRotate}
              description="Slowly spin the view"
            />
            <Slider
              label="Speed" value={speed} onChange={setSpeed}
              min={1} max={30} step={1} unit="steps/frame"
            />
            <Slider
              label="dt" value={dt} onChange={setDt}
              min={0.0001} max={0.02} step={0.0001} format={v => v.toFixed(4)}
            />
            <Slider
              label="Trail length" value={trailLength} onChange={setTrailLength}
              min={100} max={10000} step={100}
            />
          </ControlGroup>
        </ControlPanel>

        <ControlPanel title="Display" defaultOpen={false}>
          <ControlGroup>
            <Toggle
              label="Show axes"
              value={showAxes}
              onChange={setShowAxes}
              description="X / Y / Z reference frame"
            />
            <SelectControl
              label="Color scheme"
              value={colorScheme}
              onChange={setColorScheme}
              options={[
                { value: 'lorenz'   as const, label: 'Lorenz (indigo)' },
                { value: 'heat'     as const, label: 'Heat map' },
                { value: 'plasma'   as const, label: 'Plasma' },
                { value: 'neon'     as const, label: 'Neon green' },
                { value: 'velocity' as const, label: 'Velocity' },
              ]}
            />
          </ControlGroup>
        </ControlPanel>

        <div className={styles.snapRow}>
          {SNAPS.map(s => (
            <button
              key={s.label}
              className={styles.snapBtn}
              type="button"
              onClick={() => snapTo(s.rx, s.ry)}
            >
              {s.label}
            </button>
          ))}
        </div>

        <button className={styles.resetBtn} type="button" onClick={reset}>
          Reset Trajectory
        </button>
      </div>

      <div className={styles.hud}>
        <div className={styles.hudLeft}>
          <span className={styles.hudTitle}>Lorenz Attractor</span>
        </div>
        <div className={styles.hudRight}>
          <span className={styles.hudHint}>drag to rotate</span>
        </div>
      </div>
    </div>
  );
}
