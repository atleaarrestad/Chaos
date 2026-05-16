import { useState, useEffect, useRef, useCallback, type CSSProperties } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Info } from 'lucide-react';
import {
  Slider, Toggle, SelectControl,
  ControlPanel, ControlGroup, SimControls,
} from '@/components/Controls';
import { InfoDialog } from '@/components/InfoDialog';
import { detectWebGL2 } from '@/lib/gpu/context';
import { useFullscreen } from '@/hooks/useFullscreen';
import { getStrParam, useShareUrl } from '@/hooks/useUrlParams';
import ExportDialog from '../../components/ExportDialog/ExportDialog';
import { exportImage } from '../../lib/exportImage';
import styles from './Lorenz.module.css';
import { AttractorGPU, type AttractorGPUParams, type AttractorDerivFn } from './attractor-gpu';

type Vec3 = readonly [number, number, number];
type MutVec3 = [number, number, number];
type ColorScheme = 'lorenz' | 'heat' | 'plasma' | 'neon' | 'velocity';
type SectionAxis = 'x' | 'y' | 'z';

type RGB = readonly [number, number, number];

interface ParamDef {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
}

interface AttractorTypeDef {
  id: string;
  name: string;
  equations: string[];
  description: string;
  fn: AttractorDerivFn;
  dt: number;
  warmupSteps: number;
  initPos: Vec3;
  center: Vec3;
  viewUnits: number;
  maxSpeed: number;
  defaultView: { rx: number; ry: number };
  defaultSectionAxis: SectionAxis;
  defaultSectionVal: number;
  params: ParamDef[];
  accentColor: string;
  accentRgb: string;
  gpuType: number;
  sectionBounds: Record<SectionAxis, readonly [number, number, number, number]>;
  returnBounds: readonly [number, number, number, number];
  box: { x: [number, number]; y: [number, number]; z: [number, number]; step: number };
}

interface AttractorInstance {
  id: string;
  color: string;
  paramValues: Record<string, number>;
}

interface RingBuf {
  xyz: Float32Array;
  head: number;
  total: number;
}

interface PoincareBuf {
  xy: Float32Array;
  head: number;
  total: number;
  minU: number;
  maxU: number;
  minV: number;
  maxV: number;
}

interface AttractorRuntime {
  id: string;
  pos: MutVec3;
  rb: RingBuf;
  poincare: PoincareBuf;
}

interface Params {
  attractorTypeId: string;
  instances: AttractorInstance[];
  selectedInstanceId: string;
  dt: number;
  speed: number;
  trailLength: number;
  colorScheme: ColorScheme;
  running: boolean;
  showAxes: boolean;
  autoRotate: boolean;
  showPoincare: boolean;
  showReturnMap: boolean;
  poincareZ: number;
  sectionAxis: SectionAxis;
}

const MAX_BUF = 10_000;
const MAX_POINCARE = 6_000;

const ATTRACTOR_PALETTE = [
  '129,140,248',
  '251,191,36',
  '251,113,133',
  '52,211,153',
  '251,146,60',
  '167,139,250',
  '34,211,238',
];

const DEFAULT_LORENZ_PARAMS: Record<string, number> = { sigma: 10, rho: 28, beta: 8 / 3 };

const makeInstance = (id: string, idx: number, paramValues: Record<string, number>): AttractorInstance => ({
  id,
  color: ATTRACTOR_PALETTE[idx % ATTRACTOR_PALETTE.length],
  paramValues,
});

const SNAPS = [
  { label: 'XY', rx: 0, ry: 0 },
  { label: 'XZ', rx: Math.PI / 2, ry: 0 },
  { label: 'YZ', rx: 0, ry: Math.PI / 2 },
  { label: 'ISO', rx: 0.6, ry: 0.7 },
] as const;

const c = (v: number): number => Math.max(0, Math.min(255, v | 0));

const COLORS: Record<ColorScheme, (t: number) => RGB> = {
  lorenz: t => [c(30 + 200 * t * t), c(30 + 180 * t * t), c(100 + 155 * t)],
  heat: t => {
    const v = t * 3;
    return [c(v * 255), c(Math.max(0, v - 1) * 255), c(Math.max(0, v - 2) * 255)];
  },
  plasma: t => [c(13 + 220 * t), c(8 + 240 * t * t), c(135 - 70 * t)],
  neon: t => [c(80 * t * t), c(30 + 210 * t), c(60 + 160 * t * (1 - 0.4 * t))],
  velocity: t => {
    if (t < 0.20) { const s = t / 0.20; return [c(10), c(20 + 180 * s), c(160 + 60 * s)]; }
    if (t < 0.40) { const s = (t - 0.20) / 0.20; return [c(10 + 20 * s), c(200 - 20 * s), c(220 - 190 * s)]; }
    if (t < 0.60) { const s = (t - 0.40) / 0.20; return [c(30 + 180 * s), c(180 + 30 * s), c(30)]; }
    if (t < 0.80) { const s = (t - 0.60) / 0.20; return [c(210 + 45 * s), c(210 - 70 * s), c(30 - 10 * s)]; }
    const s = (t - 0.80) / 0.20;
    return [c(255), c(140 - 100 * s), c(20)];
  },
};

const ATTRACTOR_CATALOGUE: AttractorTypeDef[] = [
  {
    id: 'lorenz', name: 'Lorenz',
    equations: ['dx/dt = σ(y − x)', 'dy/dt = x(ρ − z) − y', 'dz/dt = xy − βz'],
    description: 'The original strange attractor. Discovered by Edward Lorenz in 1963 while studying atmospheric convection.',
    fn: (x, y, z, [s, r, b]) => [s * (y - x), x * (r - z) - y, x * y - b * z] as [number, number, number],
    dt: 0.002, warmupSteps: 8000, initPos: [0.1, 0, 0],
    center: [0, 0, 27], viewUnits: 80, maxSpeed: 200,
    defaultView: { rx: 0.4, ry: 0.5 },
    defaultSectionAxis: 'y', defaultSectionVal: 0,
    params: [
      { key: 'sigma', label: 'σ', min: 0, max: 30, step: 0.1, default: 10 },
      { key: 'rho', label: 'ρ', min: 0, max: 80, step: 0.5, default: 28 },
      { key: 'beta', label: 'β', min: 0, max: 10, step: 0.001, default: 8 / 3 },
    ],
    accentColor: '#818cf8', accentRgb: '129,140,248',
    gpuType: 0,
    sectionBounds: { y: [-22, 22, -1, 52], z: [-22, 22, -28, 28], x: [-28, 28, -1, 52] },
    returnBounds: [30, 48, 30, 48],
    box: { x: [-25, 25], y: [-30, 30], z: [0, 50], step: 10 },
  },
  {
    id: 'rossler', name: 'Rössler',
    equations: ['dx/dt = −y − z', 'dy/dt = x + ay', 'dz/dt = b + z(x − c)'],
    description: 'A single-scroll attractor simpler than Lorenz. Parameter c drives period-doubling bifurcations leading to chaos.',
    fn: (x, y, z, [a, b, c]) => [-(y + z), x + a * y, b + z * (x - c)] as [number, number, number],
    dt: 0.01, warmupSteps: 5000, initPos: [0.1, 0, 0],
    center: [0, 0, 12], viewUnits: 36, maxSpeed: 30,
    defaultView: { rx: 0.3, ry: 0.5 },
    defaultSectionAxis: 'y', defaultSectionVal: 0,
    params: [
      { key: 'a', label: 'a', min: 0.1, max: 0.5, step: 0.01, default: 0.2 },
      { key: 'b', label: 'b', min: 0.1, max: 0.5, step: 0.01, default: 0.2 },
      { key: 'c', label: 'c', min: 4.0, max: 12.0, step: 0.01, default: 5.7 },
    ],
    accentColor: '#f59e0b', accentRgb: '245,158,11',
    gpuType: 1,
    sectionBounds: { y: [-12, 12, 0, 20], z: [-12, 12, -12, 12], x: [-12, 12, 0, 20] },
    returnBounds: [0, 20, 0, 20],
    box: { x: [-12, 12], y: [-12, 12], z: [0, 20], step: 5 },
  },
  {
    id: 'halvorsen', name: 'Halvorsen',
    equations: ['dx/dt = −ax − 4y − 4z − y²', 'dy/dt = −ay − 4z − 4x − z²', 'dz/dt = −az − 4x − 4y − x²'],
    description: 'Cyclically symmetric. All three variables are interchangeable under a 120° rotation.',
    fn: (x, y, z, [a]) => [-a * x - 4 * y - 4 * z - y * y, -a * y - 4 * z - 4 * x - z * z, -a * z - 4 * x - 4 * y - x * x] as [number, number, number],
    dt: 0.005, warmupSteps: 10000, initPos: [1, 0, 0],
    center: [0, 0, 0], viewUnits: 22, maxSpeed: 80,
    defaultView: { rx: 0.5, ry: 0.8 },
    defaultSectionAxis: 'y', defaultSectionVal: 0,
    params: [
      { key: 'a', label: 'a', min: 0.8, max: 2.0, step: 0.01, default: 1.4 },
    ],
    accentColor: '#22d3ee', accentRgb: '34,211,238',
    gpuType: 2,
    sectionBounds: { y: [-8, 8, -8, 8], z: [-8, 8, -8, 8], x: [-8, 8, -8, 8] },
    returnBounds: [-6, 6, -6, 6],
    box: { x: [-8, 8], y: [-8, 8], z: [-8, 8], step: 4 },
  },
  {
    id: 'thomas', name: 'Thomas',
    equations: ['dx/dt = sin(y) − bx', 'dy/dt = sin(z) − by', 'dz/dt = sin(x) − bz'],
    description: "René Thomas's cyclically symmetric system. Near b ≈ 0.208 the orbit sits at the edge of chaos.",
    fn: (x, y, z, [b]) => [Math.sin(y) - b * x, Math.sin(z) - b * y, Math.sin(x) - b * z] as [number, number, number],
    dt: 0.05, warmupSteps: 2000, initPos: [0.1, 0, 0],
    center: [0, 0, 0], viewUnits: 11, maxSpeed: 2,
    defaultView: { rx: 0.4, ry: 0.6 },
    defaultSectionAxis: 'z', defaultSectionVal: 0,
    params: [
      { key: 'b', label: 'b', min: 0.1, max: 0.35, step: 0.001, default: 0.208186 },
    ],
    accentColor: '#4ade80', accentRgb: '74,222,128',
    gpuType: 3,
    sectionBounds: { y: [-5, 5, -5, 5], z: [-5, 5, -5, 5], x: [-5, 5, -5, 5] },
    returnBounds: [-4, 4, -4, 4],
    box: { x: [-5, 5], y: [-5, 5], z: [-5, 5], step: 2 },
  },
  {
    id: 'aizawa', name: 'Aizawa',
    equations: ['dx/dt = (z−b)x − dy', 'dy/dt = dx + (z−b)y', 'dz/dt = c + az − z³/3 − r²(1+ez) + fzx³'],
    description: 'Orbits spiral around a toroidal manifold, producing delicate layered scrolls.',
    fn: (x, y, z, [a, b, c, d, e, f]) => [
      (z - b) * x - d * y,
      d * x + (z - b) * y,
      c + a * z - (z * z * z) / 3 - (x * x + y * y) * (1 + e * z) + f * z * x * x * x,
    ] as [number, number, number],
    dt: 0.01, warmupSteps: 5000, initPos: [0.1, 0, 0.1],
    center: [0, 0, 0.4], viewUnits: 4, maxSpeed: 5,
    defaultView: { rx: 1.1, ry: 0.3 },
    defaultSectionAxis: 'z', defaultSectionVal: 0.5,
    params: [
      { key: 'a', label: 'a', min: 0.7, max: 1.0, step: 0.01, default: 0.95 },
      { key: 'b', label: 'b', min: 0.5, max: 0.9, step: 0.01, default: 0.7 },
      { key: 'c', label: 'c', min: 0.4, max: 0.8, step: 0.01, default: 0.6 },
      { key: 'd', label: 'd', min: 2.5, max: 4.5, step: 0.1, default: 3.5 },
      { key: 'e', label: 'e', min: 0.1, max: 0.4, step: 0.01, default: 0.25 },
      { key: 'f', label: 'f', min: 0.0, max: 0.2, step: 0.01, default: 0.1 },
    ],
    accentColor: '#f472b6', accentRgb: '244,114,182',
    gpuType: 4,
    sectionBounds: { y: [-1.8, 1.8, -0.6, 1.6], z: [-1.8, 1.8, -1.8, 1.8], x: [-1.8, 1.8, -0.6, 1.6] },
    returnBounds: [-0.5, 1.5, -0.5, 1.5],
    box: { x: [-2, 2], y: [-2, 2], z: [-0.6, 1.6], step: 1 },
  },
  {
    id: 'dadras', name: 'Dadras',
    equations: ['dx/dt = y − px + qyz', 'dy/dt = ry − xz + z', 'dz/dt = sxy − tz'],
    description: 'Two lobes connected by narrow chaotic bridges. Highly sensitive to p, which controls expansion rate.',
    fn: (x, y, z, [p, q, r, s, t]) => [y - p * x + q * y * z, r * y - x * z + z, s * x * y - t * z] as [number, number, number],
    dt: 0.01, warmupSteps: 5000, initPos: [0, 0.3, 0.3],
    center: [0, 0, 6], viewUnits: 20, maxSpeed: 25,
    defaultView: { rx: 0.4, ry: 0.3 },
    defaultSectionAxis: 'y', defaultSectionVal: 0,
    params: [
      { key: 'p', label: 'p', min: 1.0, max: 5.0, step: 0.1, default: 3.0 },
      { key: 'q', label: 'q', min: 1.0, max: 4.0, step: 0.1, default: 2.7 },
      { key: 'r', label: 'r', min: 0.5, max: 3.0, step: 0.1, default: 1.7 },
      { key: 's', label: 's', min: 1.0, max: 4.0, step: 0.1, default: 2.0 },
      { key: 't', label: 't', min: 5.0, max: 15.0, step: 0.1, default: 9.0 },
    ],
    accentColor: '#a855f7', accentRgb: '168,85,247',
    gpuType: 5,
    sectionBounds: { y: [-12, 12, 0, 16], z: [-12, 12, -15, 15], x: [-15, 15, 0, 16] },
    returnBounds: [0, 15, 0, 15],
    box: { x: [-12, 12], y: [-15, 15], z: [0, 16], step: 5 },
  },
];

const makeRing = (): RingBuf => ({
  xyz: new Float32Array(MAX_BUF * 3),
  head: 0,
  total: 0,
});

function ringPush(rb: RingBuf, x: number, y: number, z: number): void {
  const i = rb.head * 3;
  rb.xyz[i] = x;
  rb.xyz[i + 1] = y;
  rb.xyz[i + 2] = z;
  rb.head = (rb.head + 1) % MAX_BUF;
  rb.total++;
}

function ringGet(rb: RingBuf, i: number, visible: number): Vec3 {
  const slot = ((rb.head - visible + i) % MAX_BUF + MAX_BUF) % MAX_BUF;
  const b = slot * 3;
  return [rb.xyz[b], rb.xyz[b + 1], rb.xyz[b + 2]];
}

const makePoincare = (): PoincareBuf => ({
  xy: new Float32Array(MAX_POINCARE * 2),
  head: 0,
  total: 0,
  minU: Infinity,
  maxU: -Infinity,
  minV: Infinity,
  maxV: -Infinity,
});

function poincarePush(pb: PoincareBuf, u: number, v: number): void {
  const i = pb.head * 2;
  pb.xy[i] = u;
  pb.xy[i + 1] = v;
  pb.head = (pb.head + 1) % MAX_POINCARE;
  pb.total++;
  if (u < pb.minU) pb.minU = u;
  if (u > pb.maxU) pb.maxU = u;
  if (v < pb.minV) pb.minV = v;
  if (v > pb.maxV) pb.maxV = v;
}

function rk4(
  x: number,
  y: number,
  z: number,
  fn: AttractorDerivFn,
  params: number[],
  dt: number,
): MutVec3 {
  const h = dt * 0.5;
  const [k1x, k1y, k1z] = fn(x, y, z, params);
  const [k2x, k2y, k2z] = fn(x + k1x * h, y + k1y * h, z + k1z * h, params);
  const [k3x, k3y, k3z] = fn(x + k2x * h, y + k2y * h, z + k2z * h, params);
  const [k4x, k4y, k4z] = fn(x + k3x * dt, y + k3y * dt, z + k3z * dt, params);
  const s = dt / 6;
  return [
    x + (k1x + 2 * k2x + 2 * k3x + k4x) * s,
    y + (k1y + 2 * k2y + 2 * k3y + k4y) * s,
    z + (k1z + 2 * k2z + 2 * k3z + k4z) * s,
  ];
}

function warmupOrbit(def: AttractorTypeDef, params: number[]): MutVec3 {
  let [x, y, z] = def.initPos as MutVec3;
  for (let i = 0; i < def.warmupSteps; i++) {
    const next = rk4(x, y, z, def.fn, params, def.dt);
    if (!isFinite(next[0]) || !isFinite(next[1]) || !isFinite(next[2])) break;
    [x, y, z] = next;
  }
  return [x, y, z];
}


function buildInstanceStates(def: AttractorTypeDef, instances: AttractorInstance[]): AttractorRuntime[] {
  return instances.map((inst) => {
    const pv = def.params.map(p => inst.paramValues[p.key] ?? p.default);
    const pos: MutVec3 = def.id === 'lorenz'
      ? [0.1, 0, 0]
      : warmupOrbit(def, pv);
    return { id: inst.id, pos, rb: makeRing(), poincare: makePoincare() };
  });
}

function decimalsForStep(step: number): number {
  return step < 1 ? Math.max(0, Math.ceil(-Math.log10(step))) : 0;
}

function formatAxisValue(value: number): string {
  if (Math.abs(value) < 1e-9) return '0';
  if (Math.abs(value - Math.round(value)) < 1e-9) return String(Math.round(value));
  return Number(value.toFixed(2)).toString();
}


export default function Lorenz() {
  const [searchParams] = useSearchParams();
  const rawAttractorTypeId = getStrParam(searchParams, 'a', 'lorenz');
  const initialAttractorTypeId = ATTRACTOR_CATALOGUE.some((attractor) => attractor.id === rawAttractorTypeId)
    ? rawAttractorTypeId
    : 'lorenz';
  const initialDef = ATTRACTOR_CATALOGUE.find((attractor) => attractor.id === initialAttractorTypeId)!;
  const initialParamValues = initialDef.id === 'lorenz'
    ? { ...DEFAULT_LORENZ_PARAMS }
    : Object.fromEntries(initialDef.params.map((param) => [param.key, param.default]));

  const [attractorTypeId, setAttractorTypeId] = useState(initialAttractorTypeId);
  const [instances, setInstances] = useState<AttractorInstance[]>([makeInstance('i0', 0, { ...initialParamValues })]);
  const [selectedInstanceId, setSelectedInstanceId] = useState('i0');
  const [dt, setDt] = useState(initialDef.dt);
  const [speed, setSpeed] = useState(4);
  const [trailLength, setTrailLength] = useState(10_000);
  const [colorScheme, setColorScheme] = useState<ColorScheme>('velocity');
  const [running, setRunning] = useState(true);
  const [showInfo, setShowInfo] = useState(false);
  const [showExport, setShowExport] = useState(false);
  const [showAxes, setShowAxes] = useState(false);
  const [autoRotate, setAutoRotate] = useState(false);
  const [showPoincare, setShowPoincare] = useState(false);
  const [poincareZ, setPoincareZ] = useState(initialDef.defaultSectionVal);
  const [sectionAxis, setSectionAxis] = useState<SectionAxis>(initialDef.defaultSectionAxis);
  const [showReturnMap, setShowReturnMap] = useState(false);
  const [gpuAvailable, setGpuAvailable] = useState(false);
  const [copied, setCopied] = useState(false);
  const copiedTimeoutRef = useRef<number | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sidebarRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef(0);
  const attractorStatesRef = useRef<AttractorRuntime[]>([]);
  const rotRef = useRef({ x: initialDef.defaultView.rx, y: initialDef.defaultView.ry });
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const zoomRef = useRef(1);
  const gpuRef = useRef<AttractorGPU | null>(null);
  const poincarePanelRef = useRef<HTMLCanvasElement>(null);
  const returnMapPanelRef = useRef<HTMLCanvasElement>(null);

  const { isFullscreen, toggleFullscreen } = useFullscreen(containerRef);
  const { shareUrl } = useShareUrl();

  const pRef = useRef<Params>({
    attractorTypeId,
    instances,
    selectedInstanceId,
    dt,
    speed,
    trailLength,
    colorScheme,
    running,
    showAxes,
    autoRotate,
    showPoincare,
    showReturnMap,
    poincareZ,
    sectionAxis,
  });

  useEffect(() => {
    pRef.current = {
      attractorTypeId,
      instances,
      selectedInstanceId,
      dt,
      speed,
      trailLength,
      colorScheme,
      running,
      showAxes,
      autoRotate,
      showPoincare,
      showReturnMap,
      poincareZ,
      sectionAxis,
    };
  }, [attractorTypeId, instances, selectedInstanceId, dt, speed, trailLength, colorScheme, running, showAxes, autoRotate, showPoincare, showReturnMap, poincareZ, sectionAxis]);

  const currentDef = ATTRACTOR_CATALOGUE.find(a => a.id === attractorTypeId)!;

  const reset = useCallback(() => {
    const { attractorTypeId: typeId, instances: insts } = pRef.current;
    const def = ATTRACTOR_CATALOGUE.find(a => a.id === typeId)!;
    attractorStatesRef.current.forEach((s) => {
      const inst = insts.find(a => a.id === s.id);
      const pv = def.params.map(p => inst?.paramValues[p.key] ?? p.default);
      s.pos = def.id === 'lorenz' ? [0.1, 0, 0] as MutVec3
        : warmupOrbit(def, pv);
      s.rb = makeRing();
      s.poincare = makePoincare();
    });
    gpuRef.current?.reset();
  }, []);

  // ─── Keyboard shortcuts ───────────────────────────────────────────────────

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.code === 'Space') { e.preventDefault(); setRunning(r => !r); }
      if (e.code === 'KeyR')  { e.preventDefault(); reset(); }
      if (e.code === 'KeyF')  { e.preventDefault(); toggleFullscreen(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [reset, toggleFullscreen]);

  const handleShare = useCallback(() => {
    shareUrl({ a: attractorTypeId });
    setCopied(true);
    if (copiedTimeoutRef.current !== null) window.clearTimeout(copiedTimeoutRef.current);
    copiedTimeoutRef.current = window.setTimeout(() => setCopied(false), 2000);
  }, [shareUrl, attractorTypeId]);

  useEffect(() => () => {
    if (copiedTimeoutRef.current !== null) window.clearTimeout(copiedTimeoutRef.current);
  }, []);

  const snapTo = useCallback((rx: number, ry: number) => {
    rotRef.current = { x: rx, y: ry };
  }, []);

  const addInstance = useCallback(() => {
    const id = `i${Date.now()}`;
    const def = ATTRACTOR_CATALOGUE.find(a => a.id === pRef.current.attractorTypeId)!;
    const selInst = pRef.current.instances.find(a => a.id === pRef.current.selectedInstanceId);
    const paramValues = selInst
      ? { ...selInst.paramValues }
      : Object.fromEntries(def.params.map(p => [p.key, p.default]));
    const pv = def.params.map(p => paramValues[p.key] ?? p.default);
    const newInst = makeInstance(id, pRef.current.instances.length, paramValues);
    attractorStatesRef.current.push({
      id,
      pos: def.id === 'lorenz' ? [0.1, 0, 0] as MutVec3 : warmupOrbit(def, pv),
      rb: makeRing(),
      poincare: makePoincare(),
    });
    setInstances(prev => [...prev, newInst]);
    setSelectedInstanceId(id);
  }, []);

  const removeInstance = useCallback((id: string) => {
    setInstances(prev => {
      const next = prev.filter(inst => inst.id !== id);
      if (pRef.current.selectedInstanceId === id) {
        setSelectedInstanceId(next[next.length - 1]?.id ?? '');
      }
      return next;
    });
    attractorStatesRef.current = attractorStatesRef.current.filter(s => s.id !== id);
  }, []);

  const updateInstanceParam = useCallback((id: string, key: string, value: number) => {
    setInstances(prev => prev.map(inst =>
      inst.id === id ? { ...inst, paramValues: { ...inst.paramValues, [key]: value } } : inst
    ));
  }, []);

  const handleTypeChange = useCallback((newTypeId: string) => {
    const def = ATTRACTOR_CATALOGUE.find(a => a.id === newTypeId)!;
    const defaultPV = newTypeId === 'lorenz'
      ? { ...DEFAULT_LORENZ_PARAMS }
      : Object.fromEntries(def.params.map(p => [p.key, p.default]));
    const first = makeInstance('i0', 0, defaultPV);
    setAttractorTypeId(newTypeId);
    setInstances([first]);
    setSelectedInstanceId('i0');
    setDt(def.dt);
    setSectionAxis(def.defaultSectionAxis);
    setPoincareZ(def.defaultSectionVal);
    setShowAxes(false);
    rotRef.current = { x: def.defaultView.rx, y: def.defaultView.ry };
    zoomRef.current = 1;
    attractorStatesRef.current = buildInstanceStates(def, [first]);
    const gpu = gpuRef.current;
    if (gpu) {
      gpu.reinit(def.gpuType, def.params.map(p => p.default), def.fn, def.dt, [...def.initPos] as [number, number, number]);
    }
  }, []);

  useEffect(() => {
    attractorStatesRef.current = buildInstanceStates(initialDef, instances);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const available = detectWebGL2();
    setGpuAvailable(available);
    if (!available) return;
    try {
      const def = initialDef;
      const gpu = new AttractorGPU(
        def.gpuType,
        def.params.map(p => p.default),
        def.fn,
        def.dt,
        [...def.initPos] as [number, number, number],
      );
      gpuRef.current = gpu;
    } catch (err) {
      console.warn('AttractorGPU init failed:', err);
      setGpuAvailable(false);
    }
    return () => {
      gpuRef.current?.dispose();
      gpuRef.current = null;
    };
  }, []);

  useEffect(() => {
    attractorStatesRef.current.forEach(state => {
      state.poincare = makePoincare();
    });
    gpuRef.current?.reset();
  }, [poincareZ, sectionAxis]);

  useEffect(() => {
    const gpu = gpuRef.current;
    if (!gpu) return;
    const def = ATTRACTOR_CATALOGUE.find(a => a.id === attractorTypeId)!;
    const selInst = instances.find(a => a.id === selectedInstanceId) ?? instances[0];
    const gpuParams = def.params.map(p => selInst?.paramValues[p.key] ?? p.default);
    gpu.reinit(def.gpuType, gpuParams, def.fn, dt, [...def.initPos] as [number, number, number]);
  }, [attractorTypeId, instances, selectedInstanceId, dt]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      rafRef.current = requestAnimationFrame(draw);
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      rafRef.current = requestAnimationFrame(draw);
      return;
    }

    const W = canvas.width;
    const H = canvas.height;
    const {
      attractorTypeId,
      instances,
      selectedInstanceId,
      dt,
      speed,
      trailLength,
      colorScheme,
      running,
      showAxes,
      autoRotate,
      showPoincare,
      showReturnMap,
      poincareZ,
      sectionAxis,
    } = pRef.current;
    const states = attractorStatesRef.current;
    const def = ATTRACTOR_CATALOGUE.find(a => a.id === attractorTypeId)!;

    if (autoRotate && !dragRef.current) {
      rotRef.current.y += 0.004;
    }

    const { x: rotX, y: rotY } = rotRef.current;

    if (running) {
      for (const state of states) {
        const inst = instances.find(a => a.id === state.id) ?? instances[0];
        const params = def.params.map(p => inst?.paramValues[p.key] ?? p.default);

        let [px, py, pz] = state.pos;
        for (let i = 0; i < speed; i++) {
          const [opx, opy, opz] = [px, py, pz];
          [px, py, pz] = rk4(px, py, pz, def.fn, params, dt);
          if (!isFinite(px) || !isFinite(py) || !isFinite(pz)) {
            [px, py, pz] = [...def.initPos] as MutVec3;
            state.rb = makeRing();
            state.poincare = makePoincare();
            break;
          }
          if (showPoincare) {
            if (sectionAxis === 'z' && opz < poincareZ && pz >= poincareZ) {
              const f = (poincareZ - opz) / (pz - opz);
              poincarePush(state.poincare, opx + f * (px - opx), opy + f * (py - opy));
            } else if (sectionAxis === 'y' && opy < poincareZ && py >= poincareZ) {
              const f = (poincareZ - opy) / (py - opy);
              poincarePush(state.poincare, opx + f * (px - opx), opz + f * (pz - opz));
            } else if (sectionAxis === 'x' && opx < poincareZ && px >= poincareZ) {
              const f = (poincareZ - opx) / (px - opx);
              poincarePush(state.poincare, opy + f * (py - opy), opz + f * (pz - opz));
            }
          }
          ringPush(state.rb, px, py, pz);
        }
        state.pos = [px, py, pz];
      }
    }

    ctx.fillStyle = '#080812';
    ctx.fillRect(0, 0, W, H);

    const cosY = Math.cos(rotY);
    const sinY = Math.sin(rotY);
    const cosX = Math.cos(rotX);
    const sinX = Math.sin(rotX);

    const proj = (x: number, y: number, z: number): [number, number] => {
      const x1 = x * cosY + z * sinY;
      const z1 = -x * sinY + z * cosY;
      return [x1, y * cosX - z1 * sinX];
    };

    const scale = Math.min(W, H) / def.viewUnits * zoomRef.current;
    const [ox, oy] = proj(def.center[0], def.center[1], def.center[2]);
    const cx = W * 0.5 - ox * scale;
    const cy = H * 0.5 + oy * scale;

    const toS = (wx: number, wy: number, wz: number): [number, number] => {
      const [sx, sy] = proj(wx, wy, wz);
      return [cx + sx * scale, cy - sy * scale];
    };

    if (showAxes) {
      const { x: BX, y: BY, z: BZ, step: STEP } = def.box;

      const faceDepth = (nx: number, ny: number, nz: number): number => {
        const z1 = -nx * sinY + nz * cosY;
        return ny * sinX + z1 * cosX;
      };

      const FACES: { n: [number, number, number]; c: [number, number, number][] }[] = [
        { n: [1, 0, 0], c: [[BX[1], BY[0], BZ[0]], [BX[1], BY[1], BZ[0]], [BX[1], BY[1], BZ[1]], [BX[1], BY[0], BZ[1]]] },
        { n: [-1, 0, 0], c: [[BX[0], BY[1], BZ[0]], [BX[0], BY[0], BZ[0]], [BX[0], BY[0], BZ[1]], [BX[0], BY[1], BZ[1]]] },
        { n: [0, 1, 0], c: [[BX[1], BY[1], BZ[0]], [BX[0], BY[1], BZ[0]], [BX[0], BY[1], BZ[1]], [BX[1], BY[1], BZ[1]]] },
        { n: [0, -1, 0], c: [[BX[0], BY[0], BZ[0]], [BX[1], BY[0], BZ[0]], [BX[1], BY[0], BZ[1]], [BX[0], BY[0], BZ[1]]] },
        { n: [0, 0, 1], c: [[BX[0], BY[0], BZ[1]], [BX[1], BY[0], BZ[1]], [BX[1], BY[1], BZ[1]], [BX[0], BY[1], BZ[1]]] },
        { n: [0, 0, -1], c: [[BX[0], BY[0], BZ[0]], [BX[1], BY[0], BZ[0]], [BX[1], BY[1], BZ[0]], [BX[0], BY[1], BZ[0]]] },
      ];

      ctx.save();
      ctx.lineJoin = 'round';

      for (const face of FACES) {
        if (faceDepth(face.n[0], face.n[1], face.n[2]) >= 0) continue;

        const sc = face.c.map(p => toS(p[0], p[1], p[2]));

        ctx.beginPath();
        ctx.moveTo(sc[0][0], sc[0][1]);
        for (let i = 1; i < sc.length; i++) ctx.lineTo(sc[i][0], sc[i][1]);
        ctx.closePath();
        ctx.fillStyle = 'rgba(110,120,160,0.06)';
        ctx.fill();

        ctx.strokeStyle = 'rgba(160,170,220,0.28)';
        ctx.lineWidth = 1;
        ctx.setLineDash([]);
        ctx.stroke();

        ctx.strokeStyle = 'rgba(160,170,220,0.1)';
        ctx.lineWidth = 0.5;
        const [nx, ny] = face.n;

        if (Math.abs(face.n[2]) > 0.5) {
          const fz = face.n[2] > 0 ? BZ[1] : BZ[0];
          for (let gx = Math.ceil(BX[0] / STEP) * STEP; gx <= BX[1]; gx += STEP) {
            const a = toS(gx, BY[0], fz);
            const b = toS(gx, BY[1], fz);
            ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
          }
          for (let gy = Math.ceil(BY[0] / STEP) * STEP; gy <= BY[1]; gy += STEP) {
            const a = toS(BX[0], gy, fz);
            const b = toS(BX[1], gy, fz);
            ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
          }
        } else if (Math.abs(nx) > 0.5) {
          const fx = nx > 0 ? BX[1] : BX[0];
          for (let gy = Math.ceil(BY[0] / STEP) * STEP; gy <= BY[1]; gy += STEP) {
            const a = toS(fx, gy, BZ[0]);
            const b = toS(fx, gy, BZ[1]);
            ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
          }
          for (let gz = Math.ceil(BZ[0] / STEP) * STEP; gz <= BZ[1]; gz += STEP) {
            const a = toS(fx, BY[0], gz);
            const b = toS(fx, BY[1], gz);
            ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
          }
        } else {
          const fy = ny > 0 ? BY[1] : BY[0];
          for (let gx = Math.ceil(BX[0] / STEP) * STEP; gx <= BX[1]; gx += STEP) {
            const a = toS(gx, fy, BZ[0]);
            const b = toS(gx, fy, BZ[1]);
            ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
          }
          for (let gz = Math.ceil(BZ[0] / STEP) * STEP; gz <= BZ[1]; gz += STEP) {
            const a = toS(BX[0], fy, gz);
            const b = toS(BX[1], fy, gz);
            ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
          }
        }
      }

      const negYBack = faceDepth(0, -1, 0) < 0;
      const negXBack = faceDepth(-1, 0, 0) < 0;
      const posXBack = faceDepth(1, 0, 0) < 0;
      const OFS = STEP * 0.4;

      const xEdgeY = negYBack ? BY[0] : BY[1];
      const yEdgeX = negXBack ? BX[0] : BX[1];
      const zEdgeX = posXBack ? BX[1] : BX[0];
      const zEdgeY = negYBack ? BY[0] : BY[1];

      const fs = Math.max(14, scale * 1.8);
      const fsLbl = Math.max(18, scale * 2.2);

      ctx.font = `${fs}px var(--font-sans, system-ui)`;
      ctx.fillStyle = 'rgba(180,190,230,0.75)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      for (let x = Math.ceil(BX[0] / STEP) * STEP; x <= BX[1]; x += STEP) {
        const p = toS(x, xEdgeY + (negYBack ? -OFS : OFS), BZ[0]);
        ctx.fillText(formatAxisValue(x), p[0], p[1]);
      }
      for (let y = Math.ceil(BY[0] / STEP) * STEP; y <= BY[1]; y += STEP) {
        const p = toS(yEdgeX + (negXBack ? -OFS : OFS), y, BZ[0]);
        ctx.fillText(formatAxisValue(y), p[0], p[1]);
      }
      for (let z = Math.ceil(BZ[0] / STEP) * STEP; z <= BZ[1]; z += STEP) {
        const p = toS(zEdgeX + (posXBack ? OFS : -OFS), zEdgeY + (negYBack ? -OFS * 0.5 : OFS * 0.5), z);
        ctx.fillText(formatAxisValue(z), p[0], p[1]);
      }

      ctx.font = `bold ${fsLbl}px var(--font-sans, system-ui)`;
      ctx.fillStyle = 'rgba(210,220,250,0.9)';
      const xLbl = toS((BX[0] + BX[1]) * 0.5, xEdgeY + (negYBack ? -OFS * 4 : OFS * 4), BZ[0]);
      ctx.fillText('X Axis', xLbl[0], xLbl[1]);
      const yLbl = toS(yEdgeX + (negXBack ? -OFS * 4 : OFS * 4), (BY[0] + BY[1]) * 0.5, BZ[0]);
      ctx.fillText('Y Axis', yLbl[0], yLbl[1]);
      const zLbl = toS(zEdgeX + (posXBack ? OFS * 3.5 : -OFS * 3.5), zEdgeY + (negYBack ? -OFS * 1.5 : OFS * 1.5), (BZ[0] + BZ[1]) * 0.5);
      ctx.fillText('Z Axis', zLbl[0], zLbl[1]);

      ctx.restore();
    }

    if (showPoincare) {
      ctx.save();
      const { x: BX, y: BY, z: BZ } = def.box;
      const pc = sectionAxis === 'z'
        ? [
          toS(BX[0], BY[0], poincareZ), toS(BX[1], BY[0], poincareZ),
          toS(BX[1], BY[1], poincareZ), toS(BX[0], BY[1], poincareZ),
        ]
        : sectionAxis === 'y'
          ? [
            toS(BX[0], poincareZ, BZ[0]), toS(BX[1], poincareZ, BZ[0]),
            toS(BX[1], poincareZ, BZ[1]), toS(BX[0], poincareZ, BZ[1]),
          ]
          : [
            toS(poincareZ, BY[0], BZ[0]), toS(poincareZ, BY[1], BZ[0]),
            toS(poincareZ, BY[1], BZ[1]), toS(poincareZ, BY[0], BZ[1]),
          ];
      ctx.beginPath();
      ctx.moveTo(pc[0][0], pc[0][1]);
      for (let i = 1; i < pc.length; i++) ctx.lineTo(pc[i][0], pc[i][1]);
      ctx.closePath();
      ctx.fillStyle = 'rgba(129,140,248,0.07)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(129,140,248,0.45)';
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.stroke();
      const labelPt = pc[1];
      ctx.setLineDash([]);
      ctx.font = `${Math.max(11, scale * 1.6)}px var(--font-sans, system-ui)`;
      ctx.fillStyle = 'rgba(129,140,248,0.7)';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(`${sectionAxis} = ${formatAxisValue(poincareZ)}`, labelPt[0] + 6, labelPt[1]);
      ctx.restore();
    }

    const isVelocity = colorScheme === 'velocity';

    ctx.save();
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    for (let ai = 0; ai < states.length; ai++) {
      const state = states[ai];
      const rb = state.rb;
      const visible = Math.min(rb.total, trailLength, MAX_BUF);
      if (visible < 2) continue;

      const BATCH = Math.max(10, Math.ceil(visible / 150));

      for (let b = 0; b < visible - 1; b += BATCH) {
        const bEnd = Math.min(b + BATCH, visible - 1);
        const t = (b + bEnd) * 0.5 / visible;

        let r: number;
        let g: number;
        let bl: number;
        if (isVelocity) {
          const midI = Math.floor((b + bEnd) / 2);
          const [mx, my, mz] = ringGet(rb, midI, visible);
          const [nx, ny, nz] = ringGet(rb, Math.min(midI + 1, visible - 1), visible);
          const spd = Math.sqrt((nx - mx) ** 2 + (ny - my) ** 2 + (nz - mz) ** 2) / dt;
          const raw = Math.min(1, Math.log1p(spd) / Math.log1p(def.maxSpeed));
          const vt = Math.pow(raw, 3);
          [r, g, bl] = COLORS.velocity(vt);
        } else if (ai === 0) {
          [r, g, bl] = COLORS[colorScheme](t);
        } else {
          const inst = instances.find(a => a.id === state.id);
          const [pr, pg, pb] = (inst?.color ?? def.accentRgb).split(',').map(Number);
          r = c(pr * (0.15 + 0.85 * t));
          g = c(pg * (0.15 + 0.85 * t));
          bl = c(pb * (0.15 + 0.85 * t));
        }

        ctx.globalAlpha = 0.15 + 0.85 * t;
        ctx.strokeStyle = `rgb(${r},${g},${bl})`;
        ctx.lineWidth = 0.4 + 1.2 * t;

        ctx.beginPath();
        for (let i = b; i <= bEnd; i++) {
          const [x, y, z] = ringGet(rb, i, visible);
          const [sx, sy] = proj(x, y, z);
          if (i === b) ctx.moveTo(cx + sx * scale, cy - sy * scale);
          else ctx.lineTo(cx + sx * scale, cy - sy * scale);
        }
        ctx.stroke();
      }
    }

    ctx.restore();

    const glowR = Math.max(6, Math.min(W, H) * 0.013);
    for (let ai = 0; ai < states.length; ai++) {
      const state = states[ai];
      const [hx3, hy3, hz3] = state.pos;
      const [hsx, hsy] = proj(hx3, hy3, hz3);
      const hx = cx + hsx * scale;
      const hy = cy - hsy * scale;
      const r = ai === 0 ? glowR : glowR * 0.8;
      const inst = instances.find(a => a.id === state.id);
      const color = inst?.color ?? def.accentRgb;

      const grd = ctx.createRadialGradient(hx, hy, 0, hx, hy, r);
      grd.addColorStop(0, 'rgba(255,255,255,0.95)');
      grd.addColorStop(0.3, `rgba(${color},0.6)`);
      grd.addColorStop(1, `rgba(${color},0)`);
      ctx.fillStyle = grd;
      ctx.beginPath();
      ctx.arc(hx, hy, r, 0, Math.PI * 2);
      ctx.fill();
    }

    if (showPoincare) {
      ctx.save();
      for (let ai = 0; ai < states.length; ai++) {
        const state = states[ai];
        const pb = state.poincare;
        const count = Math.min(pb.total, MAX_POINCARE);
        if (count === 0) continue;
        const pInst = instances.find(a => a.id === state.id);
        const color = pInst?.color ?? def.accentRgb;
        ctx.fillStyle = `rgba(${color},0.9)`;
        for (let i = 0; i < count; i++) {
          const si = ((pb.head - count + i) % MAX_POINCARE + MAX_POINCARE) % MAX_POINCARE;
          const u = pb.xy[si * 2];
          const v = pb.xy[si * 2 + 1];
          const [wx, wy, wz] = sectionAxis === 'z'
            ? [u, v, poincareZ]
            : sectionAxis === 'y'
              ? [u, poincareZ, v]
              : [poincareZ, u, v];
          const [sx, sy] = toS(wx, wy, wz);
          ctx.beginPath();
          ctx.arc(sx, sy, 2, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.restore();
    }

    if (colorScheme === 'velocity') {
      const barW = Math.round(W * 0.28);
      const barH = Math.round(H * 0.018);
      const barX = Math.round((W - barW) / 2);
      const barY = H - Math.round(H * 0.055);
      const fs = Math.max(10, Math.round(H * 0.022));

      const grad = ctx.createLinearGradient(barX, 0, barX + barW, 0);
      for (let i = 0; i <= 20; i++) {
        const t = i / 20;
        const [r, g, b] = COLORS.velocity(t);
        grad.addColorStop(t, `rgb(${r},${g},${b})`);
      }

      ctx.save();
      ctx.globalAlpha = 0.85;
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.roundRect(barX, barY, barW, barH, barH / 2);
      ctx.fill();

      ctx.globalAlpha = 1;
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.font = `${fs}px sans-serif`;
      ctx.textBaseline = 'bottom';
      ctx.textAlign = 'left';
      ctx.fillText('slow', barX, barY - 3);
      ctx.textAlign = 'right';
      ctx.fillText('fast', barX + barW, barY - 3);
      ctx.restore();
    }

    const gpu = gpuRef.current;
    if (gpu && running && (showPoincare || showReturnMap)) {
      const selInst = instances.find(a => a.id === selectedInstanceId) ?? instances[0];
      const gpuParams: AttractorGPUParams = {
        type: def.gpuType,
        params: def.params.map(p => selInst?.paramValues[p.key] ?? p.default),
        dt,
        sectionAxis,
        sectionVal: poincareZ,
        sectionBounds: def.sectionBounds[sectionAxis],
        returnBounds: def.returnBounds,
      };
      gpu.step(gpuParams);
    }

    if (showPoincare) {
      const pc = poincarePanelRef.current;
      if (pc && pc.clientWidth > 0) {
        const panelDpr = Math.min(window.devicePixelRatio || 1, 2);
        const pw = Math.round(pc.clientWidth * panelDpr);
        const ph = Math.round(pc.clientHeight * panelDpr);
        if (pc.width !== pw || pc.height !== ph) {
          pc.width = pw;
          pc.height = ph;
        }
        const pCtx = pc.getContext('2d');
        if (pCtx) {
          if (gpu) {
            gpu.drawPanel(pCtx, 0, 0, 0, pc.width, pc.height);
          } else {
            let minU = Infinity;
            let maxU = -Infinity;
            let minV = Infinity;
            let maxV = -Infinity;
            for (const st of states) {
              if (st.poincare.total === 0) continue;
              minU = Math.min(minU, st.poincare.minU);
              maxU = Math.max(maxU, st.poincare.maxU);
              minV = Math.min(minV, st.poincare.minV);
              maxV = Math.max(maxV, st.poincare.maxV);
            }
            const fallback = def.sectionBounds[sectionAxis];
            const hasData = isFinite(minU) && isFinite(maxU);
            const rangeU = hasData ? Math.max((maxU - minU) * 1.15, 1) : (fallback[1] - fallback[0]);
            const rangeV = hasData ? Math.max((maxV - minV) * 1.15, 1) : (fallback[3] - fallback[2]);
            const cU = hasData ? (minU + maxU) / 2 : (fallback[0] + fallback[1]) * 0.5;
            const cV = hasData ? (minV + maxV) / 2 : (fallback[2] + fallback[3]) * 0.5;
            const half = Math.min(pc.width, pc.height) * 0.5;
            const pcx = pc.width / 2;
            const pcy = pc.height / 2;

            pCtx.fillStyle = '#05050f';
            pCtx.fillRect(0, 0, pc.width, pc.height);

            const axX = Math.max(0, Math.min(pc.width, pcx + (0 - cU) / (rangeU / 2) * half));
            const axY = Math.max(0, Math.min(pc.height, pcy - (0 - cV) / (rangeV / 2) * half));
            pCtx.strokeStyle = 'rgba(255,255,255,0.15)';
            pCtx.lineWidth = 1;
            pCtx.beginPath(); pCtx.moveTo(0, axY); pCtx.lineTo(pc.width, axY); pCtx.stroke();
            pCtx.beginPath(); pCtx.moveTo(axX, 0); pCtx.lineTo(axX, pc.height); pCtx.stroke();

            if (!hasData) {
              pCtx.fillStyle = 'rgba(180,190,230,0.65)';
              pCtx.font = `${Math.round(Math.min(pc.width, pc.height) * 0.07)}px system-ui, sans-serif`;
              pCtx.textAlign = 'center';
              pCtx.textBaseline = 'middle';
              pCtx.fillText('waiting for crossings…', pcx, pcy);
            } else {
              for (const state of states) {
                const pb = state.poincare;
                const count = Math.min(pb.total, MAX_POINCARE);
                if (count === 0) continue;
                const cpInst = instances.find(a => a.id === state.id);
                const color = cpInst?.color ?? def.accentRgb;
                pCtx.fillStyle = `rgba(${color},0.9)`;
                for (let i = 0; i < count; i++) {
                  const si = ((pb.head - count + i) % MAX_POINCARE + MAX_POINCARE) % MAX_POINCARE;
                  const sx = pcx + (pb.xy[si * 2] - cU) / (rangeU / 2) * half;
                  const sy = pcy - (pb.xy[si * 2 + 1] - cV) / (rangeV / 2) * half;
                  if (sx < -2 || sx > pc.width + 2 || sy < -2 || sy > pc.height + 2) continue;
                  pCtx.fillRect(sx - 1.5, sy - 1.5, 3, 3);
                }
              }
            }
          }
        }
      }
    }

    if (showReturnMap && gpu) {
      const rc = returnMapPanelRef.current;
      if (rc && rc.clientWidth > 0) {
        const panelDpr = Math.min(window.devicePixelRatio || 1, 2);
        const rw = Math.round(rc.clientWidth * panelDpr);
        const rh = Math.round(rc.clientHeight * panelDpr);
        if (rc.width !== rw || rc.height !== rh) {
          rc.width = rw;
          rc.height = rh;
        }
        const rCtx = rc.getContext('2d');
        if (rCtx) gpu.drawPanel(rCtx, 1, 0, 0, rc.width, rc.height);
      }
    }

    rafRef.current = requestAnimationFrame(draw);
  }, [currentDef]);

  useEffect(() => {
    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [draw]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      canvas.width = (rect.width * dpr) | 0;
      canvas.height = (rect.height * dpr) | 0;
    });
    ro.observe(canvas);
    return () => ro.disconnect();
  }, []);

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
      const cCanvas = canvasRef.current;
      if (!cCanvas) return;
      if (sidebarRef.current?.contains(e.target as Node)) return;
      const r = cCanvas.getBoundingClientRect();
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

  const [pcU, pcV] = sectionAxis === 'z' ? ['x', 'y'] : sectionAxis === 'y' ? ['x', 'z'] : ['y', 'z'];
  const sectionRange = currentDef.box[sectionAxis];
  const sectionStep = Math.max(0.001, Math.min(1, currentDef.box.step / 10));

  return (
    <div ref={containerRef} className={styles.container}>
      <canvas ref={canvasRef} className={styles.canvas} />

      <div ref={sidebarRef} className={styles.sidebar}>
        <div className={styles.sidebarPanels}>
        <ControlPanel title="Attractor">
          <ControlGroup>
            <SelectControl
              label="Type"
              value={attractorTypeId}
              onChange={handleTypeChange}
              options={ATTRACTOR_CATALOGUE.map(a => ({ value: a.id, label: a.name }))}
            />
          </ControlGroup>

          <div
            className={styles.equationBlock}
            style={{ '--eq-accent': currentDef.accentColor } as CSSProperties}
            title={currentDef.description}
          >
            {currentDef.equations.map((eq, i) => (
              <span key={i} className={styles.equationLine}>{eq}</span>
            ))}
          </div>

          <div className={styles.instanceTabs}>
            {instances.map((inst, idx) => {
              const isActive = inst.id === selectedInstanceId;
              return (
                <button
                  key={inst.id}
                  type="button"
                  className={`${styles.instanceTab} ${isActive ? styles.instanceTabActive : ''}`}
                  style={{ '--tab-color': `rgb(${inst.color})` } as CSSProperties}
                  onClick={() => setSelectedInstanceId(inst.id)}
                >
                  <span className={styles.instanceTabDot} style={{ background: `rgb(${inst.color})` }} />
                  {idx + 1}
                  {instances.length > 1 && (
                    <span
                      className={styles.instanceTabRemove}
                      role="button"
                      aria-label="Remove"
                      onClick={e => { e.stopPropagation(); removeInstance(inst.id); }}
                    >✕</span>
                  )}
                </button>
              );
            })}
            <button
              type="button"
              className={styles.instanceTabAdd}
              onClick={addInstance}
              title="Add attractor"
            >+</button>
          </div>

          {(() => {
            const selInst = instances.find(a => a.id === selectedInstanceId) ?? instances[0];
            if (!selInst || currentDef.params.length === 0) return null;
            return (
              <ControlGroup>
                {currentDef.params.map(param => (
                  <Slider
                    key={`${selInst.id}-${param.key}`}
                    label={param.label}
                    value={selInst.paramValues[param.key] ?? param.default}
                    onChange={v => updateInstanceParam(selInst.id, param.key, v)}
                    min={param.min}
                    max={param.max}
                    step={param.step}
                    format={v => v.toFixed(decimalsForStep(param.step))}
                    manualInput
                  />
                ))}
              </ControlGroup>
            );
          })()}
        </ControlPanel>

        <ControlPanel title="Animation">
          <ControlGroup>
            <Toggle
              label="Auto-rotate"
              value={autoRotate}
              onChange={setAutoRotate}
              description="Slowly spin the view"
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
            <Slider
              label="dt"
              value={dt}
              onChange={setDt}
              min={0.0001}
              max={0.1}
              step={0.0001}
              format={v => v.toFixed(4)}
            />
            <Slider
              label="Trail length"
              value={trailLength}
              onChange={setTrailLength}
              min={100}
              max={10000}
              step={100}
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
                { value: 'lorenz' as const, label: 'Lorenz (indigo)' },
                { value: 'heat' as const, label: 'Heat map' },
                { value: 'plasma' as const, label: 'Plasma' },
                { value: 'neon' as const, label: 'Neon green' },
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


        </div>{/* end sidebarPanels */}

        <div className={styles.sidebarActions}>
        <SimControls
          running={running}
          onToggle={() => setRunning(r => !r)}
          onReset={reset}
          onExport={() => setShowExport(true)}
        />
        </div>
      </div>

      {showExport && (
        <ExportDialog
          onClose={() => setShowExport(false)}
          onDownload={({ width, height, format }) => {
            const canvas = canvasRef.current;
            if (!canvas) return;
            exportImage(canvas, width, height, format, 'lorenz');
            setShowExport(false);
          }}
        />
      )}

      <div className={styles.hud}>
        <div className={styles.hudLeft}>
          <span className={styles.hudTitle} style={{ color: currentDef.accentColor }}>
            {currentDef.name} Attractor
          </span>
        </div>
        <div className={styles.hudRight}>
          <span className={styles.hudHint}>drag to rotate</span>
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
          <button className={styles.hudInfoBtn} onClick={() => setShowInfo(true)} title="About strange attractors">ⓘ</button>
        </div>
      </div>

      <div className={styles.panelStack}>
        {gpuAvailable && (
          <div className={`${styles.analysisPanel} ${styles.analysisPanelReturn}`}>
            <div className={`${styles.panelHeader} ${!showReturnMap ? styles.panelHeaderCollapsed : ''}`}>
              <span className={`${styles.panelTitle} ${styles.panelTitleReturn}`}>
                Return map: z<sub>n+1</sub> vs z<sub>n</sub>
              </span>
              <div className={styles.infoBtnWrapper}>
                <button className={styles.infoBtn} type="button" aria-label="About return map"><Info size={20} strokeWidth={1.5} /></button>
                <div className={styles.infoTooltip}>
                  Each point (z<sub>n</sub>,&thinsp;z<sub>n+1</sub>) plots successive local z&#8209;maxima
                  against each other. The tent&#8209;map shape confirms deterministic chaos:
                  tiny differences grow exponentially.
                </div>
              </div>
              <button
                className={styles.panelToggleBtn}
                type="button"
                aria-label={showReturnMap ? 'Collapse return map' : 'Expand return map'}
                onClick={() => setShowReturnMap(v => !v)}
              >
                {showReturnMap ? '▾' : '▸'}
              </button>
            </div>
            {showReturnMap && (
              <div className={styles.plotWrapper}>
                <canvas ref={returnMapPanelRef} className={styles.plotCanvas} />
                <span className={`${styles.axisLabel} ${styles.axisLabelH} ${styles.axisLabelReturn}`}>z<sub>n</sub>&thinsp;→</span>
                <span className={`${styles.axisLabel} ${styles.axisLabelV} ${styles.axisLabelReturn}`}>↑&thinsp;z<sub>n+1</sub></span>
                <span className={`${styles.panelBadge} ${styles.panelBadgeReturn}`}>GPU</span>
              </div>
            )}
          </div>
        )}
        <div className={styles.analysisPanel}>
          <div className={`${styles.panelHeader} ${!showPoincare ? styles.panelHeaderCollapsed : ''}`}>
            <span className={styles.panelTitle}>
              {showPoincare
                ? <>Poincaré: {sectionAxis}&thinsp;=&thinsp;{formatAxisValue(poincareZ)}</>
                : 'Poincaré section'}
            </span>
            <div className={styles.infoBtnWrapper}>
              <button className={styles.infoBtn} type="button" aria-label="About Poincaré section"><Info size={20} strokeWidth={1.5} /></button>
              <div className={styles.infoTooltip}>
                Records each crossing of the {sectionAxis}&thinsp;=&thinsp;{formatAxisValue(poincareZ)} plane.
                The crossing points form a fractal curve, revealing the attractor&rsquo;s
                self&#8209;similar structure at every scale.
              </div>
            </div>
            <button
              className={styles.panelToggleBtn}
              type="button"
              aria-label={showPoincare ? 'Collapse Poincaré section' : 'Expand Poincaré section'}
              onClick={() => setShowPoincare(v => !v)}
            >
              {showPoincare ? '▾' : '▸'}
            </button>
          </div>
          {showPoincare && (
            <>
              <div className={styles.panelControls}>
                <SelectControl
                  label="Axis"
                  value={sectionAxis}
                  onChange={setSectionAxis}
                  options={[
                    { value: 'y' as const, label: 'y = const  (shows x, z)' },
                    { value: 'z' as const, label: 'z = const  (shows x, y)' },
                    { value: 'x' as const, label: 'x = const  (shows y, z)' },
                  ]}
                />
                <Slider
                  label={`${sectionAxis} =`}
                  value={poincareZ}
                  onChange={setPoincareZ}
                  min={sectionRange[0]}
                  max={sectionRange[1]}
                  step={sectionStep}
                  format={v => v.toFixed(decimalsForStep(sectionStep))}
                />
              </div>
              <div className={styles.plotWrapper}>
                <canvas ref={poincarePanelRef} className={styles.plotCanvas} />
                <span className={`${styles.axisLabel} ${styles.axisLabelH}`}>{pcU}&thinsp;→</span>
                <span className={`${styles.axisLabel} ${styles.axisLabelV}`}>↑&thinsp;{pcV}</span>
                {gpuAvailable && <span className={styles.panelBadge}>GPU</span>}
              </div>
            </>
          )}
        </div>
      </div>

      {showInfo && (
        <InfoDialog title="Strange Attractors" onClose={() => setShowInfo(false)}>
          <p>
            A strange attractor is a fractal structure in phase space that a chaotic system
            gravitates toward. Trajectories stay bounded but never repeat the same path.
          </p>
          <h3>Lorenz attractor</h3>
          <p>
            Edward Lorenz derived this in 1963 from a simplified weather model. The
            butterfly-shaped orbit was the first strange attractor described and inspired
            the term <strong>butterfly effect</strong>: tiny differences in starting conditions
            grow into completely different outcomes.
          </p>
          <h3>Other attractors</h3>
          <p>
            Rössler, Aizawa, Thomas, and others each have their own differential equations
            but the same core property: bounded, non-repeating orbits on a fractal set.
          </p>
          <h3>Controls</h3>
          <ul>
            <li><strong>Drag:</strong> rotate the 3D view</li>
            <li><strong>+ Attractor:</strong> add a second trajectory to compare divergence</li>
            <li><strong>Poincaré / Return map:</strong> cross-section analysis</li>
          </ul>
        </InfoDialog>
      )}
    </div>
  );
}

