import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Slider, Toggle, SelectControl,
  ControlPanel, ControlGroup,
} from '@/components/Controls';
import {
  detectWebGL,
  createKochRenderer,
  generateKoch,
  type KochWebGLRenderer,
  type KochRenderParams,
  type ColorSchemeId,
  type FillModeId,
} from './koch-webgl';
import styles from './Koch.module.css';

// ─── Types & constants ────────────────────────────────────────────────────────

interface Preset {
  label: string;
  depth: number;
  antiKoch: boolean;
  colorScheme: ColorSchemeId;
  fillMode: FillModeId;
  glow: boolean;
}

const PRESETS: Preset[] = [
  { label: 'Snow',   depth: 5, antiKoch: false, colorScheme: 'frost',  fillMode: 'both',    glow: true  },
  { label: 'Aurora', depth: 4, antiKoch: false, colorScheme: 'aurora', fillMode: 'both',    glow: true  },
  { label: 'Fire',   depth: 4, antiKoch: true,  colorScheme: 'fire',   fillMode: 'filled',  glow: false },
  { label: 'Wire',   depth: 6, antiKoch: false, colorScheme: 'mono',   fillMode: 'outline', glow: true  },
  { label: 'Anti',   depth: 4, antiKoch: true,  colorScheme: 'frost',  fillMode: 'both',    glow: true  },
  { label: 'Deep',   depth: 7, antiKoch: false, colorScheme: 'frost',  fillMode: 'outline', glow: true  },
];

const DEFAULT: Preset = PRESETS[0];

const COLOR_OPTS = [
  { value: 'frost',  label: 'Frost'  },
  { value: 'aurora', label: 'Aurora' },
  { value: 'fire',   label: 'Fire'   },
  { value: 'mono',   label: 'Mono'   },
];

const FILL_OPTS = [
  { value: 'both',    label: 'Fill + Outline' },
  { value: 'filled',  label: 'Fill only'      },
  { value: 'outline', label: 'Outline only'   },
];

const VERT_COUNTS = Array.from({ length: 8 }, (_, d) => 3 * 4 ** d);

// ─── Component ────────────────────────────────────────────────────────────────

export default function Koch() {
  const gpuSupported = detectWebGL();

  // Controls state
  const [depth,       setDepth]       = useState(DEFAULT.depth);
  const [antiKoch,    setAntiKoch]    = useState(DEFAULT.antiKoch);
  const [colorScheme, setColorScheme] = useState<ColorSchemeId>(DEFAULT.colorScheme);
  const [fillMode,    setFillMode]    = useState<FillModeId>(DEFAULT.fillMode);
  const [glow,        setGlow]        = useState(DEFAULT.glow);
  const [useGPU,      setUseGPU]      = useState(gpuSupported);
  const [rotate,      setRotate]      = useState(false);
  const [zoom,        setZoom]        = useState(1);
  const [panX,        setPanX]        = useState(0);
  const [panY,        setPanY]        = useState(0);

  // GPU canvas ref
  const glCanvasRef = useRef<HTMLCanvasElement>(null);
  // CPU canvas ref
  const cpuCanvasRef = useRef<HTMLCanvasElement>(null);
  // Renderer ref
  const glRef = useRef<KochWebGLRenderer | null>(null);
  // Animation
  const rafRef  = useRef<number>(0);
  const rotRef  = useRef(0);
  // Drag state
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  // Active preset tracking
  const [activePreset, setActivePreset] = useState<number>(0);

  // Mirror current params to a ref for use inside RAF without stale closure
  const pRef = useRef<KochRenderParams>({
    depth, antiKoch, colorScheme, fillMode, glow, zoom,
    panX, panY, rotation: 0,
  });

  useEffect(() => {
    pRef.current = { depth, antiKoch, colorScheme, fillMode, glow, zoom, panX, panY, rotation: rotRef.current };
  }, [depth, antiKoch, colorScheme, fillMode, glow, zoom, panX, panY]);

  // ─── GPU init ────────────────────────────────────────────────────────────────

  useEffect(() => {
    if (!useGPU || !glCanvasRef.current) return;
    glRef.current?.dispose();
    glRef.current = createKochRenderer(glCanvasRef.current);
    return () => { glRef.current?.dispose(); glRef.current = null; };
  }, [useGPU]);

  // ─── CPU draw ────────────────────────────────────────────────────────────────

  const drawCPU = useCallback((p: KochRenderParams) => {
    const canvas = cpuCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#070712';
    ctx.fillRect(0, 0, W, H);

    const pts = generateKoch(p.depth, p.antiKoch);
    const scale = 0.88 * p.zoom * Math.min(W, H) / 2;
    const cos = Math.cos(p.rotation), sin = Math.sin(p.rotation);

    const tx = (x: number, y: number) => {
      const rx = x * cos - y * sin, ry = x * sin + y * cos;
      return [W / 2 + rx * scale + p.panX * W / 2, H / 2 - ry * scale - p.panY * H / 2] as [number, number];
    };

    const colors: Record<ColorSchemeId, [string, string]> = {
      frost:  ['#2e93fa', '#e0f0ff'],
      aurora: ['#06e088', '#a614ff'],
      fire:   ['#f11f03', '#fff0b3'],
      mono:   ['#9fb8d0', '#e8f4ff'],
    };
    const [c0, c1] = colors[p.colorScheme];

    ctx.save();
    if (p.fillMode !== 'outline') {
      const grad = ctx.createLinearGradient(0, 0, W, H);
      grad.addColorStop(0, c0 + '55');
      grad.addColorStop(1, c1 + '55');
      ctx.fillStyle = grad;
      ctx.beginPath();
      const [fx, fy] = tx(pts[0][0], pts[0][1]);
      ctx.moveTo(fx, fy);
      for (let i = 1; i < pts.length; i++) {
        const [x, y] = tx(pts[i][0], pts[i][1]);
        ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fill();
    }
    if (p.fillMode !== 'filled') {
      const grad = ctx.createLinearGradient(0, 0, W, H);
      grad.addColorStop(0, c0);
      grad.addColorStop(1, c1);
      ctx.strokeStyle = grad;
      ctx.lineWidth = 1;
      ctx.beginPath();
      const [fx, fy] = tx(pts[0][0], pts[0][1]);
      ctx.moveTo(fx, fy);
      for (let i = 1; i < pts.length; i++) {
        const [x, y] = tx(pts[i][0], pts[i][1]);
        ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.stroke();
    }
    ctx.restore();
  }, []);

  // ─── Animation loop ───────────────────────────────────────────────────────────

  const useGPURef = useRef(useGPU);
  const rotateRef = useRef(rotate);
  useEffect(() => { useGPURef.current = useGPU; }, [useGPU]);
  useEffect(() => { rotateRef.current = rotate; }, [rotate]);

  useEffect(() => {
    let stopped = false;

    function frame() {
      if (stopped) return;
      if (rotateRef.current) {
        rotRef.current += 0.004;
      }
      const p = { ...pRef.current, rotation: rotRef.current };
      if (useGPURef.current) {
        glRef.current?.render(p);
      } else {
        drawCPU(p);
      }
      rafRef.current = requestAnimationFrame(frame);
    }
    rafRef.current = requestAnimationFrame(frame);
    return () => { stopped = true; cancelAnimationFrame(rafRef.current); };
  }, [drawCPU]);

  // ─── Resize observer ──────────────────────────────────────────────────────────

  useEffect(() => {
    const gpuCanvas = glCanvasRef.current;
    const cpuCanvas = cpuCanvasRef.current;
    if (!gpuCanvas || !cpuCanvas) return;

    const obs = new ResizeObserver((entries) => {
      for (const e of entries) {
        const { width, height } = e.contentRect;
        const dpr = window.devicePixelRatio || 1;
        const w = Math.round(width * dpr), h = Math.round(height * dpr);
        gpuCanvas.width = w; gpuCanvas.height = h;
        cpuCanvas.width = w; cpuCanvas.height = h;
      }
    });

    const container = gpuCanvas.parentElement!;
    obs.observe(container);
    return () => obs.disconnect();
  }, []);

  // ─── Interaction ─────────────────────────────────────────────────────────────

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    dragRef.current = { x: e.clientX, y: e.clientY };
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }, []);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (!dragRef.current) return;
    const dx = e.clientX - dragRef.current.x;
    const dy = e.clientY - dragRef.current.y;
    dragRef.current = { x: e.clientX, y: e.clientY };
    const rect = (e.target as HTMLElement).getBoundingClientRect();
    setPanX(px => px + (2 * dx) / rect.width);
    setPanY(py => py - (2 * dy) / rect.height);
    setActivePreset(-1);
  }, []);

  const onPointerUp = useCallback(() => { dragRef.current = null; }, []);

  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    setZoom(z => Math.max(0.1, Math.min(20, z * (1 + delta))));
    setActivePreset(-1);
  }, []);

  const applyPreset = useCallback((idx: number) => {
    const p = PRESETS[idx];
    setDepth(p.depth);
    setAntiKoch(p.antiKoch);
    setColorScheme(p.colorScheme);
    setFillMode(p.fillMode);
    setGlow(p.glow);
    setActivePreset(idx);
    setZoom(1); setPanX(0); setPanY(0);
    rotRef.current = 0;
  }, []);

  const reset = useCallback(() => {
    setZoom(1); setPanX(0); setPanY(0);
    rotRef.current = 0;
  }, []);

  const nVerts = VERT_COUNTS[depth] ?? 0;

  return (
    <div className={styles.container}>
      {/* GPU canvas */}
      <canvas
        ref={glCanvasRef}
        className={styles.canvas}
        style={{ display: useGPU ? 'block' : 'none' }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onWheel={onWheel}
      />
      {/* CPU canvas */}
      <canvas
        ref={cpuCanvasRef}
        className={styles.canvas}
        style={{ display: useGPU ? 'none' : 'block' }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onWheel={onWheel}
      />

      {/* Sidebar */}
      <div className={styles.sidebar}>
        <div className={styles.sidebarPanels}>
          {/* Presets */}
          <ControlPanel title="Presets">
            <div className={styles.presetGrid}>
              {PRESETS.map((p, i) => (
                <button
                  key={p.label}
                  className={[styles.presetBtn, activePreset === i ? styles.presetBtnActive : ''].join(' ')}
                  onClick={() => applyPreset(i)}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </ControlPanel>

          {/* Fractal */}
          <ControlPanel title="Fractal">
            <ControlGroup>
              <Slider
                label="Depth"
                value={depth}
                min={0} max={7} step={1}
                onChange={v => { setDepth(v); setActivePreset(-1); }}
              />
              <Toggle
                label="Anti-Koch"
                value={antiKoch}
                onChange={v => { setAntiKoch(v); setActivePreset(-1); }}
              />
            </ControlGroup>
          </ControlPanel>

          {/* Appearance */}
          <ControlPanel title="Appearance">
            <ControlGroup>
              <SelectControl
                label="Color"
                value={colorScheme}
                options={COLOR_OPTS}
                onChange={v => { setColorScheme(v as ColorSchemeId); setActivePreset(-1); }}
              />
              <SelectControl
                label="Fill"
                value={fillMode}
                options={FILL_OPTS}
                onChange={v => { setFillMode(v as FillModeId); setActivePreset(-1); }}
              />
              <Toggle
                label="Glow"
                value={glow}
                onChange={v => { setGlow(v); setActivePreset(-1); }}
              />
            </ControlGroup>
          </ControlPanel>

          {/* Rendering */}
          <ControlPanel title="Rendering">
            <ControlGroup>
              <Toggle label="Rotate" value={rotate} onChange={setRotate} />
              {gpuSupported && (
                <Toggle label="GPU" value={useGPU} onChange={setUseGPU} />
              )}
            </ControlGroup>
          </ControlPanel>
        </div>

        {/* Actions */}
        <div className={styles.sidebarActions}>
          <button className={styles.resetBtn} onClick={reset}>
            Reset View
          </button>
        </div>
      </div>

      {/* HUD */}
      <div className={styles.hud}>
        <div className={styles.hudLeft}>
          <span className={styles.hudTitle}>Koch {antiKoch ? 'Anti-Snowflake' : 'Snowflake'}</span>
          <span className={styles.hudSub}>
            depth {depth} · {nVerts.toLocaleString()} verts
          </span>
        </div>
        <div className={styles.hudRight}>
          {useGPU && gpuSupported ? (
            <span className={styles.gpuBadge}>⬡ GPU</span>
          ) : (
            <span className={styles.hudHint}>CPU</span>
          )}
          <span className={styles.hudHint}>scroll to zoom · drag to pan</span>
        </div>
      </div>
    </div>
  );
}
