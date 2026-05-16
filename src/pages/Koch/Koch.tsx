import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Slider, Toggle, SelectControl,
  ControlPanel, ControlGroup,
} from '@/components/Controls';
import { InfoDialog } from '@/components/InfoDialog';
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
  sides: number;
  antiKoch: boolean;
  colorScheme: ColorSchemeId;
  fillMode: FillModeId;
  glow: boolean;
}

const PRESETS: Preset[] = [
  { label: 'Snow',  depth: 5, sides: 3, antiKoch: false, colorScheme: 'mono',  fillMode: 'outline', glow: true  },
  { label: 'Frost', depth: 4, sides: 3, antiKoch: false, colorScheme: 'frost', fillMode: 'both',    glow: true  },
  { label: 'Wire',  depth: 6, sides: 3, antiKoch: false, colorScheme: 'mono',  fillMode: 'outline', glow: true  },
  { label: 'Anti',  depth: 4, sides: 3, antiKoch: true,  colorScheme: 'frost', fillMode: 'both',    glow: true  },
  { label: 'Fill',  depth: 5, sides: 3, antiKoch: false, colorScheme: 'frost', fillMode: 'filled',  glow: false },
  { label: 'Deep',  depth: 7, sides: 3, antiKoch: false, colorScheme: 'mono',  fillMode: 'outline', glow: true  },
];

const DEFAULT: Preset = PRESETS[0];

const COLOR_OPTS = [
  { value: 'frost', label: 'Frost' },
  { value: 'mono',  label: 'Mono'  },
];

const FILL_OPTS = [
  { value: 'both',    label: 'Fill + Outline' },
  { value: 'filled',  label: 'Fill only'      },
  { value: 'outline', label: 'Outline only'   },
];

const SIDES_OPTS = [
  { value: '3', label: 'Triangle (3)' },
  { value: '4', label: 'Square (4)'   },
  { value: '5', label: 'Pentagon (5)' },
  { value: '6', label: 'Hexagon (6)'  },
  { value: '7', label: 'Heptagon (7)' },
  { value: '8', label: 'Octagon (8)'  },
];

const SHAPE_NAMES: Record<number, string> = {
  3: 'Triangle', 4: 'Square', 5: 'Pentagon',
  6: 'Hexagon',  7: 'Heptagon', 8: 'Octagon',
};

const VERT_COUNTS = (sides: number, depth: number) => sides * 4 ** depth;

// ─── Component ────────────────────────────────────────────────────────────────

export default function Koch() {
  // detectWebGL() creates a canvas+context — lazy initializer ensures it runs
  // exactly once, not on every re-render (which would leak WebGL contexts).
  const [gpuSupported] = useState(() => detectWebGL());

  // Controls state
  const [depth,       setDepth]       = useState(DEFAULT.depth);
  const [sides,       setSides]       = useState(DEFAULT.sides);
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
  // Container ref (for non-passive wheel listener)
  const containerRef = useRef<HTMLDivElement>(null);
  // Renderer ref
  const glRef = useRef<KochWebGLRenderer | null>(null);
  // Animation
  const rafRef  = useRef<number>(0);
  const rotRef  = useRef(0);
  // Pending canvas size written by ResizeObserver, applied inside RAF so
  // resize + redraw are atomic within one compositor frame (no black flash).
  const pendingSizeRef = useRef<{ w: number; h: number } | null>(null);
  // Drag state
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  // Active preset tracking
  const [activePreset, setActivePreset] = useState<number>(0);
  const [showInfo, setShowInfo] = useState(false);

  // Synchronously mirror state into a ref so the RAF loop always reads the
  // latest values without waiting for a useEffect to fire after paint.
  const pRef = useRef<KochRenderParams>({
    depth, sides, antiKoch, colorScheme, fillMode, glow, zoom,
    panX, panY, rotation: 0,
  });
  pRef.current.depth       = depth;
  pRef.current.sides       = sides;
  pRef.current.antiKoch    = antiKoch;
  pRef.current.colorScheme = colorScheme;
  pRef.current.fillMode    = fillMode;
  pRef.current.glow        = glow;
  pRef.current.zoom        = zoom;
  pRef.current.panX        = panX;
  pRef.current.panY        = panY;
  // rotation is managed by rotRef — updated each frame, not via state

  // ─── GPU init ────────────────────────────────────────────────────────────────

  useEffect(() => {
    if (!useGPU || !glCanvasRef.current) return;
    const canvas = glCanvasRef.current;
    glRef.current?.dispose();
    glRef.current = createKochRenderer(canvas);

    // Recreate the renderer if the browser reclaims GPU resources.
    const onLost = (e: Event) => { e.preventDefault(); glRef.current = null; };
    const onRestored = () => { glRef.current = createKochRenderer(canvas); };
    canvas.addEventListener('webglcontextlost', onLost);
    canvas.addEventListener('webglcontextrestored', onRestored);

    return () => {
      canvas.removeEventListener('webglcontextlost', onLost);
      canvas.removeEventListener('webglcontextrestored', onRestored);
      glRef.current?.dispose();
      glRef.current = null;
    };
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

    const pts = generateKoch(p.depth, p.antiKoch, p.sides);
    const scale = 0.88 * p.zoom * Math.min(W, H) / 2;
    const cos = Math.cos(p.rotation), sin = Math.sin(p.rotation);

    const tx = (x: number, y: number) => {
      const rx = x * cos - y * sin, ry = x * sin + y * cos;
      return [W / 2 + rx * scale + p.panX * W / 2, H / 2 - ry * scale - p.panY * H / 2] as [number, number];
    };

    const colors: Record<ColorSchemeId, [string, string]> = {
      frost: ['#2e93fa', '#e0f0ff'],
      mono:  ['#9fb8d0', '#e8f4ff'],
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

      // Apply any pending canvas resize here, inside the RAF callback.
      // Doing resize + redraw in the same RAF frame means the compositor
      // only ever sees the fully-rendered result — never the cleared canvas.
      const pending = pendingSizeRef.current;
      if (pending) {
        pendingSizeRef.current = null;
        const gpuCanvas = glCanvasRef.current;
        const cpuCanvas = cpuCanvasRef.current;
        if (gpuCanvas && (gpuCanvas.width !== pending.w || gpuCanvas.height !== pending.h)) {
          gpuCanvas.width = pending.w; gpuCanvas.height = pending.h;
        }
        if (cpuCanvas && (cpuCanvas.width !== pending.w || cpuCanvas.height !== pending.h)) {
          cpuCanvas.width = pending.w; cpuCanvas.height = pending.h;
        }
      }

      if (rotateRef.current) {
        rotRef.current += 0.004;
      }
      const p: KochRenderParams = { ...pRef.current, rotation: rotRef.current };
      try {
        if (useGPURef.current) {
          glRef.current?.render(p);
        } else {
          drawCPU(p);
        }
      } catch (err) {
        // Swallow render errors so the loop never dies permanently.
        console.warn('[Koch] render error:', err);
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
        if (gpuCanvas.width !== w || gpuCanvas.height !== h ||
            cpuCanvas.width !== w || cpuCanvas.height !== h) {
          // Store pending size — the RAF loop applies it so that the canvas
          // clear and redraw happen atomically inside one RAF callback.
          pendingSizeRef.current = { w, h };
        }
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

  // Non-passive wheel listener attached imperatively so e.preventDefault()
  // is guaranteed to work (React synthetic onWheel may be passive in some
  // browser/React version combinations, preventing preventDefault).
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const handler = (e: WheelEvent) => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? -0.1 : 0.1;
      setZoom(z => Math.max(0.1, Math.min(20, z * (1 + delta))));
      setActivePreset(-1);
    };
    el.addEventListener('wheel', handler, { passive: false });
    return () => el.removeEventListener('wheel', handler);
  }, []);

  const applyPreset = useCallback((idx: number) => {
    const p = PRESETS[idx];
    setDepth(p.depth);
    setSides(p.sides);
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

  const nVerts = VERT_COUNTS(sides, depth);

  return (
    <div ref={containerRef} className={styles.container}>
      {/* GPU canvas */}
      <canvas
        ref={glCanvasRef}
        className={styles.canvas}
        style={{ display: useGPU ? 'block' : 'none' }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      />
      {/* CPU canvas */}
      <canvas
        ref={cpuCanvasRef}
        className={styles.canvas}
        style={{ display: useGPU ? 'none' : 'block' }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
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
              <SelectControl
                label="Shape"
                value={String(sides)}
                options={SIDES_OPTS}
                onChange={v => { setSides(Number(v)); setActivePreset(-1); }}
              />
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
          <span className={styles.hudTitle}>Koch {antiKoch ? 'Anti-' : ''}{SHAPE_NAMES[sides] ?? `${sides}-gon`}</span>
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
          <button className={styles.infoBtn} onClick={() => setShowInfo(true)} title="About Koch snowflake">ⓘ</button>
        </div>
      </div>

      {showInfo && (
        <InfoDialog title="Koch Snowflake" onClose={() => setShowInfo(false)}>
          <p>
            Described by Helge von Koch in 1904. Each iteration replaces every line segment
            with a smaller triangle bump. Zoom in and the same structure keeps reappearing.
          </p>
          <h3>Infinite perimeter, finite area</h3>
          <p>
            The perimeter grows by 4/3 each step and diverges to infinity, but the enclosed
            area converges to 8/5 of the original triangle.
          </p>
          <h3>Controls</h3>
          <ul>
            <li><strong>Scroll:</strong> zoom</li>
            <li><strong>Drag:</strong> pan</li>
            <li><strong>Depth:</strong> number of iterations</li>
            <li><strong>Sides:</strong> change the base polygon</li>
          </ul>
        </InfoDialog>
      )}
    </div>
  );
}
