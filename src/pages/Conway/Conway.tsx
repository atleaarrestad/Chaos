import { useState, useEffect, useRef, useCallback } from 'react';
import { Slider, ControlPanel, ControlGroup } from '@/components/Controls';
import styles from './Conway.module.css';

// ─── Grid constants ───────────────────────────────────────────────────────────

const COLS         = 300;
const ROWS         = 200;
const INIT_CELL_PX = 14;  // starting cell size in CSS pixels
const MIN_CELL_PX  = 3;
const MAX_CELL_PX  = 64;
const PAN_THRESHOLD = 4;  // CSS px of movement before a drag becomes a pan

type Grid = Uint8Array;

const cellIdx = (r: number, c: number) => r * COLS + c;
const wrap    = (v: number, max: number) => ((v % max) + max) % max;

const emptyGrid = (): Grid => new Uint8Array(COLS * ROWS);

function randomGrid(density = 0.3): Grid {
  const g = new Uint8Array(COLS * ROWS);
  for (let i = 0; i < g.length; i++) g[i] = Math.random() < density ? 1 : 0;
  return g;
}

function stepGrid(g: Grid): Grid {
  const next = new Uint8Array(COLS * ROWS);
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      let n = 0;
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          if (dr === 0 && dc === 0) continue;
          n += g[cellIdx(wrap(r + dr, ROWS), wrap(c + dc, COLS))];
        }
      }
      const v = g[cellIdx(r, c)];
      next[cellIdx(r, c)] = v ? (n === 2 || n === 3 ? 1 : 0) : (n === 3 ? 1 : 0);
    }
  }
  return next;
}

function countPop(g: Grid): number {
  let n = 0;
  for (let i = 0; i < g.length; i++) n += g[i];
  return n;
}

function applyPattern(cells: [number, number][]): Grid {
  const grid = emptyGrid();
  if (!cells.length) return grid;
  const minR = Math.min(...cells.map(([r]) => r));
  const maxR = Math.max(...cells.map(([r]) => r));
  const minC = Math.min(...cells.map(([, c]) => c));
  const maxC = Math.max(...cells.map(([, c]) => c));
  const offR = Math.floor((ROWS - (maxR - minR + 1)) / 2) - minR;
  const offC = Math.floor((COLS - (maxC - minC + 1)) / 2) - minC;
  for (const [r, c] of cells) {
    const nr = r + offR, nc = c + offC;
    if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS) grid[cellIdx(nr, nc)] = 1;
  }
  return grid;
}

// ─── Preset patterns ──────────────────────────────────────────────────────────

interface Preset {
  label: string;
  sub: string;
  cells: [number, number][];
}

const PRESETS: Preset[] = [
  {
    label: 'Glider',
    sub: 'Spaceship',
    cells: [[0,1],[1,2],[2,0],[2,1],[2,2]],
  },
  {
    label: 'Glider Gun',
    sub: 'Gun',
    // Gosper glider gun — fires a glider every 30 generations
    cells: [
      [0,24],
      [1,22],[1,24],
      [2,12],[2,13],[2,20],[2,21],[2,34],[2,35],
      [3,11],[3,15],[3,20],[3,21],[3,34],[3,35],
      [4,0],[4,1],[4,10],[4,16],[4,20],[4,21],
      [5,0],[5,1],[5,10],[5,14],[5,16],[5,17],[5,22],[5,24],
      [6,10],[6,16],[6,24],
      [7,11],[7,15],
      [8,12],[8,13],
    ],
  },
  {
    label: 'Pulsar',
    sub: 'Osc · P3',
    // 13×13 period-3 oscillator
    cells: [
      [0,2],[0,3],[0,4],[0,8],[0,9],[0,10],
      [2,0],[2,5],[2,7],[2,12],
      [3,0],[3,5],[3,7],[3,12],
      [4,0],[4,5],[4,7],[4,12],
      [5,2],[5,3],[5,4],[5,8],[5,9],[5,10],
      [7,2],[7,3],[7,4],[7,8],[7,9],[7,10],
      [8,0],[8,5],[8,7],[8,12],
      [9,0],[9,5],[9,7],[9,12],
      [10,0],[10,5],[10,7],[10,12],
      [12,2],[12,3],[12,4],[12,8],[12,9],[12,10],
    ],
  },
  {
    label: 'Pentadecathlon',
    sub: 'Osc · P15',
    cells: [
      [0,1],[1,1],[2,0],[2,2],[3,1],[4,1],[5,1],[6,1],[7,0],[7,2],[8,1],[9,1],
    ],
  },
  {
    label: 'LWSS',
    sub: 'Spaceship',
    // Lightweight spaceship — moves horizontally at c/2
    cells: [
      [0,1],[0,2],[0,3],[0,4],
      [1,0],[1,4],
      [2,4],
      [3,0],[3,3],
    ],
  },
  {
    label: 'Blinker',
    sub: 'Osc · P2',
    cells: [[0,0],[0,1],[0,2]],
  },
  {
    label: 'Beacon',
    sub: 'Osc · P2',
    cells: [
      [0,0],[0,1],[1,0],[1,1],
      [2,2],[2,3],[3,2],[3,3],
    ],
  },
  {
    label: 'R-Pentomino',
    sub: 'Methuselah',
    cells: [[0,1],[0,2],[1,0],[1,1],[2,1]],
  },
  {
    label: 'Diehard',
    sub: 'Methuselah',
    // Dies completely after exactly 130 generations
    cells: [
      [0,6],
      [1,0],[1,1],
      [2,1],[2,5],[2,6],[2,7],
    ],
  },
  {
    label: 'Acorn',
    sub: 'Methuselah',
    // Grows for 5206 generations before stabilizing
    cells: [[0,1],[1,3],[2,0],[2,1],[2,4],[2,5],[2,6]],
  },
];

// ─── Canvas rendering ─────────────────────────────────────────────────────────

const ALIVE_COLOR      = '#6ee7b7';
const DEAD_COLOR       = '#0c0c1e';
const BG_COLOR         = '#070712';
const GRID_LINE_COLOR  = 'rgba(255,255,255,0.06)';

/**
 * Render the grid into the canvas using the current viewport.
 * viewX/viewY are the top-left corner in cell-space (fractional).
 * cellSizeCss is the cell size in CSS logical pixels.
 */
function renderGrid(
  canvas: HTMLCanvasElement,
  grid: Grid,
  viewX: number,
  viewY: number,
  cellSizeCss: number,
  showGrid: boolean,
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const W   = canvas.width;
  const H   = canvas.height;
  const dpr = window.devicePixelRatio || 1;
  const cs  = cellSizeCss * dpr; // physical pixels per cell

  // Range of cells visible in the viewport (with 1-cell margin for sub-pixel safety)
  const startC = Math.floor(viewX) - 1;
  const endC   = Math.ceil(viewX + W / cs) + 1;
  const startR = Math.floor(viewY) - 1;
  const endR   = Math.ceil(viewY + H / cs) + 1;

  // At small zoom, skip per-cell dead rendering — fill canvas with dead color
  const showCellBg = cs >= 6;
  const gap = showCellBg ? 1 : 0;

  ctx.fillStyle = showCellBg ? BG_COLOR : DEAD_COLOR;
  ctx.fillRect(0, 0, W, H);

  // Batch dead cells into a single path fill (only when zoomed in enough)
  if (showCellBg) {
    ctx.fillStyle = DEAD_COLOR;
    ctx.beginPath();
    for (let r = startR; r <= endR; r++) {
      for (let c = startC; c <= endC; c++) {
        if (!grid[cellIdx(wrap(r, ROWS), wrap(c, COLS))]) {
          const x = Math.floor((c - viewX) * cs);
          const y = Math.floor((r - viewY) * cs);
          ctx.rect(x + 1, y + 1, Math.ceil(cs) - 1, Math.ceil(cs) - 1);
        }
      }
    }
    ctx.fill();
  }

  // Batch alive cells into a single path fill
  ctx.fillStyle = ALIVE_COLOR;
  ctx.beginPath();
  for (let r = startR; r <= endR; r++) {
    for (let c = startC; c <= endC; c++) {
      if (grid[cellIdx(wrap(r, ROWS), wrap(c, COLS))]) {
        const x = Math.floor((c - viewX) * cs);
        const y = Math.floor((r - viewY) * cs);
        ctx.rect(x + gap, y + gap, Math.ceil(cs) - gap, Math.ceil(cs) - gap);
      }
    }
  }
  ctx.fill();

  // Subtle grid lines when paused and zoomed in
  if (showGrid && cs >= 8) {
    ctx.strokeStyle = GRID_LINE_COLOR;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let c = startC; c <= endC + 1; c++) {
      const x = Math.round((c - viewX) * cs) + 0.5;
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
    }
    for (let r = startR; r <= endR + 1; r++) {
      const y = Math.round((r - viewY) * cs) + 0.5;
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
    }
    ctx.stroke();
  }
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function Conway() {
  const canvasRef      = useRef<HTMLCanvasElement>(null);
  const gridRef        = useRef<Grid>(emptyGrid());
  const genRef         = useRef(0);
  const playingRef     = useRef(false);
  const speedRef       = useRef(10);
  const lastStepRef    = useRef(0);
  const rafRef         = useRef(0);
  const pendingSizeRef = useRef<{ w: number; h: number } | null>(null);

  // Viewport — stored as refs so the RAF loop reads them without re-renders
  const viewRef     = useRef({ x: COLS / 2 - 30, y: ROWS / 2 - 20 });
  const cellSizeRef = useRef(INIT_CELL_PX); // CSS logical pixels

  // Pointer drag state
  const dragRef = useRef<{
    startClientX: number;
    startClientY: number;
    startCell: [number, number];
    hasPanned: boolean;
  } | null>(null);

  const [playing,      setPlaying]      = useState(false);
  const [speed,        setSpeed]        = useState(10);
  const [generation,   setGeneration]   = useState(0);
  const [population,   setPopulation]   = useState(0);
  const [activePreset, setActivePreset] = useState<number>(-1);

  // Sync state → refs
  useEffect(() => { playingRef.current = playing; }, [playing]);
  useEffect(() => { speedRef.current = speed; }, [speed]);

  // ─── RAF loop ─────────────────────────────────────────────────────────────────

  useEffect(() => {
    let mounted = true;

    function frame(ts: number) {
      if (!mounted) return;

      const pending = pendingSizeRef.current;
      if (pending) {
        pendingSizeRef.current = null;
        const canvas = canvasRef.current;
        if (canvas && (canvas.width !== pending.w || canvas.height !== pending.h)) {
          canvas.width  = pending.w;
          canvas.height = pending.h;
        }
      }

      if (playingRef.current) {
        const interval = 1000 / speedRef.current;
        if (ts - lastStepRef.current >= interval) {
          lastStepRef.current = ts;
          gridRef.current = stepGrid(gridRef.current);
          genRef.current += 1;
          setGeneration(genRef.current);
          setPopulation(countPop(gridRef.current));
        }
      }

      if (canvasRef.current) {
        renderGrid(
          canvasRef.current,
          gridRef.current,
          viewRef.current.x,
          viewRef.current.y,
          cellSizeRef.current,
          !playingRef.current,
        );
      }

      rafRef.current = requestAnimationFrame(frame);
    }

    rafRef.current = requestAnimationFrame(frame);
    return () => { mounted = false; cancelAnimationFrame(rafRef.current); };
  }, []);

  // ─── Resize observer ──────────────────────────────────────────────────────────

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let firstResize = true;

    const obs = new ResizeObserver((entries) => {
      for (const e of entries) {
        const { width, height } = e.contentRect;

        // On the first resize, center the viewport on the grid
        if (firstResize) {
          firstResize = false;
          viewRef.current = {
            x: COLS / 2 - width  / (2 * cellSizeRef.current),
            y: ROWS / 2 - height / (2 * cellSizeRef.current),
          };
        }

        const dpr = window.devicePixelRatio || 1;
        pendingSizeRef.current = {
          w: Math.round(width  * dpr),
          h: Math.round(height * dpr),
        };
      }
    });

    obs.observe(canvas.parentElement!);
    return () => obs.disconnect();
  }, []);

  // ─── Wheel zoom (zoom toward cursor) ──────────────────────────────────────────

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect   = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left; // CSS pixels from canvas left
      const mouseY = e.clientY - rect.top;

      const oldCs  = cellSizeRef.current;
      const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
      const newCs  = Math.max(MIN_CELL_PX, Math.min(MAX_CELL_PX, oldCs * factor));

      // Keep the cell under the cursor fixed while zooming
      viewRef.current.x += mouseX * (1 / oldCs - 1 / newCs);
      viewRef.current.y += mouseY * (1 / oldCs - 1 / newCs);
      cellSizeRef.current = newCs;
    };

    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', onWheel);
  }, []);

  // ─── Pointer interaction: drag to pan, click-release to toggle ────────────────

  const getCellAt = useCallback((clientX: number, clientY: number): [number, number] => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const c = Math.floor(viewRef.current.x + (clientX - rect.left) / cellSizeRef.current);
    const r = Math.floor(viewRef.current.y + (clientY - rect.top)  / cellSizeRef.current);
    return [wrap(r, ROWS), wrap(c, COLS)];
  }, []);

  const onPointerDown = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) return;
    dragRef.current = {
      startClientX: e.clientX,
      startClientY: e.clientY,
      startCell: getCellAt(e.clientX, e.clientY),
      hasPanned: false,
    };
    canvasRef.current!.style.cursor = 'grabbing';
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }, [getCellAt]);

  const onPointerMove = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    if (!drag) return;

    const dist = Math.hypot(
      e.clientX - drag.startClientX,
      e.clientY - drag.startClientY,
    );

    if (!drag.hasPanned && dist > PAN_THRESHOLD) {
      drag.hasPanned = true;
    }

    if (drag.hasPanned) {
      // movementX/Y are CSS pixels — same space as cellSizeRef (CSS px)
      viewRef.current.x -= e.movementX / cellSizeRef.current;
      viewRef.current.y -= e.movementY / cellSizeRef.current;
    }
  }, []);

  const onPointerUp = useCallback(() => {
    const drag = dragRef.current;
    if (!drag) return;

    if (!drag.hasPanned) {
      // Short click with no panning: toggle the cell that was pressed
      const [r, c] = drag.startCell;
      gridRef.current[cellIdx(r, c)] ^= 1;
      setActivePreset(-1);
      setPopulation(countPop(gridRef.current));
    }

    dragRef.current = null;
    if (canvasRef.current) canvasRef.current.style.cursor = 'grab';
  }, []);

  const onPointerCancel = useCallback(() => {
    dragRef.current = null;
    if (canvasRef.current) canvasRef.current.style.cursor = 'grab';
  }, []);

  // ─── Actions ──────────────────────────────────────────────────────────────────

  const centerView = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    viewRef.current = {
      x: COLS / 2 - rect.width  / (2 * cellSizeRef.current),
      y: ROWS / 2 - rect.height / (2 * cellSizeRef.current),
    };
  }, []);

  const applyPreset = useCallback((i: number) => {
    gridRef.current = applyPattern(PRESETS[i].cells);
    genRef.current = 0;
    setGeneration(0);
    setPopulation(countPop(gridRef.current));
    setActivePreset(i);
    setPlaying(false);
    // Reset zoom and re-center
    cellSizeRef.current = INIT_CELL_PX;
    centerView();
  }, [centerView]);

  const handleStep = useCallback(() => {
    gridRef.current = stepGrid(gridRef.current);
    genRef.current += 1;
    setGeneration(genRef.current);
    setPopulation(countPop(gridRef.current));
  }, []);

  const handleRandom = useCallback(() => {
    gridRef.current = randomGrid();
    genRef.current = 0;
    setGeneration(0);
    setPopulation(countPop(gridRef.current));
    setActivePreset(-1);
    setPlaying(false);
  }, []);

  const handleClear = useCallback(() => {
    gridRef.current = emptyGrid();
    genRef.current = 0;
    setGeneration(0);
    setPopulation(0);
    setActivePreset(-1);
    setPlaying(false);
  }, []);

  const togglePlay = useCallback(() => setPlaying(p => !p), []);

  return (
    <div className={styles.container}>
      <canvas
        ref={canvasRef}
        className={styles.canvas}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerCancel}
      />

      {/* Sidebar */}
      <div className={styles.sidebar}>
        <div className={styles.sidebarPanels}>

          <button
            className={[styles.playBtn, playing ? styles.playBtnActive : ''].join(' ')}
            onClick={togglePlay}
          >
            {playing ? '⏸  Pause' : '▶  Play'}
          </button>

          <ControlPanel title="Speed">
            <ControlGroup>
              <Slider
                label="Steps / sec"
                value={speed}
                min={1} max={30} step={1}
                unit="fps"
                onChange={setSpeed}
              />
            </ControlGroup>
          </ControlPanel>

          <ControlPanel title="Presets">
            <div className={styles.presetGrid}>
              {PRESETS.map((p, i) => (
                <button
                  key={p.label}
                  className={[
                    styles.presetBtn,
                    activePreset === i ? styles.presetBtnActive : '',
                  ].join(' ')}
                  onClick={() => applyPreset(i)}
                >
                  <span className={styles.presetLabel}>{p.label}</span>
                  <span className={styles.presetSub}>{p.sub}</span>
                </button>
              ))}
            </div>
          </ControlPanel>

        </div>

        <div className={styles.sidebarActions}>
          <div className={styles.actionPanel}>
            <span className={styles.actionPanelLabel}>Actions</span>
            <div className={styles.actionRow}>
              <button className={styles.actionBtn} onClick={handleStep} disabled={playing}>
                Step
              </button>
              <button className={styles.actionBtn} onClick={handleRandom}>
                Random
              </button>
              <button className={styles.actionBtn} onClick={handleClear}>
                Clear
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* HUD */}
      <div className={styles.hud}>
        <div className={styles.hudLeft}>
          <span className={styles.hudTitle}>Conway's Game of Life</span>
          <span className={styles.hudSub}>
            gen {generation.toLocaleString()} · pop {population.toLocaleString()}
          </span>
        </div>
        <div className={styles.hudRight}>
          <span className={styles.hudHint}>scroll to zoom · drag to pan · click to draw</span>
        </div>
      </div>
    </div>
  );
}
