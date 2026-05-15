import { useState, useEffect, useRef, useCallback } from 'react';
import { Slider, ControlPanel, ControlGroup } from '@/components/Controls';
import styles from './Conway.module.css';

// ─── Grid constants ───────────────────────────────────────────────────────────

const COLS = 60;
const ROWS = 40;

type Grid = Uint8Array;

const cellIdx = (r: number, c: number) => r * COLS + c;

const emptyGrid = (): Grid => new Uint8Array(COLS * ROWS);

function randomGrid(density = 0.35): Grid {
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
          n += g[cellIdx((r + dr + ROWS) % ROWS, (c + dc + COLS) % COLS)];
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
    // 10 cells forming a period-15 oscillator
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
    // Two 2×2 blocks sharing a corner
    cells: [
      [0,0],[0,1],[1,0],[1,1],
      [2,2],[2,3],[3,2],[3,3],
    ],
  },
  {
    label: 'R-Pentomino',
    sub: 'Methuselah',
    // 1103 generations before stabilizing
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

const ALIVE_COLOR      = '#6ee7b7';               // --col-conway
const DEAD_COLOR       = '#0c0c1e';               // subtle cell background
const BG_COLOR         = '#070712';               // container background
const GRID_LINE_COLOR  = 'rgba(255,255,255,0.06)'; // visible grid lines when paused

function renderGrid(canvas: HTMLCanvasElement, grid: Grid, showGrid: boolean) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const W = canvas.width, H = canvas.height;
  const cw = W / COLS, ch = H / ROWS;

  // Background fills the 1px gaps between cells
  ctx.fillStyle = BG_COLOR;
  ctx.fillRect(0, 0, W, H);

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const x = Math.floor(c * cw) + 1;
      const y = Math.floor(r * ch) + 1;
      const w = Math.floor((c + 1) * cw) - Math.floor(c * cw) - 1;
      const h = Math.floor((r + 1) * ch) - Math.floor(r * ch) - 1;
      ctx.fillStyle = grid[cellIdx(r, c)] ? ALIVE_COLOR : DEAD_COLOR;
      ctx.fillRect(x, y, w, h);
    }
  }

  // When paused, overlay a visible grid so it's easy to place cells precisely
  if (showGrid) {
    ctx.strokeStyle = GRID_LINE_COLOR;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let c = 0; c <= COLS; c++) {
      const x = Math.round(c * cw) + 0.5;
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
    }
    for (let r = 0; r <= ROWS; r++) {
      const y = Math.round(r * ch) + 0.5;
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
    }
    ctx.stroke();
  }
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function Conway() {
  const canvasRef     = useRef<HTMLCanvasElement>(null);
  const gridRef       = useRef<Grid>(emptyGrid());
  const genRef        = useRef(0);
  const playingRef    = useRef(false);
  const speedRef      = useRef(10);
  const lastStepRef   = useRef(0);
  const rafRef        = useRef(0);
  const paintModeRef  = useRef<0 | 1 | null>(null);
  const pendingSizeRef = useRef<{ w: number; h: number } | null>(null);

  const [playing,     setPlaying]     = useState(false);
  const [speed,       setSpeed]       = useState(10);
  const [generation,  setGeneration]  = useState(0);
  const [population,  setPopulation]  = useState(0);
  const [activePreset, setActivePreset] = useState<number>(-1);

  // Sync state → refs so the RAF loop always reads the latest values
  useEffect(() => { playingRef.current = playing; }, [playing]);
  useEffect(() => { speedRef.current = speed; }, [speed]);

  // ─── Animation loop ─────────────────────────────────────────────────────────

  useEffect(() => {
    let mounted = true;

    function frame(ts: number) {
      if (!mounted) return;

      // Apply pending canvas resize atomically before drawing
      const pending = pendingSizeRef.current;
      if (pending) {
        pendingSizeRef.current = null;
        const canvas = canvasRef.current;
        if (canvas && (canvas.width !== pending.w || canvas.height !== pending.h)) {
          canvas.width = pending.w;
          canvas.height = pending.h;
        }
      }

      // Advance simulation at the requested fps
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

      if (canvasRef.current) renderGrid(canvasRef.current, gridRef.current, !playingRef.current);

      rafRef.current = requestAnimationFrame(frame);
    }

    rafRef.current = requestAnimationFrame(frame);
    return () => {
      mounted = false;
      cancelAnimationFrame(rafRef.current);
    };
  }, []);

  // ─── Resize observer ────────────────────────────────────────────────────────

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const obs = new ResizeObserver((entries) => {
      for (const e of entries) {
        const { width, height } = e.contentRect;
        const dpr = window.devicePixelRatio || 1;
        pendingSizeRef.current = {
          w: Math.round(width * dpr),
          h: Math.round(height * dpr),
        };
      }
    });
    // Observe the parent container so width/height match CSS dimensions
    obs.observe(canvas.parentElement!);
    return () => obs.disconnect();
  }, []);

  // ─── Pointer interaction (draw / erase cells) ────────────────────────────────

  const getCellAt = useCallback((clientX: number, clientY: number): [number, number] | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const c = Math.floor(((clientX - rect.left) / rect.width) * COLS);
    const r = Math.floor(((clientY - rect.top) / rect.height) * ROWS);
    if (r >= 0 && r < ROWS && c >= 0 && c < COLS) return [r, c];
    return null;
  }, []);

  const onPointerDown = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    const cell = getCellAt(e.clientX, e.clientY);
    if (!cell) return;
    const [r, c] = cell;
    // Toggle: clicking a live cell erases, clicking a dead cell draws
    const newVal: 0 | 1 = gridRef.current[cellIdx(r, c)] ? 0 : 1;
    paintModeRef.current = newVal;
    gridRef.current[cellIdx(r, c)] = newVal;
    setActivePreset(-1);
    setPopulation(countPop(gridRef.current));
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }, [getCellAt]);

  const onPointerMove = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    if (paintModeRef.current === null) return;
    const cell = getCellAt(e.clientX, e.clientY);
    if (!cell) return;
    const [r, c] = cell;
    gridRef.current[cellIdx(r, c)] = paintModeRef.current;
  }, [getCellAt]);

  const onPointerUp = useCallback(() => {
    if (paintModeRef.current !== null) {
      paintModeRef.current = null;
      setPopulation(countPop(gridRef.current));
    }
  }, []);

  // ─── Actions ─────────────────────────────────────────────────────────────────

  const applyPreset = useCallback((i: number) => {
    gridRef.current = applyPattern(PRESETS[i].cells);
    genRef.current = 0;
    setGeneration(0);
    setPopulation(countPop(gridRef.current));
    setActivePreset(i);
    setPlaying(false);
  }, []);

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
      />

      {/* Sidebar */}
      <div className={styles.sidebar}>
        <div className={styles.sidebarPanels}>

          {/* Play / Pause */}
          <button
            className={[styles.playBtn, playing ? styles.playBtnActive : ''].join(' ')}
            onClick={togglePlay}
          >
            {playing ? '⏸  Pause' : '▶  Play'}
          </button>

          {/* Speed */}
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

          {/* Presets */}
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

        {/* Action buttons */}
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
          <span className={styles.hudHint}>click · drag to draw</span>
        </div>
      </div>
    </div>
  );
}
