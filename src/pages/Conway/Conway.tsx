import { useState, useEffect, useRef, useCallback } from 'react';
import { Slider, Toggle, ControlPanel, ControlGroup } from '@/components/Controls';
import styles from './Conway.module.css';

// ─── Grid constants ───────────────────────────────────────────────────────────

const COLS         = 300;
const ROWS         = 200;
const INIT_CELL_PX = 14;
const MIN_CELL_PX  = 3;
const MAX_CELL_PX  = 64;
const MAX_AGE      = 64;   // age caps here for color bucketing
const SPARK_LEN    = 150;  // generations shown in the sparkline

type Grid    = Uint8Array;
type AgeGrid = Uint16Array;

const cellIdx = (r: number, c: number) => r * COLS + c;
const wrap    = (v: number, max: number) => ((v % max) + max) % max;

const emptyGrid    = (): Grid    => new Uint8Array(COLS * ROWS);
const emptyAgeGrid = (): AgeGrid => new Uint16Array(COLS * ROWS);


function initialAgeGrid(g: Grid): AgeGrid {
  const ages = new Uint16Array(COLS * ROWS);
  for (let i = 0; i < g.length; i++) ages[i] = g[i] ? 1 : 0;
  return ages;
}

function stepGrid(g: Grid, ages: AgeGrid): [Grid, AgeGrid] {
  const next     = new Uint8Array(COLS * ROWS);
  const nextAges = new Uint16Array(COLS * ROWS);
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      let n = 0;
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          if (dr === 0 && dc === 0) continue;
          n += g[cellIdx(wrap(r + dr, ROWS), wrap(c + dc, COLS))];
        }
      }
      const i     = cellIdx(r, c);
      const alive = g[i] ? (n === 2 || n === 3 ? 1 : 0) : (n === 3 ? 1 : 0);
      next[i]     = alive;
      nextAges[i] = alive ? Math.min(ages[i] + 1, MAX_AGE) : 0;
    }
  }
  return [next, nextAges];
}

function countPop(g: Grid): number {
  let n = 0;
  for (let i = 0; i < g.length; i++) n += g[i];
  return n;
}

function applyPattern(cells: [number, number][]): [Grid, AgeGrid] {
  const grid = emptyGrid();
  if (!cells.length) return [grid, emptyAgeGrid()];
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
  return [grid, initialAgeGrid(grid)];
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

const DEAD_COLOR      = '#0c0c1e';
const BG_COLOR        = '#070712';
const GRID_LINE_COLOR = 'rgba(255,255,255,0.06)';

// Age-based colour buckets (index = bucket):
//   0 = newborn (age 1)  — near-white flash
//   1 = young   (2–3)    — light green
//   2 = mature  (4–9)    — standard Conway green
//   3 = old     (10+)    — deeper green
const AGE_COLORS = ['#f0fdf9', '#a7f3d0', '#6ee7b7', '#34d399'] as const;

function ageBucket(age: number): 0 | 1 | 2 | 3 {
  if (age <= 1) return 0;
  if (age <= 3) return 1;
  if (age <= 9) return 2;
  return 3;
}

/**
 * Render the grid into the canvas using the current viewport.
 * viewX/viewY are the top-left corner in cell-space (fractional).
 * cellSizeCss is the cell size in CSS logical pixels.
 */
function renderGrid(
  canvas: HTMLCanvasElement,
  grid: Grid,
  ages: AgeGrid,
  viewX: number,
  viewY: number,
  cellSizeCss: number,
  showGrid: boolean,
  useAgeColor: boolean,
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

  // Single pass: bucket alive cells by age, then draw each bucket with its colour
  type XY = [number, number];
  const cellW = Math.ceil(cs) - gap;

  if (useAgeColor) {
    const buckets: [XY[], XY[], XY[], XY[]] = [[], [], [], []];
    for (let r = startR; r <= endR; r++) {
      for (let c = startC; c <= endC; c++) {
        const i = cellIdx(wrap(r, ROWS), wrap(c, COLS));
        if (grid[i]) {
          buckets[ageBucket(ages[i])].push([
            Math.floor((c - viewX) * cs),
            Math.floor((r - viewY) * cs),
          ]);
        }
      }
    }
    for (let b = 0; b < 4; b++) {
      if (!buckets[b].length) continue;
      ctx.fillStyle = AGE_COLORS[b];
      ctx.beginPath();
      for (const [x, y] of buckets[b]) {
        ctx.rect(x + gap, y + gap, cellW, cellW);
      }
      ctx.fill();
    }
  } else {
    ctx.fillStyle = AGE_COLORS[2]; // flat standard green
    ctx.beginPath();
    for (let r = startR; r <= endR; r++) {
      for (let c = startC; c <= endC; c++) {
        const i = cellIdx(wrap(r, ROWS), wrap(c, COLS));
        if (grid[i]) {
          const x = Math.floor((c - viewX) * cs);
          const y = Math.floor((r - viewY) * cs);
          ctx.rect(x + gap, y + gap, cellW, cellW);
        }
      }
    }
    ctx.fill();
  }

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
  const ageGridRef     = useRef<AgeGrid>(emptyAgeGrid());
  const ageColorRef    = useRef(true);
  const genRef         = useRef(0);
  const playingRef     = useRef(false);
  const speedRef       = useRef(10);
  const lastStepRef    = useRef(0);
  const rafRef         = useRef(0);
  const pendingSizeRef = useRef<{ w: number; h: number } | null>(null);
  const sparkBufRef    = useRef<number[]>([]);

  // Viewport — stored as refs so the RAF loop reads them without re-renders
  const viewRef     = useRef({ x: COLS / 2 - 30, y: ROWS / 2 - 20 });
  const cellSizeRef = useRef(INIT_CELL_PX); // CSS logical pixels

  // Pointer drag state: left-drag = paint, right/middle-drag = pan
  const dragRef = useRef<{
    isPan: boolean;
    paintState: 0 | 1;
    paintedCells: Set<number>;
  } | null>(null);

  const [playing,      setPlaying]      = useState(false);
  const [speed,        setSpeed]        = useState(10);
  const [ageColor,     setAgeColor]     = useState(true);
  const [showGraph,    setShowGraph]    = useState(true);
  const [showInfo,     setShowInfo]     = useState(false);
  const [generation,   setGeneration]   = useState(0);
  const [population,   setPopulation]   = useState(0);
  const [activePreset, setActivePreset] = useState<number>(-1);
  const [sparkData,    setSparkData]    = useState<number[]>([]);

  // Sync state → refs
  useEffect(() => { playingRef.current = playing; }, [playing]);
  useEffect(() => { speedRef.current = speed; }, [speed]);
  useEffect(() => { ageColorRef.current = ageColor; }, [ageColor]);

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
          const [nextGrid, nextAges] = stepGrid(gridRef.current, ageGridRef.current);
          gridRef.current    = nextGrid;
          ageGridRef.current = nextAges;
          genRef.current += 1;
          const pop = countPop(gridRef.current);
          sparkBufRef.current = [...sparkBufRef.current.slice(-(SPARK_LEN - 1)), pop];
          setGeneration(genRef.current);
          setPopulation(pop);
          setSparkData([...sparkBufRef.current]);
        }
      }

      if (canvasRef.current) {
        renderGrid(
          canvasRef.current,
          gridRef.current,
          ageGridRef.current,
          viewRef.current.x,
          viewRef.current.y,
          cellSizeRef.current,
          !playingRef.current,
          ageColorRef.current,
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

  // ─── Pointer interaction: left-drag paints, right/middle-drag pans ──────────

  const getCellAt = useCallback((clientX: number, clientY: number): [number, number] => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const c = Math.floor(viewRef.current.x + (clientX - rect.left) / cellSizeRef.current);
    const r = Math.floor(viewRef.current.y + (clientY - rect.top)  / cellSizeRef.current);
    return [wrap(r, ROWS), wrap(c, COLS)];
  }, []);

  const onPointerDown = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    (e.target as HTMLElement).setPointerCapture(e.pointerId);

    if (e.button === 1 || e.button === 2) {
      // Middle / right-click → pan
      dragRef.current = { isPan: true, paintState: 0, paintedCells: new Set() };
      canvasRef.current!.style.cursor = 'grabbing';
      return;
    }

    if (e.button !== 0) return;

    // Left-click → immediately toggle and enter paint mode
    const [r, c] = getCellAt(e.clientX, e.clientY);
    const idx        = cellIdx(r, c);
    const newState: 0 | 1 = gridRef.current[idx] ? 0 : 1;
    gridRef.current[idx]    = newState;
    ageGridRef.current[idx] = newState; // 1 = just born, 0 = just died
    dragRef.current = { isPan: false, paintState: newState, paintedCells: new Set([idx]) };
    setActivePreset(-1);
    setPopulation(countPop(gridRef.current));
  }, [getCellAt]);

  const onPointerMove = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    if (!drag) return;

    if (drag.isPan) {
      viewRef.current.x -= e.movementX / cellSizeRef.current;
      viewRef.current.y -= e.movementY / cellSizeRef.current;
      return;
    }

    // Paint: apply paintState to every new cell the pointer enters
    const [r, c] = getCellAt(e.clientX, e.clientY);
    const idx = cellIdx(r, c);
    if (!drag.paintedCells.has(idx)) {
      drag.paintedCells.add(idx);
      gridRef.current[idx]    = drag.paintState;
      ageGridRef.current[idx] = drag.paintState;
      setPopulation(countPop(gridRef.current));
    }
  }, [getCellAt]);

  const onPointerUp = useCallback(() => {
    dragRef.current = null;
    if (canvasRef.current) canvasRef.current.style.cursor = 'crosshair';
  }, []);

  const onPointerCancel = useCallback(() => {
    dragRef.current = null;
    if (canvasRef.current) canvasRef.current.style.cursor = 'crosshair';
  }, []);

  const onContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
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
    const [g, ages] = applyPattern(PRESETS[i].cells);
    gridRef.current    = g;
    ageGridRef.current = ages;
    genRef.current     = 0;
    sparkBufRef.current = [];
    setGeneration(0);
    setPopulation(countPop(g));
    setSparkData([]);
    setActivePreset(i);
    setPlaying(false);
    // Reset zoom and re-center
    cellSizeRef.current = INIT_CELL_PX;
    centerView();
  }, [centerView]);

  const handleStep = useCallback(() => {
    const [nextGrid, nextAges] = stepGrid(gridRef.current, ageGridRef.current);
    gridRef.current    = nextGrid;
    ageGridRef.current = nextAges;
    genRef.current += 1;
    const pop = countPop(gridRef.current);
    sparkBufRef.current = [...sparkBufRef.current.slice(-(SPARK_LEN - 1)), pop];
    setGeneration(genRef.current);
    setPopulation(pop);
    setSparkData([...sparkBufRef.current]);
  }, []);

  const handleClear = useCallback(() => {
    gridRef.current    = emptyGrid();
    ageGridRef.current = emptyAgeGrid();
    genRef.current     = 0;
    sparkBufRef.current = [];
    setGeneration(0);
    setPopulation(0);
    setSparkData([]);
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
        onContextMenu={onContextMenu}
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

          <ControlPanel title="Display">
            <ControlGroup>
              <Toggle
                label="Age colouring"
                value={ageColor}
                onChange={setAgeColor}
                description="Shade cells by how long they've been alive"
              />
              {ageColor && (
                <div className={styles.ageLegend}>
                  {(
                    [
                      ['#f0fdf9', 'Newborn', '1 gen'],
                      ['#a7f3d0', 'Young',   '2–3'],
                      ['#6ee7b7', 'Mature',  '4–9'],
                      ['#34d399', 'Old',     '10+'],
                    ] as const
                  ).map(([color, label, range]) => (
                    <div key={label} className={styles.ageLegendRow}>
                      <span className={styles.ageSwatch} style={{ background: color }} />
                      <span className={styles.ageLegendLabel}>{label}</span>
                      <span className={styles.ageLegendRange}>{range}</span>
                    </div>
                  ))}
                </div>
              )}
              <Toggle
                label="Population graph"
                value={showGraph}
                onChange={setShowGraph}
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
              <button className={styles.actionBtn} onClick={handleClear}>
                Clear
              </button>
              <button className={styles.actionBtn} onClick={centerView}>
                Center
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Population graph panel */}
      {showGraph && sparkData.length >= 2 && (() => {
        const SVG_W = 240, SVG_H = 100;
        const PL = 38, PR = 8, PT = 8, PB = 24;
        const plotW = SVG_W - PL - PR;
        const plotH = SVG_H - PT - PB;
        const max    = Math.max(...sparkData, 1);
        const startGen = Math.max(0, generation - sparkData.length + 1);
        const fmtN = (n: number) =>
          n >= 10000 ? `${(n / 1000).toFixed(0)}k`
          : n >= 1000 ? `${(n / 1000).toFixed(1)}k`
          : `${n}`;

        const linePts = sparkData
          .map((v, i) => {
            const x = PL + (i / (sparkData.length - 1)) * plotW;
            const y = PT + (1 - v / max) * plotH;
            return `${x.toFixed(1)},${y.toFixed(1)}`;
          })
          .join(' ');

        const fillPts = `${PL},${PT + plotH} ${linePts} ${(PL + plotW).toFixed(1)},${PT + plotH}`;

        return (
          <div className={styles.popPanel}>
            <div className={styles.popPanelHeader}>
              <span className={styles.popPanelTitle}>Population</span>
              <button
                className={styles.popCloseBtn}
                onClick={() => setShowGraph(false)}
                aria-label="Close graph"
              >
                ×
              </button>
            </div>
            <div className={styles.popPlot}>
              <svg
                width={SVG_W} height={SVG_H}
                viewBox={`0 0 ${SVG_W} ${SVG_H}`}
                className={styles.popSvg}
                aria-hidden="true"
              >
                {/* Axis lines */}
                <line x1={PL} y1={PT} x2={PL} y2={PT + plotH}
                  stroke="rgba(255,255,255,0.15)" strokeWidth="1" />
                <line x1={PL} y1={PT + plotH} x2={PL + plotW} y2={PT + plotH}
                  stroke="rgba(255,255,255,0.15)" strokeWidth="1" />

                {/* Y ticks: 0, mid, max */}
                {([0, 0.5, 1] as const).map((frac) => {
                  const val = Math.round(max * frac);
                  const y   = PT + (1 - frac) * plotH;
                  return (
                    <g key={frac}>
                      <line x1={PL - 3} y1={y} x2={PL} y2={y}
                        stroke="rgba(255,255,255,0.2)" strokeWidth="1" />
                      <text x={PL - 5} y={y + 3.5} textAnchor="end"
                        fill="rgba(255,255,255,0.4)" fontSize="8">
                        {fmtN(val)}
                      </text>
                    </g>
                  );
                })}

                {/* X tick labels: start gen + current gen */}
                <text x={PL} y={PT + plotH + 13} textAnchor="middle"
                  fill="rgba(255,255,255,0.35)" fontSize="8">
                  {startGen.toLocaleString()}
                </text>
                <text x={PL + plotW} y={PT + plotH + 13} textAnchor="middle"
                  fill="rgba(255,255,255,0.35)" fontSize="8">
                  {generation.toLocaleString()}
                </text>

                {/* Axis labels */}
                <text x={PL + plotW / 2} y={SVG_H - 1} textAnchor="middle"
                  fill="rgba(255,255,255,0.25)" fontSize="7.5">
                  generation →
                </text>
                <text
                  x={8} y={PT + plotH / 2}
                  textAnchor="middle"
                  fill="rgba(255,255,255,0.25)"
                  fontSize="7.5"
                  transform={`rotate(-90, 8, ${PT + plotH / 2})`}
                >
                  ↑ pop
                </text>

                {/* Fill under the line */}
                <polygon
                  points={fillPts}
                  fill="var(--col-conway)"
                  opacity="0.08"
                />

                {/* Data line */}
                <polyline
                  points={linePts}
                  fill="none"
                  stroke="var(--col-conway)"
                  strokeWidth="1.5"
                  strokeLinejoin="round"
                  strokeLinecap="round"
                  opacity="0.85"
                />
              </svg>
            </div>
          </div>
        );
      })()}

      {/* HUD */}
      <div className={styles.hud}>
        <div className={styles.hudLeft}>
          <span className={styles.hudTitle}>Conway's Game of Life</span>
          <span className={styles.hudSub}>
            gen {generation.toLocaleString()} · pop {population.toLocaleString()}
          </span>
        </div>
        <div className={styles.hudRight}>
          <span className={styles.hudHint}>right-drag to pan · scroll to zoom · click/drag to draw</span>
          <button className={styles.infoBtn} onClick={() => setShowInfo(true)} title="About Conway's Game of Life">ⓘ</button>
        </div>
      </div>

      {/* Info dialog */}
      {showInfo && (
        <div className={styles.infoOverlay} onClick={() => setShowInfo(false)}>
          <div className={styles.infoDialog} onClick={e => e.stopPropagation()}>
            <div className={styles.infoHeader}>
              <span className={styles.infoTitle}>Conway's Game of Life</span>
              <button className={styles.infoCloseBtn} onClick={() => setShowInfo(false)}>×</button>
            </div>
            <div className={styles.infoBody}>
              <p>
                Devised by John Horton Conway in 1970. Each cell is alive or dead, and every
                generation all cells update at the same time based on four simple rules.
              </p>

              <h3>The rules</h3>
              <ol>
                <li><strong>Underpopulation:</strong> fewer than 2 live neighbours dies.</li>
                <li><strong>Survival:</strong> 2 or 3 live neighbours survives.</li>
                <li><strong>Overpopulation:</strong> more than 3 live neighbours dies.</li>
                <li><strong>Reproduction:</strong> exactly 3 live neighbours becomes alive.</li>
              </ol>

              <h3>Controls</h3>
              <ul>
                <li><strong>Click / drag:</strong> draw or erase cells</li>
                <li><strong>Right-drag:</strong> pan</li>
                <li><strong>Scroll:</strong> zoom</li>
              </ul>

              <h3>Age colours</h3>
              <ul>
                <li><span style={{ color: '#f0fdf9' }}>■</span> <strong>Newborn:</strong> 1 generation</li>
                <li><span style={{ color: '#a7f3d0' }}>■</span> <strong>Young:</strong> 2–3 generations</li>
                <li><span style={{ color: '#6ee7b7' }}>■</span> <strong>Mature:</strong> 4–9 generations</li>
                <li><span style={{ color: '#34d399' }}>■</span> <strong>Old:</strong> 10+</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
