import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ControlGroup, ControlPanel, SimControls, Slider, Toggle } from '@/components/Controls';
import { useFullscreen } from '@/hooks/useFullscreen';
import ExportDialog from '../../components/ExportDialog/ExportDialog';
import { exportImage } from '../../lib/exportImage';
import styles from './CellularAutomata.module.css';

// Constants
const COLS           = 300;
const ROWS           = 200;
const BG_HEX         = '#0c0c1e';
const BG_COLOR       = '#070712';
const DEAD_COLOR     = '#0c0c1e';
const GRID_LINE_COLOR = 'rgba(255,255,255,0.06)';
const DEFAULT_DENSITY = 30;
const CENTER_X       = Math.floor(COLS / 2);
const CENTER_Y       = Math.floor(ROWS / 2);
const INIT_CELL_PX   = 14;
const INIT_CELL_PX_1D = 3;
const MIN_CELL_PX    = 2;
const MAX_CELL_PX    = 64;
const MAX_AGE        = 64;
const SPARK_LEN      = 150;

// Types
type SimType = '2d' | 'brain' | 'ant' | '1d';
type Rgb     = [number, number, number];

interface RulesetDef {
  id: string;
  name: string;
  category: string;
  description: string;
  rules: string[];
  notation?: string;
  simType: SimType;
  color: string;
  birth?: number[];
  survive?: number[];
  rule?: number;
  antTurns?: number[];
  antStateColors?: string[];
  defaultDensity?: number;
  defaultDelay?: number;
  startEmpty?: boolean;
}

interface Ant { x: number; y: number; dir: number; }

interface SimState {
  grid: Uint8Array;
  ageGrid: Uint16Array;
  ants: Ant[];
  row: Uint8Array;
  rowHistory: Uint8Array;
  rowIdx: number;
  generation: number;
  offscreen: OffscreenCanvas;
  offCtx: OffscreenCanvasRenderingContext2D;
}

// Rulesets
const RULESETS = [
  {
    id: 'conway',
    name: "Conway's Life",
    category: '2D Automaton',
    description:
      "The original zero-player game devised by John Conway in 1970. Four simple rules (underpopulation, survival, overpopulation, reproduction) produce oscillators, spaceships, and unbounded growth.",
    rules: [
      'A live cell with fewer than 2 neighbors dies (underpopulation)',
      'A live cell with 2 or 3 neighbors survives',
      'A live cell with more than 3 neighbors dies (overpopulation)',
      'A dead cell with exactly 3 live neighbors becomes alive (reproduction)',
    ],
    simType: '2d',
    birth: [3],
    survive: [2, 3],
    color: '#6ee7b7',
    notation: 'B3/S23',
    startEmpty: true,
  },
  {
    id: 'highlife',
    name: 'HighLife',
    category: '2D Automaton',
    description:
      "Conway's rules plus birth at 6 neighbors. Contains a self-replicating pattern (the 'replicator') that spawns copies of itself.",
    rules: [
      'A live cell with fewer than 2 neighbors dies',
      'A live cell with 2 or 3 neighbors survives',
      'A live cell with more than 3 neighbors dies',
      'A dead cell with 3 or 6 live neighbors becomes alive',
    ],
    simType: '2d',
    birth: [3, 6],
    survive: [2, 3],
    color: '#4ade80',
    notation: 'B36/S23',
    startEmpty: true,
  },
  {
    id: 'day-and-night',
    name: 'Day & Night',
    category: '2D Automaton',
    description:
      'Symmetric rules: live and dead regions are interchangeable. Produces stable isolated islands with rich internal structure.',
    rules: [
      'Born with 3, 6, 7, or 8 live neighbors',
      'Survives with 3, 4, 6, 7, or 8 live neighbors',
      'Live and dead obey identical rules (symmetric)',
    ],
    simType: '2d',
    birth: [3, 6, 7, 8],
    survive: [3, 4, 6, 7, 8],
    color: '#38bdf8',
    notation: 'B3678/S34678',
    startEmpty: true,
  },
  {
    id: 'maze',
    name: 'Maze',
    category: '2D Automaton',
    description:
      'Cells are highly survivable and grow outward into winding corridors. Once a wall forms, it never dissolves.',
    rules: [
      'Born with exactly 3 live neighbors',
      'Survives with 1, 2, 3, 4, or 5 live neighbors',
      'Dies with 0 or 6+ live neighbors (walls never dissolve)',
    ],
    simType: '2d',
    birth: [3],
    survive: [1, 2, 3, 4, 5],
    color: '#a78bfa',
    notation: 'B3/S12345',
    startEmpty: true,
  },
  {
    id: 'seeds',
    name: 'Seeds',
    category: '2D Automaton',
    description:
      'Nothing survives, but any dead cell with exactly 2 live neighbors springs to life. Produces explosive wave-like growth.',
    rules: [
      'Born with exactly 2 live neighbors',
      'No cell ever survives to the next generation',
    ],
    simType: '2d',
    birth: [2],
    survive: [],
    color: '#fbbf24',
    notation: 'B2/S',
    startEmpty: true,
    defaultDelay: 60,
  },
  {
    id: 'replicator',
    name: 'Replicator',
    category: '2D Automaton',
    description:
      'Odd-neighbor birth and survival rules cause every finite pattern to replicate itself, producing an infinite mosaic of copies.',
    rules: [
      'Born with 1, 3, 5, or 7 live neighbors',
      'Survives with 1, 3, 5, or 7 live neighbors',
      'Every finite pattern eventually replicates itself',
    ],
    simType: '2d',
    birth: [1, 3, 5, 7],
    survive: [1, 3, 5, 7],
    color: '#f472b6',
    notation: 'B1357/S1357',
    startEmpty: true,
    defaultDelay: 80,
  },
  {
    id: 'brians-brain',
    name: "Brian's Brain",
    category: '3-State CA',
    description:
      'Cells cycle: dead → alive → dying → dead. Nearly every random initial condition produces gliders. A born cell needs exactly 2 alive neighbors.',
    rules: [
      'A dead cell with exactly 2 alive neighbors becomes alive',
      'An alive cell becomes dying (refractory state)',
      'A dying cell becomes dead',
    ],
    simType: 'brain',
    color: '#60a5fa',
    notation: 'B2 / dying / dead',
    startEmpty: true,
  },
  {
    id: 'langtons-ant',
    name: "Langton's Ant",
    category: 'Ant Automaton',
    description:
      "An ant traverses a grid: turn right on a white cell, turn left on black, then flip. From apparent chaos emerges a periodic 'highway' after ~10,000 steps.",
    rules: [
      'On a white cell: turn right 90°, flip to black, step forward',
      'On a black cell: turn left 90°, flip to white, step forward',
    ],
    simType: 'ant',
    antTurns: [1, -1],
    antStateColors: ['#0c0c1e', '#c084fc'],
    color: '#c084fc',
  },
  {
    id: 'langtons-ant-rrll',
    name: 'Ant · RRLL',
    category: 'Ant Automaton',
    description:
      '4-color Langton variant. Two right-turns then two left-turns produce intricate symmetric spirals and structured growth.',
    rules: [
      'Four cell states cycle: 0, 1, 2, 3',
      'On state 0 or 1: turn right 90°',
      'On state 2 or 3: turn left 90°',
      'Cell advances to the next state on each visit',
    ],
    simType: 'ant',
    antTurns: [1, 1, -1, -1],
    antStateColors: ['#0c0c1e', '#c084fc', '#818cf8', '#60a5fa'],
    color: '#f0abfc',
  },
  {
    id: 'rule-30',
    name: 'Rule 30',
    category: '1D Automaton',
    description:
      "Wolfram's famously chaotic 1D automaton. The center column passes all randomness tests and was used as Mathematica's built-in random number generator.",
    rules: [
      '111→0  110→0  101→0  100→1',
      '011→1  010→1  001→1  000→0',
      'New cell = XOR(left, center OR right)',
    ],
    simType: '1d',
    rule: 30,
    color: '#f87171',
    notation: 'Rule 30',
    defaultDelay: 80,
  },
  {
    id: 'rule-110',
    name: 'Rule 110',
    category: '1D Automaton',
    description:
      'Proven Turing complete, capable of universal computation. Complex glider-like structures emerge, collide, and interact.',
    rules: [
      '111→0  110→1  101→1  100→0',
      '011→1  010→1  001→1  000→0',
      'Turing-complete: can simulate any computation',
    ],
    simType: '1d',
    rule: 110,
    color: '#fb923c',
    notation: 'Rule 110',
    defaultDelay: 80,
  },
  {
    id: 'rule-90',
    name: 'Rule 90',
    category: '1D Automaton',
    description:
      "Generates a perfect Sierpiński triangle from a single active cell. Equivalent to Pascal's triangle modulo 2.",
    rules: [
      '111→0  110→1  101→0  100→1',
      '011→1  010→0  001→1  000→0',
      'New cell = left XOR right (Pascal\'s triangle mod 2)',
    ],
    simType: '1d',
    rule: 90,
    color: '#34d399',
    notation: 'Rule 90',
    defaultDelay: 80,
  },
] as const satisfies readonly RulesetDef[];

type RuleId = (typeof RULESETS)[number]['id'];

const RULESET_MAP = new Map<RuleId, RulesetDef>(RULESETS.map(r => [r.id, r]));
const RULESET_GROUPS = Array.from(
  RULESETS.reduce((groups, ruleset) => {
    const existing = groups.get(ruleset.category);
    if (existing) existing.push(ruleset);
    else groups.set(ruleset.category, [ruleset]);
    return groups;
  }, new Map<string, RulesetDef[]>()),
  ([category, rulesets]) => ({ category, rulesets }),
);

// Presets
interface Preset { label: string; sub: string; cells: [number, number][]; }

const PRESETS_BY_RULESET: Record<string, Preset[]> = {
  conway: [
    { label: 'Glider',          sub: 'Spaceship',   cells: [[0,1],[1,2],[2,0],[2,1],[2,2]] },
    {
      label: 'Glider Gun', sub: 'Gun',
      cells: [
        [0,24],[1,22],[1,24],[2,12],[2,13],[2,20],[2,21],[2,34],[2,35],
        [3,11],[3,15],[3,20],[3,21],[3,34],[3,35],
        [4,0],[4,1],[4,10],[4,16],[4,20],[4,21],
        [5,0],[5,1],[5,10],[5,14],[5,16],[5,17],[5,22],[5,24],
        [6,10],[6,16],[6,24],[7,11],[7,15],[8,12],[8,13],
      ],
    },
    {
      label: 'Pulsar', sub: 'Osc · P3',
      cells: [
        [0,2],[0,3],[0,4],[0,8],[0,9],[0,10],
        [2,0],[2,5],[2,7],[2,12],[3,0],[3,5],[3,7],[3,12],
        [4,0],[4,5],[4,7],[4,12],[5,2],[5,3],[5,4],[5,8],[5,9],[5,10],
        [7,2],[7,3],[7,4],[7,8],[7,9],[7,10],
        [8,0],[8,5],[8,7],[8,12],[9,0],[9,5],[9,7],[9,12],
        [10,0],[10,5],[10,7],[10,12],[12,2],[12,3],[12,4],[12,8],[12,9],[12,10],
      ],
    },
    { label: 'Pentadecathlon', sub: 'Osc · P15', cells: [[0,1],[1,1],[2,0],[2,2],[3,1],[4,1],[5,1],[6,1],[7,0],[7,2],[8,1],[9,1]] },
    { label: 'LWSS',           sub: 'Spaceship',  cells: [[0,1],[0,2],[0,3],[0,4],[1,0],[1,4],[2,4],[3,0],[3,3]] },
    { label: 'Blinker',        sub: 'Osc · P2',   cells: [[0,0],[0,1],[0,2]] },
    { label: 'Beacon',         sub: 'Osc · P2',   cells: [[0,0],[0,1],[1,0],[1,1],[2,2],[2,3],[3,2],[3,3]] },
    { label: 'R-Pentomino',    sub: 'Methuselah', cells: [[0,1],[0,2],[1,0],[1,1],[2,1]] },
    { label: 'Diehard',        sub: 'Methuselah', cells: [[0,6],[1,0],[1,1],[2,1],[2,5],[2,6],[2,7]] },
    { label: 'Acorn',          sub: 'Methuselah', cells: [[0,1],[1,3],[2,0],[2,1],[2,4],[2,5],[2,6]] },
  ],

  highlife: [
    { label: 'Glider',      sub: 'Spaceship',  cells: [[0,1],[1,2],[2,0],[2,1],[2,2]] },
    { label: 'LWSS',        sub: 'Spaceship',  cells: [[0,1],[0,2],[0,3],[0,4],[1,0],[1,4],[2,4],[3,0],[3,3]] },
    {
      label: 'Pulsar', sub: 'Osc · P3',
      cells: [
        [0,2],[0,3],[0,4],[0,8],[0,9],[0,10],
        [2,0],[2,5],[2,7],[2,12],[3,0],[3,5],[3,7],[3,12],
        [4,0],[4,5],[4,7],[4,12],[5,2],[5,3],[5,4],[5,8],[5,9],[5,10],
        [7,2],[7,3],[7,4],[7,8],[7,9],[7,10],
        [8,0],[8,5],[8,7],[8,12],[9,0],[9,5],[9,7],[9,12],
        [10,0],[10,5],[10,7],[10,12],[12,2],[12,3],[12,4],[12,8],[12,9],[12,10],
      ],
    },
    { label: 'R-Pentomino', sub: 'Methuselah', cells: [[0,1],[0,2],[1,0],[1,1],[2,1]] },
    { label: 'Acorn',       sub: 'Methuselah', cells: [[0,1],[1,3],[2,0],[2,1],[2,4],[2,5],[2,6]] },
    {
      label: 'Replicator seed', sub: 'Explosive',
      cells: [
        [0,3],[0,4],[0,5],[1,2],[1,5],[2,1],[2,5],[3,1],[3,4],
        [4,1],[4,2],[4,3],[5,0],[6,0],[6,1],[6,2],[6,3],
      ],
    },
  ],

  'day-and-night': [
    { label: 'Block',     sub: 'Still life', cells: [[0,0],[0,1],[1,0],[1,1]] },
    { label: 'Pond',      sub: 'Still life', cells: [[0,1],[0,2],[1,0],[1,3],[2,0],[2,3],[3,1],[3,2]] },
    {
      label: 'Fat cross', sub: 'Seed',
      cells: [
        [0,2],[1,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3],[2,4],
        [3,1],[3,2],[3,3],[4,2],
      ],
    },
    {
      label: '4×4 block', sub: 'Seed',
      cells: [
        [0,0],[0,1],[0,2],[0,3],[1,0],[1,1],[1,2],[1,3],
        [2,0],[2,1],[2,2],[2,3],[3,0],[3,1],[3,2],[3,3],
      ],
    },
    {
      label: 'Diamond',   sub: 'Seed',
      cells: [[0,3],[1,1],[1,2],[1,4],[1,5],[2,0],[2,6],[3,1],[3,2],[3,4],[3,5],[4,3]],
    },
  ],

  maze: [
    { label: '2×2 block', sub: 'Small seed', cells: [[0,0],[0,1],[1,0],[1,1]] },
    { label: 'Plus',      sub: 'Small seed', cells: [[0,1],[1,0],[1,1],[1,2],[2,1]] },
    {
      label: '3×3 block', sub: 'Medium seed',
      cells: [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]],
    },
    {
      label: '5-cell row', sub: 'Line seed',
      cells: [[0,0],[0,1],[0,2],[0,3],[0,4]],
    },
    {
      label: 'Diagonal',  sub: 'Line seed',
      cells: [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5]],
    },
  ],

  seeds: [
    { label: '3-cell row',  sub: 'Seed', cells: [[0,0],[0,1],[0,2]] },
    { label: '5-cell row',  sub: 'Seed', cells: [[0,0],[0,1],[0,2],[0,3],[0,4]] },
    { label: '2×2 block',   sub: 'Seed', cells: [[0,0],[0,1],[1,0],[1,1]] },
    {
      label: 'Diamond', sub: 'Seed',
      cells: [[0,2],[1,0],[1,4],[2,2]],
    },
    {
      label: 'Cross',   sub: 'Seed',
      cells: [[0,2],[1,0],[1,1],[1,3],[1,4],[2,2]],
    },
  ],

  replicator: [
    { label: 'Single cell', sub: 'Seed',       cells: [[0,0]] },
    { label: 'Block',       sub: '2×2 seed',   cells: [[0,0],[0,1],[1,0],[1,1]] },
    { label: 'Blinker',     sub: 'Row of 3',   cells: [[0,0],[0,1],[0,2]] },
    { label: 'R-Pentomino', sub: 'Methuselah', cells: [[0,1],[0,2],[1,0],[1,1],[2,1]] },
    { label: 'Glider',      sub: 'Spaceship',  cells: [[0,1],[1,2],[2,0],[2,1],[2,2]] },
  ],

  'brians-brain': [
    { label: 'Row of 3',  sub: 'Seed', cells: [[0,0],[0,1],[0,2]] },
    { label: 'Square',    sub: 'Seed', cells: [[0,0],[0,1],[1,0],[1,1]] },
    { label: 'Diamond',   sub: 'Seed', cells: [[0,1],[1,0],[1,2],[2,1]] },
    { label: 'Cross',     sub: 'Seed', cells: [[0,1],[1,0],[1,1],[1,2],[2,1]] },
    { label: 'Row of 6',  sub: 'Seed', cells: [[0,0],[0,1],[0,2],[0,3],[0,4],[0,5]] },
    { label: '3×3 block', sub: 'Seed', cells: [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]] },
  ],
};

// Helpers
const BG_RGB: Rgb = hexToRgb(BG_HEX);

function wrap(value: number, max: number) {
  return ((value % max) + max) % max;
}

function indexAt(x: number, y: number) {
  return y * COLS + x;
}

function hexToRgb(hex: string): Rgb {
  const v = Number.parseInt(hex.replace('#', ''), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

function mixRgb(fg: Rgb, bg: Rgb, t: number): Rgb {
  return [
    Math.round(fg[0] * t + bg[0] * (1 - t)),
    Math.round(fg[1] * t + bg[1] * (1 - t)),
    Math.round(fg[2] * t + bg[2] * (1 - t)),
  ];
}

function makeAgeColors(hexColor: string): [string, string, string, string] {
  const accent = hexToRgb(hexColor);
  const white: Rgb  = [248, 254, 252];
  const toRgb = ([r, g, b]: Rgb) => `rgb(${r},${g},${b})`;
  return [
    toRgb(mixRgb(white, accent, 0.80)),   // newborn : near-white
    toRgb(mixRgb(white, accent, 0.45)),   // young   : lighter
    hexColor,                               // mature  : full accent
    toRgb(mixRgb(accent, BG_RGB, 0.60)),  // old     : darker
  ];
}

function ageBucket(age: number): 0 | 1 | 2 | 3 {
  if (age <= 1) return 0;
  if (age <= 3) return 1;
  if (age <= 9) return 2;
  return 3;
}

function countPop(grid: Uint8Array): number {
  let n = 0;
  for (let i = 0; i < grid.length; i++) if (grid[i] === 1) n++;
  return n;
}

// SimState
function createSimState(): SimState {
  const offscreen = new OffscreenCanvas(COLS, ROWS);
  const offCtx = offscreen.getContext('2d', { alpha: false })!;
  offCtx.imageSmoothingEnabled = false;
  return {
    grid: new Uint8Array(COLS * ROWS),
    ageGrid: new Uint16Array(COLS * ROWS),
    ants: [],
    row: new Uint8Array(COLS),
    rowHistory: new Uint8Array(COLS * ROWS),
    rowIdx: 0,
    generation: 0,
    offscreen,
    offCtx,
  };
}

function recreateOffscreen(sim: SimState) {
  const offscreen = new OffscreenCanvas(COLS, ROWS);
  const offCtx = offscreen.getContext('2d', { alpha: false })!;
  offCtx.imageSmoothingEnabled = false;
  sim.offscreen = offscreen;
  sim.offCtx    = offCtx;
}

// Init
function initRandom(density: number): Uint8Array {
  const grid = new Uint8Array(COLS * ROWS);
  for (let i = 0; i < grid.length; i++) grid[i] = Math.random() < density ? 1 : 0;
  return grid;
}

function initAnt() {
  return { grid: new Uint8Array(COLS * ROWS), ants: [{ x: CENTER_X, y: CENTER_Y, dir: 0 }] };
}

function init1D() {
  const row = new Uint8Array(COLS);
  const rowHistory = new Uint8Array(COLS * ROWS);
  row[Math.floor(COLS / 2)] = 1;
  rowHistory.set(row, 0);
  return { row, rowHistory, rowIdx: 1 };
}

function buildSimState(ruleset: RulesetDef, mode: 'reset' | 'clear'): SimState {
  const sim = createSimState();

  if (ruleset.simType === '2d' || ruleset.simType === 'brain') {
    const empty = mode === 'clear' || !!ruleset.startEmpty;
    const density = ruleset.defaultDensity ?? DEFAULT_DENSITY / 100;
    sim.grid = empty ? new Uint8Array(COLS * ROWS) : initRandom(density);
    sim.ageGrid = new Uint16Array(COLS * ROWS);
    if (!empty) for (let i = 0; i < sim.grid.length; i++) sim.ageGrid[i] = sim.grid[i] ? 1 : 0;
    return sim;
  }

  if (ruleset.simType === 'ant') {
    const { ants } = initAnt();
    sim.ants = ants;
    sim.grid = new Uint8Array(COLS * ROWS);
    return sim;
  }

  if (mode === 'clear') return sim;
  const init = init1D();
  sim.row = init.row;
  sim.rowHistory = init.rowHistory;
  sim.rowIdx = init.rowIdx;
  return sim;
}

function buildFromPreset(cells: [number, number][]): Pick<SimState, 'grid' | 'ageGrid'> {
  const grid = new Uint8Array(COLS * ROWS);
  if (cells.length) {
    const minR = Math.min(...cells.map(([r]) => r));
    const maxR = Math.max(...cells.map(([r]) => r));
    const minC = Math.min(...cells.map(([, c]) => c));
    const maxC = Math.max(...cells.map(([, c]) => c));
    const offR = Math.floor((ROWS - (maxR - minR + 1)) / 2) - minR;
    const offC = Math.floor((COLS - (maxC - minC + 1)) / 2) - minC;
    for (const [r, c] of cells) {
      const nr = r + offR, nc = c + offC;
      if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS) grid[indexAt(nc, nr)] = 1;
    }
  }
  const ageGrid = new Uint16Array(COLS * ROWS);
  for (let i = 0; i < grid.length; i++) ageGrid[i] = grid[i] ? 1 : 0;
  return { grid, ageGrid };
}

// Step
function step2D(
  grid: Uint8Array,
  ageGrid: Uint16Array,
  birth: Set<number>,
  survive: Set<number>,
): [Uint8Array, Uint16Array] {
  const next     = new Uint8Array(grid.length);
  const nextAges = new Uint16Array(grid.length);
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      let n = 0;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          n += grid[indexAt(wrap(x + dx, COLS), wrap(y + dy, ROWS))] > 0 ? 1 : 0;
        }
      const idx = indexAt(x, y);
      const alive: 0 | 1 = grid[idx] > 0
        ? (survive.has(n) ? 1 : 0)
        : (birth.has(n)   ? 1 : 0);
      next[idx]     = alive;
      nextAges[idx] = alive ? Math.min(ageGrid[idx] + 1, MAX_AGE) : 0;
    }
  }
  return [next, nextAges];
}

function stepBrain(grid: Uint8Array): Uint8Array {
  const next = new Uint8Array(grid.length);
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      let alive = 0;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          alive += grid[indexAt(wrap(x + dx, COLS), wrap(y + dy, ROWS))] === 1 ? 1 : 0;
        }
      const idx = indexAt(x, y);
      if      (grid[idx] === 1) next[idx] = 2;
      else if (grid[idx] === 2) next[idx] = 0;
      else                       next[idx] = alive === 2 ? 1 : 0;
    }
  }
  return next;
}

function stepAnt(grid: Uint8Array, ants: Ant[], turns: number[], nStates: number) {
  for (const ant of ants) {
    const idx = indexAt(ant.x, ant.y);
    const state = grid[idx];
    ant.dir   = wrap(ant.dir + (turns[state] ?? 1), 4);
    grid[idx] = (state + 1) % nStates;
    if      (ant.dir === 0) ant.y = wrap(ant.y - 1, ROWS);
    else if (ant.dir === 1) ant.x = wrap(ant.x + 1, COLS);
    else if (ant.dir === 2) ant.y = wrap(ant.y + 1, ROWS);
    else                     ant.x = wrap(ant.x - 1, COLS);
  }
}

function step1D(row: Uint8Array, rule: number): Uint8Array {
  const next = new Uint8Array(COLS);
  for (let x = 0; x < COLS; x++) {
    const pattern = (row[wrap(x - 1, COLS)] << 2) | (row[x] << 1) | row[wrap(x + 1, COLS)];
    next[x] = (rule >> pattern) & 1;
  }
  return next;
}

// Render
/** Viewport render for 2D / brain / ant rules */
function renderViewport(
  canvas: HTMLCanvasElement,
  sim: SimState,
  ruleset: RulesetDef,
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
  const cs  = cellSizeCss * dpr;

  const startC = Math.floor(viewX) - 1;
  const endC   = Math.ceil(viewX + W / cs) + 1;
  const startR = Math.floor(viewY) - 1;
  const endR   = Math.ceil(viewY + H / cs) + 1;

  const showCellBg = cs >= 6;
  const gap        = showCellBg ? 1 : 0;
  const cellW      = Math.ceil(cs) - gap;

  ctx.fillStyle = showCellBg ? BG_COLOR : DEAD_COLOR;
  ctx.fillRect(0, 0, W, H);

  if (ruleset.simType === '2d') {
    if (showCellBg) {
      ctx.fillStyle = DEAD_COLOR;
      ctx.beginPath();
      for (let r = startR; r <= endR; r++)
        for (let c = startC; c <= endC; c++)
          if (!sim.grid[indexAt(wrap(c, COLS), wrap(r, ROWS))])
            ctx.rect(Math.floor((c - viewX) * cs) + 1, Math.floor((r - viewY) * cs) + 1, Math.ceil(cs) - 1, Math.ceil(cs) - 1);
      ctx.fill();
    }

    if (useAgeColor) {
      const colors = makeAgeColors(ruleset.color);
      const buckets: [number, number][][] = [[], [], [], []];
      for (let r = startR; r <= endR; r++)
        for (let c = startC; c <= endC; c++) {
          const i = indexAt(wrap(c, COLS), wrap(r, ROWS));
          if (sim.grid[i])
            buckets[ageBucket(sim.ageGrid[i])].push([Math.floor((c - viewX) * cs), Math.floor((r - viewY) * cs)]);
        }
      for (let b = 0; b < 4; b++) {
        if (!buckets[b].length) continue;
        ctx.fillStyle = colors[b];
        ctx.beginPath();
        for (const [x, y] of buckets[b]) ctx.rect(x + gap, y + gap, cellW, cellW);
        ctx.fill();
      }
    } else {
      ctx.fillStyle = ruleset.color;
      ctx.beginPath();
      for (let r = startR; r <= endR; r++)
        for (let c = startC; c <= endC; c++)
          if (sim.grid[indexAt(wrap(c, COLS), wrap(r, ROWS))])
            ctx.rect(Math.floor((c - viewX) * cs) + gap, Math.floor((r - viewY) * cs) + gap, cellW, cellW);
      ctx.fill();
    }

  } else if (ruleset.simType === 'brain') {
    const accRgb  = hexToRgb(ruleset.color);
    const dieRgb  = mixRgb(accRgb, BG_RGB, 0.35);
    const dieCol  = `rgb(${dieRgb[0]},${dieRgb[1]},${dieRgb[2]})`;

    ctx.fillStyle = ruleset.color;
    ctx.beginPath();
    for (let r = startR; r <= endR; r++)
      for (let c = startC; c <= endC; c++)
        if (sim.grid[indexAt(wrap(c, COLS), wrap(r, ROWS))] === 1)
          ctx.rect(Math.floor((c - viewX) * cs) + gap, Math.floor((r - viewY) * cs) + gap, cellW, cellW);
    ctx.fill();

    ctx.fillStyle = dieCol;
    ctx.beginPath();
    for (let r = startR; r <= endR; r++)
      for (let c = startC; c <= endC; c++)
        if (sim.grid[indexAt(wrap(c, COLS), wrap(r, ROWS))] === 2)
          ctx.rect(Math.floor((c - viewX) * cs) + gap, Math.floor((r - viewY) * cs) + gap, cellW, cellW);
    ctx.fill();

  } else if (ruleset.simType === 'ant') {
    const colors = ruleset.antStateColors ?? ['#0c0c1e', ruleset.color];
    for (let s = 1; s < colors.length; s++) {
      ctx.fillStyle = colors[s];
      ctx.beginPath();
      for (let r = startR; r <= endR; r++)
        for (let c = startC; c <= endC; c++)
          if (sim.grid[indexAt(wrap(c, COLS), wrap(r, ROWS))] === s)
            ctx.rect(Math.floor((c - viewX) * cs) + gap, Math.floor((r - viewY) * cs) + gap, cellW, cellW);
      ctx.fill();
    }
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    for (const ant of sim.ants)
      ctx.rect(Math.floor((ant.x - viewX) * cs) + gap, Math.floor((ant.y - viewY) * cs) + gap, cellW, cellW);
    ctx.fill();
  }

  if (showGrid && cs >= 8) {
    ctx.strokeStyle = GRID_LINE_COLOR;
    ctx.lineWidth   = 1;
    ctx.beginPath();
    for (let c = startC; c <= endC + 1; c++) {
      const x = Math.round((c - viewX) * cs) + 0.5;
      ctx.moveTo(x, 0); ctx.lineTo(x, H);
    }
    for (let r = startR; r <= endR + 1; r++) {
      const y = Math.round((r - viewY) * cs) + 0.5;
      ctx.moveTo(0, y); ctx.lineTo(W, y);
    }
    ctx.stroke();
  }
}

/** Stretch-to-fill render for 1D rules */
function render1D(canvas: HTMLCanvasElement, sim: SimState, ruleset: RulesetDef) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const image  = new ImageData(COLS, ROWS);
  const { data } = image;
  const accent = hexToRgb(ruleset.color);

  for (let py = 0; py < ROWS; py++) {
    const histRow = ((sim.rowIdx - ROWS + py + ROWS) % ROWS) * COLS;
    for (let px = 0; px < COLS; px++) {
      const offset = (py * COLS + px) * 4;
      const rgb    = sim.rowHistory[histRow + px] ? accent : BG_RGB;
      data[offset]     = rgb[0];
      data[offset + 1] = rgb[1];
      data[offset + 2] = rgb[2];
      data[offset + 3] = 255;
    }
  }

  sim.offCtx.putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(sim.offscreen, 0, 0, canvas.width, canvas.height);
}

// Component
export default function CellularAutomata() {
  const containerRef   = useRef<HTMLDivElement>(null);
  const canvasRef      = useRef<HTMLCanvasElement>(null);
  const simRef         = useRef<SimState>(createSimState());
  const rafRef         = useRef(0);
  const runningRef     = useRef(false);
  const speedRef       = useRef(1);
  const delayRef       = useRef((RULESETS[0] as RulesetDef).defaultDelay ?? 0);
  const activeRuleRef  = useRef<RulesetDef>(RULESETS[0] as RulesetDef);
  const lastStepRef    = useRef(0);
  const ageColorRef    = useRef(true);
  const sparkBufRef    = useRef<number[]>([]);

  // Viewport (2D/brain/ant rules only)
  const viewRef     = useRef({ x: COLS / 2 - 30, y: ROWS / 2 - 20 });
  const cellSizeRef = useRef(INIT_CELL_PX);

  // Drag state: isPan=true → panning, isPan=false → painting cells
  const dragRef = useRef<{
    isPan: boolean;
    paintState: 0 | 1;
    paintedCells: Set<number>;
  } | null>(null);

  const [activeRuleId, setActiveRuleId] = useState<RuleId>(RULESETS[0].id);
  const [running,      setRunning]      = useState(false);
  const [speed,        setSpeed]        = useState(1);
  const [delayMs,      setDelayMs]      = useState((RULESETS[0] as RulesetDef).defaultDelay ?? 0);
  const [generation,   setGeneration]   = useState(0);
  const [population,   setPopulation]   = useState(0);
  const [sparkData,    setSparkData]    = useState<number[]>([]);
  const [ageColor,     setAgeColor]     = useState(true);
  const [showGraph,    setShowGraph]    = useState(false);
  const [activePreset, setActivePreset] = useState(-1);
  const [showExport,   setShowExport]   = useState(false);
  const [infoCollapsed, setInfoCollapsed] = useState(false);

  const activeRule    = RULESET_MAP.get(activeRuleId) ?? (RULESETS[0] as RulesetDef);
  const is1D          = activeRule.simType === '1d';
  const canPaint      = activeRule.simType === '2d' || activeRule.simType === 'brain';
  const showPop       = activeRule.simType === '2d' || activeRule.simType === 'brain';
  const showAgeColor  = activeRule.simType === '2d';

  const presets     = useMemo(() => PRESETS_BY_RULESET[activeRuleId] ?? [], [activeRuleId]);
  const showPresets = (activeRule.simType === '2d' || activeRule.simType === 'brain') && presets.length > 0;

  const { isFullscreen, toggleFullscreen } = useFullscreen(containerRef);

  useEffect(() => { ageColorRef.current = ageColor; }, [ageColor]);

  // Draw
  const drawFrame = useCallback(() => {
    const canvas  = canvasRef.current;
    const sim     = simRef.current;
    if (!canvas || canvas.width === 0 || canvas.height === 0) return;

    if (activeRuleRef.current.simType === '1d') {
      render1D(canvas, sim, activeRuleRef.current);
    } else {
      renderViewport(
        canvas, sim, activeRuleRef.current,
        viewRef.current.x, viewRef.current.y,
        cellSizeRef.current,
        !runningRef.current,
        ageColorRef.current,
      );
    }
  }, []);

  // Step
  const stepSimulation = useCallback((steps: number) => {
    const sim     = simRef.current;
    const ruleset = activeRuleRef.current;

    for (let i = 0; i < steps; i++) {
      if (ruleset.simType === '2d') {
        const [ng, na] = step2D(sim.grid, sim.ageGrid, new Set(ruleset.birth ?? []), new Set(ruleset.survive ?? []));
        sim.grid    = ng;
        sim.ageGrid = na;
      } else if (ruleset.simType === 'brain') {
        sim.grid = stepBrain(sim.grid);
      } else if (ruleset.simType === 'ant') {
        stepAnt(sim.grid, sim.ants, ruleset.antTurns ?? [1, -1], ruleset.antTurns?.length ?? 2);
      } else {
        const next = step1D(sim.row, ruleset.rule ?? 110);
        sim.row = next;
        sim.rowHistory.set(next, sim.rowIdx * COLS);
        sim.rowIdx = (sim.rowIdx + 1) % ROWS;
      }
      sim.generation++;
    }

    setGeneration(sim.generation);

    if (ruleset.simType === '2d' || ruleset.simType === 'brain') {
      const pop = countPop(sim.grid);
      sparkBufRef.current = [...sparkBufRef.current.slice(-(SPARK_LEN - 1)), pop];
      setPopulation(pop);
      setSparkData([...sparkBufRef.current]);
    }
  }, []);

  // RAF loop
  const stopLoop = useCallback(() => {
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = 0; }
  }, []);

  const startLoop = useCallback(() => {
    if (rafRef.current) return;
    const frame = (now: number) => {
      const delay = delayRef.current;
      if (delay === 0 || now - lastStepRef.current >= delay) {
        stepSimulation(speedRef.current);
        drawFrame();
        lastStepRef.current = now;
      }
      rafRef.current = requestAnimationFrame(frame);
    };
    rafRef.current = requestAnimationFrame(frame);
  }, [drawFrame, stepSimulation]);

  // Init helpers
  const centerViewport = useCallback((ruleset: RulesetDef) => {
    const canvas = canvasRef.current;
    const w = canvas ? canvas.getBoundingClientRect().width  : 800;
    const h = canvas ? canvas.getBoundingClientRect().height : 600;
    cellSizeRef.current = ruleset.simType === '1d' ? INIT_CELL_PX_1D : INIT_CELL_PX;
    viewRef.current = {
      x: COLS / 2 - w / (2 * cellSizeRef.current),
      y: ROWS / 2 - h / (2 * cellSizeRef.current),
    };
  }, []);

  const initSim = useCallback((ruleset: RulesetDef, mode: 'reset' | 'clear') => {
    const wasRunning = runningRef.current;
    stopLoop();
    activeRuleRef.current = ruleset;
    simRef.current        = buildSimState(ruleset, mode);
    sparkBufRef.current   = [];
    setGeneration(0);
    setPopulation(0);
    setSparkData([]);
    setActivePreset(-1);
    centerViewport(ruleset);
    drawFrame();
    if (wasRunning) startLoop();
  }, [drawFrame, startLoop, stopLoop, centerViewport]);

  // Handlers
  const handleRulesetSelect = useCallback((ruleset: RulesetDef) => {
    const nextDelay = ruleset.defaultDelay ?? 0;
    delayRef.current = nextDelay;
    setDelayMs(nextDelay);
    setActiveRuleId(ruleset.id as RuleId);
    initSim(ruleset, 'reset');
  }, [initSim]);

  const handleReset = useCallback(() => initSim(activeRuleRef.current, 'reset'), [initSim]);

  const handleStep = useCallback(() => {
    if (runningRef.current) return;
    stepSimulation(1);
    drawFrame();
  }, [drawFrame, stepSimulation]);

  const handleToggleRunning = useCallback(() => setRunning(v => !v), []);
  const handleSpeedChange   = useCallback((v: number) => { speedRef.current = v; setSpeed(v); }, []);
  const handleDelayChange   = useCallback((v: number) => { delayRef.current = v; setDelayMs(v); }, []);

  const handleApplyPreset = useCallback((i: number) => {
    const rulePresets = PRESETS_BY_RULESET[activeRuleId] ?? [];
    const preset = rulePresets[i];
    if (!preset) return;
    const { grid, ageGrid } = buildFromPreset(preset.cells);
    const sim = simRef.current;
    sim.grid       = grid;
    sim.ageGrid    = ageGrid;
    sim.generation = 0;
    sparkBufRef.current = [];
    setGeneration(0);
    setPopulation(countPop(grid));
    setSparkData([]);
    setActivePreset(i);
    setRunning(false);
    // Reset zoom and center
    cellSizeRef.current = INIT_CELL_PX;
    const canvas = canvasRef.current;
    if (canvas) {
      const rect = canvas.getBoundingClientRect();
      viewRef.current = {
        x: COLS / 2 - rect.width  / (2 * INIT_CELL_PX),
        y: ROWS / 2 - rect.height / (2 * INIT_CELL_PX),
      };
    }
    drawFrame();
  }, [drawFrame, activeRuleId]);

  const handleCenter = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    viewRef.current = {
      x: COLS / 2 - rect.width  / (2 * cellSizeRef.current),
      y: ROWS / 2 - rect.height / (2 * cellSizeRef.current),
    };
  }, []);

  // Running effect
  useEffect(() => {
    runningRef.current = running;
    if (running) startLoop();
    else { stopLoop(); drawFrame(); }
  }, [running, startLoop, stopLoop, drawFrame]);

  // Resize observer
  useEffect(() => {
    const container = containerRef.current;
    const canvas    = canvasRef.current;
    if (!container || !canvas) return;
    let firstResize = true;

    const obs = new ResizeObserver((entries) => {
      for (const e of entries) {
        const { width, height } = e.contentRect;
        if (firstResize) {
          firstResize = false;
          viewRef.current = {
            x: COLS / 2 - width  / (2 * cellSizeRef.current),
            y: ROWS / 2 - height / (2 * cellSizeRef.current),
          };
        }
        const dpr = window.devicePixelRatio || 1;
        const w   = Math.max(1, Math.round(width  * dpr));
        const h   = Math.max(1, Math.round(height * dpr));
        if (canvas.width !== w || canvas.height !== h) {
          canvas.width  = w;
          canvas.height = h;
          recreateOffscreen(simRef.current);
          drawFrame();
        }
      }
    });

    obs.observe(container);
    return () => obs.disconnect();
  }, [drawFrame]);

  // Bootstrap
  useEffect(() => {
    initSim(activeRuleRef.current, 'reset');
    return () => stopLoop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Wheel zoom
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (e: WheelEvent) => {
      if (activeRuleRef.current.simType === '1d') return;
      e.preventDefault();
      const rect   = canvas.getBoundingClientRect();
      const mx     = e.clientX - rect.left;
      const my     = e.clientY - rect.top;
      const oldCs  = cellSizeRef.current;
      const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
      const newCs  = Math.max(MIN_CELL_PX, Math.min(MAX_CELL_PX, oldCs * factor));
      viewRef.current.x += mx * (1 / oldCs - 1 / newCs);
      viewRef.current.y += my * (1 / oldCs - 1 / newCs);
      cellSizeRef.current = newCs;
      if (!runningRef.current) drawFrame();
    };
    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', onWheel);
  }, [drawFrame]);

  // Pointer: paint (left) / pan (right/middle/any for non-paintable)
  const getCellAt = useCallback((clientX: number, clientY: number): [number, number] => {
    const canvas = canvasRef.current!;
    const rect   = canvas.getBoundingClientRect();
    const c = Math.floor(viewRef.current.x + (clientX - rect.left) / cellSizeRef.current);
    const r = Math.floor(viewRef.current.y + (clientY - rect.top)  / cellSizeRef.current);
    return [wrap(r, ROWS), wrap(c, COLS)];
  }, []);

  const onPointerDown = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    const paintable = activeRuleRef.current.simType === '2d' || activeRuleRef.current.simType === 'brain';

    if (!paintable || e.button === 1 || e.button === 2) {
      dragRef.current = { isPan: true, paintState: 0, paintedCells: new Set() };
      canvasRef.current!.style.cursor = 'grabbing';
      return;
    }
    if (e.button !== 0) return;

    const [r, c] = getCellAt(e.clientX, e.clientY);
    const idx    = indexAt(c, r);
    const newSt: 0 | 1 = simRef.current.grid[idx] ? 0 : 1;
    simRef.current.grid[idx]    = newSt;
    simRef.current.ageGrid[idx] = newSt;
    dragRef.current = { isPan: false, paintState: newSt, paintedCells: new Set([idx]) };
    setActivePreset(-1);
    setPopulation(countPop(simRef.current.grid));
    if (!runningRef.current) drawFrame();
  }, [getCellAt, drawFrame]);

  const onPointerMove = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    if (!drag) return;
    if (drag.isPan) {
      viewRef.current.x -= e.movementX / cellSizeRef.current;
      viewRef.current.y -= e.movementY / cellSizeRef.current;
      if (!runningRef.current) drawFrame();
      return;
    }
    const [r, c] = getCellAt(e.clientX, e.clientY);
    const idx    = indexAt(c, r);
    if (!drag.paintedCells.has(idx)) {
      drag.paintedCells.add(idx);
      simRef.current.grid[idx]    = drag.paintState;
      simRef.current.ageGrid[idx] = drag.paintState;
      setPopulation(countPop(simRef.current.grid));
      if (!runningRef.current) drawFrame();
    }
  }, [getCellAt, drawFrame]);

  const onPointerUp = useCallback(() => {
    dragRef.current = null;
    if (canvasRef.current) {
      const p = activeRuleRef.current.simType === '2d' || activeRuleRef.current.simType === 'brain';
      canvasRef.current.style.cursor = p ? 'crosshair' : 'grab';
    }
  }, []);

  const onContextMenu = useCallback((e: React.MouseEvent) => e.preventDefault(), []);

  // Keyboard shortcuts
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement) return;
      if (e.code === 'Space') { e.preventDefault(); handleToggleRunning(); }
      if (e.code === 'KeyR')  { e.preventDefault(); handleReset(); }
      if (e.code === 'KeyF')  { e.preventDefault(); toggleFullscreen(); }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [handleToggleRunning, handleReset, toggleFullscreen]);

  const groupedRulesets = useMemo(() => RULESET_GROUPS, []);
  const ageColors       = useMemo(() => makeAgeColors(activeRule.color), [activeRule.color]);

  // JSX
  return (
    <div
      ref={containerRef}
      className={styles.container}
      style={{ '--accent': activeRule.color } as React.CSSProperties}
    >
      <canvas
        ref={canvasRef}
        className={styles.canvas}
        style={{ cursor: canPaint ? 'crosshair' : 'grab' }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onContextMenu={onContextMenu}
      />

      {/* Info overlay */}
      <div className={styles.infoOverlay}>
        <div className={styles.infoOverlayHeader}>
          <div className={styles.infoOverlayName}>
            <span className={styles.infoOverlayDot} style={{ backgroundColor: activeRule.color }} />
            {activeRule.name}
          </div>
          <button
            className={styles.infoOverlayToggle}
            onClick={() => setInfoCollapsed(v => !v)}
            aria-label={infoCollapsed ? 'Expand' : 'Collapse'}
          >
            {infoCollapsed ? '▸' : '▾'}
          </button>
        </div>
        {!infoCollapsed && (
          <>
            <div className={styles.infoOverlayMeta}>
              <span className={styles.categoryBadge}>{activeRule.category}</span>
              {activeRule.notation && <span className={styles.notationBadge}>{activeRule.notation}</span>}
            </div>
            <ul className={styles.infoOverlayRules}>
              {activeRule.rules.map((rule, i) => (
                <li key={i} className={styles.infoOverlayRule}>{rule}</li>
              ))}
            </ul>
          </>
        )}
      </div>

      {/* Sidebar */}
      <div className={styles.sidebar}>
        <div className={styles.sidebarPanels}>

          <ControlPanel title="Rulesets">
            <div className={styles.rulesetList}>
              {groupedRulesets.map(group => (
                <div key={group.category} className={styles.rulesetSection}>
                  <div className={styles.rulesetHeader}>{group.category}</div>
                  <div className={styles.rulesetGrid}>
                    {group.rulesets.map(ruleset => (
                      <button
                        key={ruleset.id}
                        type="button"
                        className={[styles.rulesetBtn, activeRuleId === ruleset.id ? styles.rulesetBtnActive : ''].join(' ')}
                        onClick={() => handleRulesetSelect(ruleset)}
                      >
                        <span className={styles.rulesetNameRow}>
                          <span className={styles.rulesetDot} style={{ backgroundColor: ruleset.color }} aria-hidden />
                          <span className={styles.rulesetName}>{ruleset.name}</span>
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </ControlPanel>

          {showPresets && (
            <ControlPanel title="Presets">
              <div className={styles.presetGrid}>
                {presets.map((p, i) => (
                  <button
                    key={p.label}
                    className={[styles.presetBtn, activePreset === i ? styles.presetBtnActive : ''].join(' ')}
                    onClick={() => handleApplyPreset(i)}
                  >
                    <span className={styles.presetLabel}>{p.label}</span>
                    <span className={styles.presetSub}>{p.sub}</span>
                  </button>
                ))}
              </div>
            </ControlPanel>
          )}

          <ControlPanel title="Speed">
            <ControlGroup>
              <Slider label="Steps / frame" value={speed} min={1} max={20} step={1} onChange={handleSpeedChange} />
              <Slider label="Delay between frames" value={delayMs} min={0} max={500} step={10} unit="ms" onChange={handleDelayChange} />
            </ControlGroup>
          </ControlPanel>

          {showAgeColor && (
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
                    {(['Newborn', 'Young', 'Mature', 'Old'] as const).map((label, i) => (
                      <div key={label} className={styles.ageLegendRow}>
                        <span className={styles.ageSwatch} style={{ background: ageColors[i] }} />
                        <span className={styles.ageLegendLabel}>{label}</span>
                        <span className={styles.ageLegendRange}>{(['1 gen', '2–3', '4–9', '10+'])[i]}</span>
                      </div>
                    ))}
                  </div>
                )}
              </ControlGroup>
            </ControlPanel>
          )}

        </div>

        <div className={styles.sidebarActions}>
          <div className={styles.actionPanel}>
            <span className={styles.actionPanelLabel}>Actions</span>
            <div className={styles.actionRow}>
              <button type="button" className={styles.actionBtn} onClick={handleStep} disabled={running}>Step</button>
              {!is1D && (
                <button type="button" className={styles.actionBtn} onClick={handleCenter}>Center</button>
              )}
            </div>
          </div>
          <SimControls
            running={running}
            onToggle={handleToggleRunning}
            onReset={handleReset}
            onExport={() => setShowExport(true)}
          />
        </div>
      </div>

      {/* Population graph */}
      {showPop && (
        <div className={styles.popPanel}>
          <div className={[styles.popPanelHeader, !showGraph ? styles.popPanelHeaderCollapsed : ''].join(' ')}>
            <span className={styles.popPanelTitle}>Population</span>
            <button className={styles.popToggleBtn} onClick={() => setShowGraph(v => !v)} aria-label={showGraph ? 'Collapse' : 'Expand'}>
              {showGraph ? '▾' : '▸'}
            </button>
          </div>
          {showGraph && (
            <div className={styles.popPlot}>
              {sparkData.length < 2 ? (
                <div className={styles.popEmpty}>waiting for data…</div>
              ) : (() => {
                const SVG_W = 360, SVG_H = 150, PL = 57, PR = 12, PT = 12, PB = 36;
                const plotW = SVG_W - PL - PR;
                const plotH = SVG_H - PT - PB;
                const max   = Math.max(...sparkData, 1);
                const startGen = Math.max(0, generation - sparkData.length + 1);
                const fmtN  = (n: number) => n >= 10_000 ? `${(n / 1000).toFixed(0)}k` : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`;
                const linePts = sparkData.map((v, i) =>
                  `${(PL + (i / (sparkData.length - 1)) * plotW).toFixed(1)},${(PT + (1 - v / max) * plotH).toFixed(1)}`
                ).join(' ');
                const fillPts = `${PL},${PT + plotH} ${linePts} ${(PL + plotW).toFixed(1)},${PT + plotH}`;
                return (
                  <svg width={SVG_W} height={SVG_H} viewBox={`0 0 ${SVG_W} ${SVG_H}`} className={styles.popSvg} aria-hidden>
                    <line x1={PL} y1={PT} x2={PL} y2={PT + plotH} stroke="rgba(255,255,255,0.4)" strokeWidth="1.5" />
                    <line x1={PL} y1={PT + plotH} x2={PL + plotW} y2={PT + plotH} stroke="rgba(255,255,255,0.4)" strokeWidth="1.5" />
                    {([0, 0.5, 1] as const).map(frac => {
                      const val = Math.round(max * frac);
                      const y   = PT + (1 - frac) * plotH;
                      return (
                        <g key={frac}>
                          <line x1={PL - 5} y1={y} x2={PL} y2={y} stroke="rgba(255,255,255,0.5)" strokeWidth="1.5" />
                          <text x={PL - 8} y={y + 5} textAnchor="end" fill="rgba(255,255,255,0.85)" fontSize="12">{fmtN(val)}</text>
                        </g>
                      );
                    })}
                    <text x={PL} y={PT + plotH + 20} textAnchor="middle" fill="rgba(255,255,255,0.75)" fontSize="12">{startGen.toLocaleString()}</text>
                    <text x={PL + plotW} y={PT + plotH + 20} textAnchor="middle" fill="rgba(255,255,255,0.75)" fontSize="12">{generation.toLocaleString()}</text>
                    <text x={PL + plotW / 2} y={SVG_H - 2} textAnchor="middle" fill="rgba(255,255,255,0.55)" fontSize="11">generation →</text>
                    <text x={12} y={PT + plotH / 2} textAnchor="middle" fill="rgba(255,255,255,0.55)" fontSize="11" transform={`rotate(-90,12,${PT + plotH / 2})`}>↑ pop</text>
                    <polygon points={fillPts} fill={activeRule.color} opacity="0.08" />
                    <polyline points={linePts} fill="none" stroke={activeRule.color} strokeWidth="2.25" strokeLinejoin="round" strokeLinecap="round" opacity="0.85" />
                  </svg>
                );
              })()}
            </div>
          )}
        </div>
      )}

      {/* Export dialog */}
      {showExport && (
        <ExportDialog
          onClose={() => setShowExport(false)}
          onDownload={({ width, height, format }) => {
            if (canvasRef.current) exportImage(canvasRef.current, width, height, format, 'cellular-automata');
            setShowExport(false);
          }}
        />
      )}

      {/* HUD */}
      <div className={styles.hud}>
        <div className={styles.hudLeft}>
          <span className={styles.hudTitle} style={{ color: activeRule.color }}>{activeRule.name}</span>
          <span className={styles.hudSub}>
            gen {generation.toLocaleString()}
            {showPop && ` · pop ${population.toLocaleString()}`}
          </span>
        </div>
        <div className={styles.hudRight}>
          {!is1D && <span className={styles.hudHint}>scroll to zoom · right-drag to pan · click to draw</span>}
          <button type="button" className={styles.hudBtn} onClick={toggleFullscreen} title={isFullscreen ? 'Exit fullscreen (F)' : 'Fullscreen (F)'}>
            {isFullscreen ? '⤡' : '⤢'}
          </button>
        </div>
      </div>
    </div>
  );
}
