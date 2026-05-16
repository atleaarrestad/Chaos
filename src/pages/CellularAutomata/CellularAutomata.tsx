import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ControlGroup, ControlPanel, SimControls, Slider } from '@/components/Controls';
import { useFullscreen } from '@/hooks/useFullscreen';
import styles from './CellularAutomata.module.css';

const COLS = 300;
const ROWS = 200;
const BG_HEX = '#0c0c1e';
const DEFAULT_DENSITY = 30;
const CENTER_X = Math.floor(COLS / 2);
const CENTER_Y = Math.floor(ROWS / 2);

type SimType = '2d' | 'brain' | 'ant' | '1d';

interface RulesetDef {
  id: string;
  name: string;
  category: string;
  description: string;
  notation?: string;
  simType: SimType;
  color: string;
  birth?: number[];
  survive?: number[];
  rule?: number;
  antTurns?: number[];
  antStateColors?: string[];
  defaultDensity?: number;
  defaultDelay?: number;  // ms between frames (0 = full speed)
}

interface Ant {
  x: number;
  y: number;
  dir: number;
}

interface SimState {
  grid: Uint8Array;
  ants: Ant[];
  row: Uint8Array;
  rowHistory: Uint8Array;
  rowIdx: number;
  generation: number;
  offscreen: OffscreenCanvas;
  offCtx: OffscreenCanvasRenderingContext2D;
}

const RULESETS = [
  {
    id: 'langtons-ant',
    name: "Langton's Ant",
    category: 'Ant Automaton',
    description:
      "An ant traverses a grid: turn right on a white cell, turn left on black, then flip. From apparent chaos emerges a periodic 'highway' after ~10,000 steps.",
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
      'Proven Turing complete — capable of universal computation. Complex glider-like structures emerge, collide, and interact.',
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
    simType: '1d',
    rule: 90,
    color: '#34d399',
    notation: 'Rule 90',
    defaultDelay: 80,
  },
  {
    id: 'brians-brain',
    name: "Brian's Brain",
    category: '3-State CA',
    description:
      'Cells cycle: dead → alive → dying → dead. Nearly every random initial condition produces gliders. A born cell needs exactly 2 alive neighbors.',
    simType: 'brain',
    color: '#60a5fa',
    notation: 'B2 / dying / dead',
  },
  {
    id: 'seeds',
    name: 'Seeds',
    category: '2D Automaton',
    description:
      'Nothing survives — but any dead cell with exactly 2 live neighbors springs to life. Produces explosive wave-like growth.',
    simType: '2d',
    birth: [2],
    survive: [],
    color: '#fbbf24',
    notation: 'B2/S',
    defaultDensity: 0.05,
  },
  {
    id: 'highlife',
    name: 'HighLife',
    category: '2D Automaton',
    description:
      "Conway's rules plus birth at 6 neighbors. Contains a self-replicating pattern (the 'replicator') that spawns copies of itself.",
    simType: '2d',
    birth: [3, 6],
    survive: [2, 3],
    color: '#4ade80',
    notation: 'B36/S23',
  },
  {
    id: 'day-and-night',
    name: 'Day & Night',
    category: '2D Automaton',
    description:
      'Symmetric rules — live and dead regions are interchangeable. Produces stable isolated islands with rich internal structure.',
    simType: '2d',
    birth: [3, 6, 7, 8],
    survive: [3, 4, 6, 7, 8],
    color: '#38bdf8',
    notation: 'B3678/S34678',
  },
  {
    id: 'maze',
    name: 'Maze',
    category: '2D Automaton',
    description:
      'Cells are highly survivable and grow outward into winding corridors. Once a wall forms, it never dissolves.',
    simType: '2d',
    birth: [3],
    survive: [1, 2, 3, 4, 5],
    color: '#a78bfa',
    notation: 'B3/S12345',
    defaultDensity: 0.05,
  },
  {
    id: 'replicator',
    name: 'Replicator',
    category: '2D Automaton',
    description:
      'Odd-neighbor birth and survival rules cause every finite pattern to replicate itself, producing an infinite mosaic of copies.',
    simType: '2d',
    birth: [1, 3, 5, 7],
    survive: [1, 3, 5, 7],
    color: '#f472b6',
    notation: 'B1357/S1357',
    defaultDensity: 0.1,
  },
] as const satisfies readonly RulesetDef[];

type RuleId = (typeof RULESETS)[number]['id'];

type Rgb = [number, number, number];

const RULESET_MAP = new Map<RuleId, RulesetDef>(RULESETS.map((ruleset) => [ruleset.id, ruleset]));
const RULESET_GROUPS = Array.from(
  RULESETS.reduce((groups, ruleset) => {
    const existing = groups.get(ruleset.category);
    if (existing) {
      existing.push(ruleset);
    } else {
      groups.set(ruleset.category, [ruleset]);
    }
    return groups;
  }, new Map<string, RulesetDef[]>()),
  ([category, rulesets]) => ({ category, rulesets }),
);
const BG_RGB = hexToRgb(BG_HEX);

function wrap(value: number, max: number) {
  return ((value % max) + max) % max;
}

function indexAt(x: number, y: number) {
  return y * COLS + x;
}

function hexToRgb(hex: string): Rgb {
  const value = hex.replace('#', '');
  const parsed = Number.parseInt(value, 16);
  return [
    (parsed >> 16) & 255,
    (parsed >> 8) & 255,
    parsed & 255,
  ];
}

function mixRgb(fg: Rgb, bg: Rgb, amount: number): Rgb {
  return [
    Math.round(fg[0] * amount + bg[0] * (1 - amount)),
    Math.round(fg[1] * amount + bg[1] * (1 - amount)),
    Math.round(fg[2] * amount + bg[2] * (1 - amount)),
  ];
}

function createSimState(): SimState {
  const offscreen = new OffscreenCanvas(COLS, ROWS);
  const offCtx = offscreen.getContext('2d', { alpha: false });
  if (!offCtx) {
    throw new Error('Unable to create offscreen canvas context.');
  }
  offCtx.imageSmoothingEnabled = false;
  return {
    grid: new Uint8Array(COLS * ROWS),
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
  const offCtx = offscreen.getContext('2d', { alpha: false });
  if (!offCtx) {
    throw new Error('Unable to recreate offscreen canvas context.');
  }
  offCtx.imageSmoothingEnabled = false;
  sim.offscreen = offscreen;
  sim.offCtx = offCtx;
}

function initRandom(density: number) {
  const grid = new Uint8Array(COLS * ROWS);
  for (let i = 0; i < grid.length; i += 1) {
    grid[i] = Math.random() < density ? 1 : 0;
  }
  return grid;
}

function initAnt() {
  return {
    grid: new Uint8Array(COLS * ROWS),
    ants: [{ x: CENTER_X, y: CENTER_Y, dir: 0 }],
  };
}

function init1D() {
  const row = new Uint8Array(COLS);
  const rowHistory = new Uint8Array(COLS * ROWS);
  row[Math.floor(COLS / 2)] = 1;
  rowHistory.set(row, 0);
  return { row, rowHistory, rowIdx: 1 };
}

function initRandom1D(density: number) {
  const row = new Uint8Array(COLS);
  const rowHistory = new Uint8Array(COLS * ROWS);
  let active = 0;
  for (let i = 0; i < COLS; i += 1) {
    const value = Math.random() < density ? 1 : 0;
    row[i] = value;
    active += value;
  }
  if (active === 0) {
    row[Math.floor(COLS / 2)] = 1;
  }
  rowHistory.set(row, 0);
  return { row, rowHistory, rowIdx: 1 };
}

function initRandomAnt(nStates: number, density: number) {
  const grid = new Uint8Array(COLS * ROWS);
  for (let i = 0; i < grid.length; i += 1) {
    if (Math.random() < density) {
      grid[i] = 1 + Math.floor(Math.random() * Math.max(1, nStates - 1));
    }
  }
  return grid;
}

function step2D(grid: Uint8Array, birth: Set<number>, survive: Set<number>): Uint8Array {
  const next = new Uint8Array(grid.length);
  for (let y = 0; y < ROWS; y += 1) {
    for (let x = 0; x < COLS; x += 1) {
      let neighbors = 0;
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dx = -1; dx <= 1; dx += 1) {
          if (dx === 0 && dy === 0) continue;
          neighbors += grid[indexAt(wrap(x + dx, COLS), wrap(y + dy, ROWS))];
        }
      }
      const idx = indexAt(x, y);
      next[idx] = grid[idx]
        ? (survive.has(neighbors) ? 1 : 0)
        : (birth.has(neighbors) ? 1 : 0);
    }
  }
  return next;
}

function stepBrain(grid: Uint8Array): Uint8Array {
  const next = new Uint8Array(grid.length);
  for (let y = 0; y < ROWS; y += 1) {
    for (let x = 0; x < COLS; x += 1) {
      let aliveNeighbors = 0;
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dx = -1; dx <= 1; dx += 1) {
          if (dx === 0 && dy === 0) continue;
          aliveNeighbors += grid[indexAt(wrap(x + dx, COLS), wrap(y + dy, ROWS))] === 1 ? 1 : 0;
        }
      }
      const idx = indexAt(x, y);
      if (grid[idx] === 1) {
        next[idx] = 2;
      } else if (grid[idx] === 2) {
        next[idx] = 0;
      } else {
        next[idx] = aliveNeighbors === 2 ? 1 : 0;
      }
    }
  }
  return next;
}

function stepAnt(grid: Uint8Array, ants: Ant[], turns: number[], nStates: number) {
  for (const ant of ants) {
    const idx = indexAt(ant.x, ant.y);
    const state = grid[idx];
    const turn = turns[state] ?? 1;
    ant.dir = wrap(ant.dir + turn, 4);
    grid[idx] = (state + 1) % nStates;

    if (ant.dir === 0) ant.y = wrap(ant.y - 1, ROWS);
    if (ant.dir === 1) ant.x = wrap(ant.x + 1, COLS);
    if (ant.dir === 2) ant.y = wrap(ant.y + 1, ROWS);
    if (ant.dir === 3) ant.x = wrap(ant.x - 1, COLS);
  }
}

function step1D(row: Uint8Array, rule: number): Uint8Array {
  const next = new Uint8Array(COLS);
  for (let x = 0; x < COLS; x += 1) {
    const pattern =
      (row[wrap(x - 1, COLS)] << 2)
      | (row[x] << 1)
      | row[wrap(x + 1, COLS)];
    next[x] = (rule >> pattern) & 1;
  }
  return next;
}

function densityForRuleset(ruleset: RulesetDef) {
  return Math.round((ruleset.defaultDensity ?? DEFAULT_DENSITY / 100) * 100);
}

function buildSimState(ruleset: RulesetDef, mode: 'reset' | 'random' | 'clear', densityPercent: number) {
  const sim = createSimState();
  const density = densityPercent / 100;

  if (ruleset.simType === '2d' || ruleset.simType === 'brain') {
    sim.grid = mode === 'clear' ? new Uint8Array(COLS * ROWS) : initRandom(density);
    return sim;
  }

  if (ruleset.simType === 'ant') {
    const { ants } = initAnt();
    sim.ants = ants;
    sim.grid = mode === 'random'
      ? initRandomAnt(ruleset.antTurns?.length ?? 2, density)
      : new Uint8Array(COLS * ROWS);
    return sim;
  }

  if (mode === 'clear') {
    return sim;
  }

  const init = mode === 'random' ? initRandom1D(density) : init1D();
  sim.row = init.row;
  sim.rowHistory = init.rowHistory;
  sim.rowIdx = init.rowIdx;
  return sim;
}

function writePixel(data: Uint8ClampedArray, offset: number, color: Rgb) {
  data[offset] = color[0];
  data[offset + 1] = color[1];
  data[offset + 2] = color[2];
  data[offset + 3] = 255;
}

function renderSim(sim: SimState, ruleset: RulesetDef) {
  const image = new ImageData(COLS, ROWS);
  const { data } = image;
  const accent = hexToRgb(ruleset.color);
  const dying = mixRgb(accent, BG_RGB, 0.35);

  if (ruleset.simType === '1d') {
    for (let py = 0; py < ROWS; py += 1) {
      const historyRow = ((sim.rowIdx - ROWS + py + ROWS) % ROWS) * COLS;
      for (let px = 0; px < COLS; px += 1) {
        const offset = (py * COLS + px) * 4;
        writePixel(data, offset, sim.rowHistory[historyRow + px] ? accent : BG_RGB);
      }
    }
  } else if (ruleset.simType === 'brain') {
    for (let i = 0; i < sim.grid.length; i += 1) {
      const offset = i * 4;
      if (sim.grid[i] === 1) {
        writePixel(data, offset, accent);
      } else if (sim.grid[i] === 2) {
        writePixel(data, offset, dying);
      } else {
        writePixel(data, offset, BG_RGB);
      }
    }
  } else if (ruleset.simType === 'ant') {
    const colors = (ruleset.antStateColors ?? [BG_HEX, ruleset.color]).map(hexToRgb);
    for (let i = 0; i < sim.grid.length; i += 1) {
      const offset = i * 4;
      writePixel(data, offset, colors[sim.grid[i]] ?? BG_RGB);
    }
    for (const ant of sim.ants) {
      for (let dy = 0; dy < 2; dy += 1) {
        for (let dx = 0; dx < 2; dx += 1) {
          const px = ant.x + dx;
          const py = ant.y + dy;
          if (px >= COLS || py >= ROWS) continue;
          const offset = (py * COLS + px) * 4;
          data[offset] = 255;
          data[offset + 1] = 255;
          data[offset + 2] = 255;
          data[offset + 3] = 255;
        }
      }
    }
  } else {
    for (let i = 0; i < sim.grid.length; i += 1) {
      const offset = i * 4;
      writePixel(data, offset, sim.grid[i] ? accent : BG_RGB);
    }
  }

  sim.offCtx.putImageData(image, 0, 0);
}

export default function CellularAutomata() {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const simRef = useRef<SimState>(createSimState());
  const rafRef = useRef(0);
  const runningRef = useRef(false);
  const speedRef = useRef(1);
  const delayRef = useRef((RULESETS[0] as RulesetDef).defaultDelay ?? 0);
  const densityRef = useRef(densityForRuleset(RULESETS[0]));
  const activeRuleRef = useRef<RulesetDef>(RULESETS[0]);
  const lastStepTimeRef = useRef(0);

  const [activeRuleId, setActiveRuleId] = useState<RuleId>(RULESETS[0].id);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [delayMs, setDelayMs] = useState((RULESETS[0] as RulesetDef).defaultDelay ?? 0);
  const [density, setDensity] = useState(densityForRuleset(RULESETS[0]));
  const [generation, setGeneration] = useState(0);

  const activeRule: RulesetDef = RULESET_MAP.get(activeRuleId) ?? RULESETS[0];
  const showDensity = activeRule.simType === '2d' || activeRule.simType === 'brain';
  const { isFullscreen, toggleFullscreen } = useFullscreen(containerRef);

  const drawFrame = useCallback(() => {
    const canvas = canvasRef.current;
    const sim = simRef.current;
    if (!canvas || canvas.width === 0 || canvas.height === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    renderSim(sim, activeRuleRef.current);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(sim.offscreen, 0, 0, canvas.width, canvas.height);
  }, []);

  const stepSimulation = useCallback((steps: number) => {
    const sim = simRef.current;
    const ruleset = activeRuleRef.current;

    for (let i = 0; i < steps; i += 1) {
      if (ruleset.simType === '2d') {
        sim.grid = step2D(
          sim.grid,
          new Set(ruleset.birth ?? []),
          new Set(ruleset.survive ?? []),
        );
      } else if (ruleset.simType === 'brain') {
        sim.grid = stepBrain(sim.grid);
      } else if (ruleset.simType === 'ant') {
        stepAnt(
          sim.grid,
          sim.ants,
          ruleset.antTurns ?? [1, -1],
          ruleset.antTurns?.length ?? 2,
        );
      } else {
        const nextRow = step1D(sim.row, ruleset.rule ?? 110);
        sim.row = nextRow;
        sim.rowHistory.set(nextRow, sim.rowIdx * COLS);
        sim.rowIdx = (sim.rowIdx + 1) % ROWS;
      }
      sim.generation += 1;
    }

    setGeneration(sim.generation);
  }, []);

  const stopLoop = useCallback(() => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = 0;
    }
  }, []);

  const startLoop = useCallback(() => {
    if (rafRef.current) return;

    const frame = (now: number) => {
      const delay = delayRef.current;
      if (delay === 0 || now - lastStepTimeRef.current >= delay) {
        stepSimulation(speedRef.current);
        drawFrame();
        lastStepTimeRef.current = now;
      }
      rafRef.current = requestAnimationFrame(frame);
    };

    rafRef.current = requestAnimationFrame(frame);
  }, [drawFrame, stepSimulation]);

  const initializeSimulation = useCallback((ruleset: RulesetDef, mode: 'reset' | 'random' | 'clear', densityPercent = densityRef.current) => {
    const wasRunning = runningRef.current;
    stopLoop();
    activeRuleRef.current = ruleset;
    simRef.current = buildSimState(ruleset, mode, densityPercent);
    setGeneration(0);
    drawFrame();
    if (wasRunning) {
      startLoop();
    }
  }, [drawFrame, startLoop, stopLoop]);

  const handleRulesetSelect = useCallback((ruleset: RulesetDef) => {
    const nextDensity = densityForRuleset(ruleset);
    const nextDelay = ruleset.defaultDelay ?? 0;
    densityRef.current = nextDensity;
    delayRef.current = nextDelay;
    activeRuleRef.current = ruleset;
    setDensity(nextDensity);
    setDelayMs(nextDelay);
    setActiveRuleId(ruleset.id as RuleId);
    initializeSimulation(ruleset, 'reset', nextDensity);
  }, [initializeSimulation]);

  const handleReset = useCallback(() => {
    initializeSimulation(activeRuleRef.current, 'reset', densityRef.current);
  }, [initializeSimulation]);

  const handleRandom = useCallback(() => {
    initializeSimulation(activeRuleRef.current, 'random', densityRef.current);
  }, [initializeSimulation]);

  const handleClear = useCallback(() => {
    initializeSimulation(activeRuleRef.current, 'clear', densityRef.current);
  }, [initializeSimulation]);

  const handleStep = useCallback(() => {
    if (runningRef.current) return;
    stepSimulation(1);
    drawFrame();
  }, [drawFrame, stepSimulation]);

  const handleToggleRunning = useCallback(() => {
    setRunning((value) => !value);
  }, []);

  const handleSpeedChange = useCallback((value: number) => {
    speedRef.current = value;
    setSpeed(value);
  }, []);

  const handleDelayChange = useCallback((value: number) => {
    delayRef.current = value;
    setDelayMs(value);
  }, []);

  const handleDensityChange = useCallback((value: number) => {
    densityRef.current = value;
    setDensity(value);
  }, []);

  useEffect(() => {
    runningRef.current = running;
    if (running) {
      startLoop();
    } else {
      stopLoop();
      drawFrame();
    }
  }, [drawFrame, running, startLoop, stopLoop]);

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const updateSize = () => {
      const rect = container.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const width = Math.max(1, Math.round(rect.width * dpr));
      const height = Math.max(1, Math.round(rect.height * dpr));

      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
        recreateOffscreen(simRef.current);
        drawFrame();
      }
    };

    const observer = new ResizeObserver(updateSize);
    observer.observe(container);
    updateSize();

    return () => observer.disconnect();
  }, [drawFrame]);

  useEffect(() => {
    initializeSimulation(activeRuleRef.current, 'reset', densityRef.current);
    return () => stopLoop();
  }, [initializeSimulation, stopLoop]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLSelectElement) {
        return;
      }

      if (event.code === 'Space') {
        event.preventDefault();
        handleToggleRunning();
      }

      if (event.code === 'KeyR') {
        event.preventDefault();
        handleReset();
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [handleReset, handleToggleRunning]);

  const groupedRulesets = useMemo(() => RULESET_GROUPS, []);

  return (
    <div ref={containerRef} className={styles.container}>
      <canvas ref={canvasRef} className={styles.canvas} />

      <div className={styles.sidebar}>
        <div className={styles.sidebarPanels}>
          <ControlPanel title="Rulesets">
            <div className={styles.rulesetList}>
              {groupedRulesets.map((group) => (
                <div key={group.category} className={styles.rulesetSection}>
                  <div className={styles.rulesetHeader}>{group.category}</div>
                  <div className={styles.rulesetGrid}>
                    {group.rulesets.map((ruleset) => (
                      <button
                        key={ruleset.id}
                        type="button"
                        className={[
                          styles.rulesetBtn,
                          activeRuleId === ruleset.id ? styles.rulesetBtnActive : '',
                        ].join(' ')}
                        onClick={() => handleRulesetSelect(ruleset)}
                      >
                        <span className={styles.rulesetNameRow}>
                          <span
                            className={styles.rulesetDot}
                            style={{ backgroundColor: ruleset.color }}
                            aria-hidden="true"
                          />
                          <span className={styles.rulesetName}>{ruleset.name}</span>
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </ControlPanel>

          <ControlPanel title="Simulation Info">
            <div className={styles.infoPanel}>
              <div className={styles.infoMeta}>
                <span className={styles.categoryBadge}>{activeRule.category}</span>
                {activeRule.notation && (
                  <span className={styles.notationBadge}>{activeRule.notation}</span>
                )}
              </div>
              <p className={styles.infoText}>{activeRule.description}</p>
            </div>
          </ControlPanel>

          <ControlPanel title="Speed">
            <ControlGroup>
              <Slider
                label="Steps / frame"
                value={speed}
                min={1}
                max={20}
                step={1}
                onChange={handleSpeedChange}
              />
              <Slider
                label="Delay between frames"
                value={delayMs}
                min={0}
                max={500}
                step={10}
                unit="ms"
                onChange={handleDelayChange}
              />
            </ControlGroup>
          </ControlPanel>

          {showDensity && (
            <ControlPanel title="Density">
              <ControlGroup>
                <Slider
                  label="Random fill"
                  value={density}
                  min={5}
                  max={80}
                  step={1}
                  unit="%"
                  onChange={handleDensityChange}
                />
              </ControlGroup>
            </ControlPanel>
          )}
        </div>

        <div className={styles.sidebarActions}>
          <div className={styles.actionPanel}>
            <span className={styles.actionPanelLabel}>Actions</span>
            <div className={styles.actionRow}>
              <button type="button" className={styles.actionBtn} onClick={handleRandom}>
                Random
              </button>
              <button type="button" className={styles.actionBtn} onClick={handleClear}>
                Clear
              </button>
              <button type="button" className={styles.actionBtn} onClick={handleStep} disabled={running}>
                Step
              </button>
            </div>
          </div>
          <SimControls
            running={running}
            onToggle={handleToggleRunning}
            onReset={handleReset}
          />
        </div>
      </div>

      <div className={styles.hud}>
        <div className={styles.hudBlock}>
          <span className={styles.hudTitle}>{activeRule.name}</span>
          <span className={styles.hudSub}>Gen {generation.toLocaleString()}</span>
        </div>
        <div className={styles.hudActions}>
          <button
            type="button"
            className={styles.hudBtn}
            onClick={toggleFullscreen}
            title={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
          >
            {isFullscreen ? '⤡' : '⤢'}
          </button>
        </div>
      </div>
    </div>
  );
}
