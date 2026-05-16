import { useEffect, useRef } from 'react';
import styles from './MiniPreview.module.css';

export type PreviewType =
  | 'lorenz'
  | 'mandelbrot'
  | 'cardioid'
  | 'bifurcation'
  | 'koch'
  | 'pendulum'
  | 'conway'
  | 'cellular'
  | 'threebody'
  | 'reaction';

type Renderer = (canvas: HTMLCanvasElement) => () => void;

// ─── Lorenz Attractor ────────────────────────────────────────────────────────

const lorenzRenderer: Renderer = (canvas) => {
  const dpr = window.devicePixelRatio || 1;
  const W = (canvas.width  = Math.round(canvas.offsetWidth  * dpr));
  const H = (canvas.height = Math.round(canvas.offsetHeight * dpr));
  const ctx = canvas.getContext('2d')!;

  const N = 5000;
  const xs = new Float32Array(N), zs = new Float32Array(N);
  let lx = 0.1, ly = 0, lz = 0;
  const σ = 10, ρ = 28, β = 8 / 3, dt = 0.005;

  for (let i = 0; i < 500; i++) {
    const dx = σ*(ly-lx), dy = lx*(ρ-lz)-ly, dz = lx*ly-β*lz;
    lx += dx*dt; ly += dy*dt; lz += dz*dt;
  }
  for (let i = 0; i < N; i++) {
    const dx = σ*(ly-lx), dy = lx*(ρ-lz)-ly, dz = lx*ly-β*lz;
    lx += dx*dt; ly += dy*dt; lz += dz*dt;
    xs[i] = lx; zs[i] = lz;
  }

  let xMin = Infinity, xMax = -Infinity, zMin = Infinity, zMax = -Infinity;
  for (let i = 0; i < N; i++) {
    if (xs[i] < xMin) xMin = xs[i]; if (xs[i] > xMax) xMax = xs[i];
    if (zs[i] < zMin) zMin = zs[i]; if (zs[i] > zMax) zMax = zs[i];
  }
  const pad = 0.06, xR = xMax - xMin, zR = zMax - zMin;
  const mapX = (v: number) => (pad + (v - xMin) / xR * (1 - 2*pad)) * W;
  const mapY = (v: number) => (pad + (v - zMin) / zR * (1 - 2*pad)) * H;

  const bg = new OffscreenCanvas(W, H);
  const bgCtx = bg.getContext('2d')!;
  bgCtx.fillStyle = '#0b0b18';
  bgCtx.fillRect(0, 0, W, H);
  bgCtx.strokeStyle = 'rgba(129,140,248,0.14)';
  bgCtx.lineWidth = 0.8 * dpr;
  bgCtx.beginPath();
  bgCtx.moveTo(mapX(xs[0]), mapY(zs[0]));
  for (let i = 1; i < N; i++) bgCtx.lineTo(mapX(xs[i]), mapY(zs[i]));
  bgCtx.stroke();

  let head = 0, raf: number;
  const TRAIL = 700;

  function frame() {
    ctx.drawImage(bg, 0, 0);

    const tStart = Math.max(0, head - TRAIL);
    ctx.lineWidth = 1.5 * dpr;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(mapX(xs[tStart]), mapY(zs[tStart]));
    for (let i = tStart + 1; i <= head; i++) ctx.lineTo(mapX(xs[i % N]), mapY(zs[i % N]));
    ctx.strokeStyle = 'rgba(165,180,252,0.7)';
    ctx.stroke();

    const hx = mapX(xs[head]), hy = mapY(zs[head]);
    const g = ctx.createRadialGradient(hx, hy, 0, hx, hy, 7 * dpr);
    g.addColorStop(0, 'rgba(224,231,255,0.95)');
    g.addColorStop(1, 'rgba(129,140,248,0)');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(hx, hy, 7 * dpr, 0, Math.PI * 2);
    ctx.fill();

    head = (head + 1) % N;
    raf = requestAnimationFrame(frame);
  }

  raf = requestAnimationFrame(frame);
  return () => cancelAnimationFrame(raf);
};

// ─── Mandelbrot Set ──────────────────────────────────────────────────────────

const mandelbrotRenderer: Renderer = (canvas) => {
  const dpr = window.devicePixelRatio || 1;
  const W = (canvas.width  = Math.round(canvas.offsetWidth  * dpr));
  const H = (canvas.height = Math.round(canvas.offsetHeight * dpr));
  const ctx = canvas.getContext('2d')!;

  const scale = 0.6;
  const w = Math.max(1, Math.round(W * scale));
  const h = Math.max(1, Math.round(H * scale));
  const img = new ImageData(w, h);
  const { data } = img;

  const MAX = 72, aspect = w / h;
  const cX = -0.5, viewH = 2.4, viewW = viewH * aspect;

  for (let py = 0; py < h; py++) {
    for (let px = 0; px < w; px++) {
      const cr = cX - viewW/2 + (px/w) * viewW;
      const ci =    - viewH/2 + (py/h) * viewH;
      let zr = 0, zi = 0, n = 0;
      while (n < MAX && zr*zr + zi*zi < 4) {
        const t = zr*zr - zi*zi + cr; zi = 2*zr*zi + ci; zr = t; n++;
      }
      const idx = (py * w + px) * 4;
      if (n === MAX) {
        data[idx] = 8; data[idx+1] = 4; data[idx+2] = 22; data[idx+3] = 255;
      } else {
        const t = Math.sqrt(n / MAX);   // sqrt brightens near-boundary bands
        const t2 = t * t;
        data[idx]   = Math.round(Math.min(255, 20  + 235 * t2));       // R: dark → bright
        data[idx+1] = Math.round(Math.min(255, 5   + 190 * t2 * t));   // G: subtle accent
        data[idx+2] = Math.round(Math.min(255, 80  + 175 * t));        // B: always-visible base
        data[idx+3] = 255;
      }
    }
  }

  const off = new OffscreenCanvas(w, h);
  off.getContext('2d')!.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'medium';
  ctx.drawImage(off, 0, 0, W, H);

  return () => {};
};

// ─── Cardioid (string art) ───────────────────────────────────────────────────

const cardioidRenderer: Renderer = (canvas) => {
  const dpr = window.devicePixelRatio || 1;
  const W = (canvas.width  = Math.round(canvas.offsetWidth  * dpr));
  const H = (canvas.height = Math.round(canvas.offsetHeight * dpr));
  const ctx = canvas.getContext('2d')!;

  const N = 200;
  const TAU = 2 * Math.PI;
  const cx = W / 2, cy = H / 2;
  const R  = Math.min(W * 0.44, H * 0.44);
  let mult = 2.0, frameN = 0, raf: number;

  function draw() {
    ctx.fillStyle = '#0b0b18';
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = 'rgba(251,146,60,0.16)';
    ctx.lineWidth = 0.8 * dpr;
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, TAU);
    ctx.stroke();

    ctx.strokeStyle = 'rgba(251,146,60,0.44)';
    ctx.lineWidth = 0.65 * dpr;
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const a1 = (TAU * i) / N;
      const a2 = (TAU * i * mult) / N;
      ctx.moveTo(cx + R * Math.cos(a1), cy + R * Math.sin(a1));
      ctx.lineTo(cx + R * Math.cos(a2), cy + R * Math.sin(a2));
    }
    ctx.stroke();

    frameN++;
    mult = 2.0 + (Math.sin(frameN * 0.004) + 1) * 0.8;
    raf = requestAnimationFrame(draw);
  }

  raf = requestAnimationFrame(draw);
  return () => cancelAnimationFrame(raf);
};

// ─── Bifurcation Diagram ─────────────────────────────────────────────────────

const bifurcationRenderer: Renderer = (canvas) => {
  const dpr = window.devicePixelRatio || 1;
  const W = (canvas.width  = Math.round(canvas.offsetWidth  * dpr));
  const H = (canvas.height = Math.round(canvas.offsetHeight * dpr));
  const ctx = canvas.getContext('2d')!;

  const img = new ImageData(W, H);
  const { data } = img;
  for (let i = 0; i < data.length; i += 4) {
    data[i] = 7; data[i+1] = 7; data[i+2] = 18; data[i+3] = 255;
  }

  for (let px = 0; px < W; px++) {
    const r = 2.5 + (px / W) * 1.5;
    let x = 0.5;
    for (let i = 0; i < 250; i++) x = r * x * (1 - x);
    for (let i = 0; i < 150; i++) {
      x = r * x * (1 - x);
      const py = Math.round((1 - x) * (H - 1));
      if (py >= 0 && py < H) {
        const idx = (py * W + px) * 4;
        data[idx] = 34; data[idx+1] = 211; data[idx+2] = 238; data[idx+3] = 200;
      }
    }
  }

  ctx.putImageData(img, 0, 0);
  return () => {};
};

// ─── Koch Snowflake ──────────────────────────────────────────────────────────

function kochSeg(
  ctx: CanvasRenderingContext2D,
  x1: number, y1: number,
  x2: number, y2: number,
  depth: number,
) {
  if (depth === 0) { ctx.lineTo(x2, y2); return; }
  const dx = x2 - x1, dy = y2 - y1;
  const ax = x1 + dx/3, ay = y1 + dy/3;
  const bx = x1 + 2*dx/3, by = y1 + 2*dy/3;
  const ex = dx/3, ey = dy/3;
  const sin60 = Math.sqrt(3) / 2;
  // Rotate (ex,ey) by +60° in screen coords → outward bump
  const px = ax + 0.5*ex - sin60*ey;
  const py = ay + sin60*ex + 0.5*ey;
  kochSeg(ctx, x1, y1, ax, ay, depth - 1);
  kochSeg(ctx, ax, ay, px, py, depth - 1);
  kochSeg(ctx, px, py, bx, by, depth - 1);
  kochSeg(ctx, bx, by, x2, y2, depth - 1);
}

const kochRenderer: Renderer = (canvas) => {
  const dpr = window.devicePixelRatio || 1;
  const W = (canvas.width  = Math.round(canvas.offsetWidth  * dpr));
  const H = (canvas.height = Math.round(canvas.offsetHeight * dpr));
  const ctx = canvas.getContext('2d')!;

  const s  = Math.min(W * 0.80, H * 0.80);
  const tH = s * Math.sqrt(3) / 2;
  const cx = W / 2, cy = H / 2;
  const v0 = [cx - s/2, cy + tH/3] as const;
  const v1 = [cx + s/2, cy + tH/3] as const;
  const v2 = [cx,        cy - 2*tH/3] as const;

  let iter = 0, frameN = 0, lastChange = 0, raf: number;

  function draw() {
    ctx.fillStyle = '#0b0b18';
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = '#93c5fd';
    ctx.lineWidth = 1 * dpr;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(v0[0], v0[1]);
    kochSeg(ctx, v0[0], v0[1], v1[0], v1[1], iter);
    kochSeg(ctx, v1[0], v1[1], v2[0], v2[1], iter);
    kochSeg(ctx, v2[0], v2[1], v0[0], v0[1], iter);
    ctx.closePath();
    ctx.stroke();

    frameN++;
    if (frameN - lastChange > 110) { lastChange = frameN; iter = (iter + 1) % 6; }
    raf = requestAnimationFrame(draw);
  }

  raf = requestAnimationFrame(draw);
  return () => cancelAnimationFrame(raf);
};

// ─── Double Pendulum ─────────────────────────────────────────────────────────

const pendulumRenderer: Renderer = (canvas) => {
  const dpr = window.devicePixelRatio || 1;
  const W = (canvas.width  = Math.round(canvas.offsetWidth  * dpr));
  const H = (canvas.height = Math.round(canvas.offsetHeight * dpr));
  const ctx = canvas.getContext('2d')!;

  const g = 9.8;
  const RAD = Math.PI / 180;
  let θ1 = 50 * RAD, ω1 = 0.0, θ2 = -20 * RAD, ω2 = 0.0;
  const pivX = W * 0.5, pivY = H * 0.18;
  const arm  = Math.min(W, H) * 0.34;
  const TRAIL = 350;
  const trailX = new Float32Array(TRAIL);
  const trailY = new Float32Array(TRAIL);
  let head = 0, full = false, raf: number;

  function step(dt: number) {
    const f = (t1: number, w1: number, t2: number, w2: number) => {
      const Δ = t1 - t2;
      const den = 3 - Math.cos(2 * Δ);
      const a1 = (-g*(2*Math.sin(t1) + Math.sin(t1-2*t2)) - 2*Math.sin(Δ)*(w2*w2 + w1*w1*Math.cos(Δ))) / den;
      const a2 = (2*Math.sin(Δ)*(2*w1*w1 + g*Math.cos(t1) + w2*w2*Math.cos(Δ))) / den;
      return [w1, a1, w2, a2] as const;
    };
    const [k1a,k1b,k1c,k1d] = f(θ1,ω1,θ2,ω2);
    const [k2a,k2b,k2c,k2d] = f(θ1+dt/2*k1a, ω1+dt/2*k1b, θ2+dt/2*k1c, ω2+dt/2*k1d);
    const [k3a,k3b,k3c,k3d] = f(θ1+dt/2*k2a, ω1+dt/2*k2b, θ2+dt/2*k2c, ω2+dt/2*k2d);
    const [k4a,k4b,k4c,k4d] = f(θ1+dt*k3a,   ω1+dt*k3b,   θ2+dt*k3c,   ω2+dt*k3d);
    θ1 += dt/6*(k1a+2*k2a+2*k3a+k4a);
    ω1 += dt/6*(k1b+2*k2b+2*k3b+k4b);
    θ2 += dt/6*(k1c+2*k2c+2*k3c+k4c);
    ω2 += dt/6*(k1d+2*k2d+2*k3d+k4d);
  }

  function frame() {
    step(0.006);

    const b1x = pivX + arm * Math.sin(θ1);
    const b1y = pivY + arm * Math.cos(θ1);
    const b2x = b1x  + arm * Math.sin(θ2);
    const b2y = b1y  + arm * Math.cos(θ2);

    trailX[head] = b2x; trailY[head] = b2y;
    head = (head + 1) % TRAIL;
    if (head === 0) full = true;

    ctx.fillStyle = '#0b0b18';
    ctx.fillRect(0, 0, W, H);

    const len = full ? TRAIL : head;
    if (len > 1) {
      ctx.lineJoin = 'round';
      ctx.lineWidth = 1.5 * dpr;
      for (let i = 1; i < len; i++) {
        const t   = i / len;
        const cur = (head - len + i + TRAIL) % TRAIL;
        const prv = (head - len + i - 1 + TRAIL) % TRAIL;
        ctx.strokeStyle = `rgba(74,222,128,${(t * 0.65).toFixed(2)})`;
        ctx.beginPath();
        ctx.moveTo(trailX[prv], trailY[prv]);
        ctx.lineTo(trailX[cur], trailY[cur]);
        ctx.stroke();
      }
    }

    ctx.strokeStyle = 'rgba(255,255,255,0.55)';
    ctx.lineWidth = 2 * dpr;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(pivX, pivY); ctx.lineTo(b1x, b1y);
    ctx.moveTo(b1x,  b1y);  ctx.lineTo(b2x, b2y);
    ctx.stroke();

    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.beginPath(); ctx.arc(pivX, pivY, 3*dpr, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.8)';
    ctx.beginPath(); ctx.arc(b1x,  b1y,  4*dpr, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = '#4ade80';
    ctx.beginPath(); ctx.arc(b2x,  b2y,  5*dpr, 0, Math.PI*2); ctx.fill();

    raf = requestAnimationFrame(frame);
  }

  raf = requestAnimationFrame(frame);
  return () => cancelAnimationFrame(raf);
};

// ─── Conway's Game of Life (mini) ────────────────────────────────────────────

const conwayRenderer: Renderer = (canvas) => {
  const dpr = window.devicePixelRatio || 1;
  const W = (canvas.width  = Math.round(canvas.offsetWidth  * dpr));
  const H = (canvas.height = Math.round(canvas.offsetHeight * dpr));
  const ctx = canvas.getContext('2d')!;

  const COLS = 72, ROWS = 36;
  let grid = new Uint8Array(COLS * ROWS);
  const ci = (r: number, c: number) => r * COLS + c;
  const wr = (v: number, m: number) => ((v % m) + m) % m;

  // Gosper Glider Gun offset to fit the grid
  const GUN: [number,number][] = [
    [5,1],[5,2],[6,1],[6,2],
    [5,11],[6,11],[7,11],[4,12],[8,12],[3,13],[9,13],[3,14],[9,14],
    [6,15],[4,16],[8,16],[5,17],[6,17],[7,17],[6,18],
    [3,21],[4,21],[5,21],[3,22],[4,22],[5,22],[2,23],[6,23],
    [1,25],[2,25],[6,25],[7,25],
    [3,35],[4,35],[3,36],[4,36],
  ];
  const offR = 3, offC = 2;
  for (const [r,c] of GUN) {
    const nr = r+offR, nc = c+offC;
    if (nr < ROWS && nc < COLS) grid[ci(nr, nc)] = 1;
  }

  function step() {
    const next = new Uint8Array(COLS * ROWS);
    for (let r = 0; r < ROWS; r++)
      for (let c = 0; c < COLS; c++) {
        let n = 0;
        for (let dr = -1; dr <= 1; dr++)
          for (let dc = -1; dc <= 1; dc++) {
            if (dr === 0 && dc === 0) continue;
            n += grid[ci(wr(r+dr, ROWS), wr(c+dc, COLS))];
          }
        const a = grid[ci(r, c)];
        next[ci(r, c)] = a ? (n === 2 || n === 3 ? 1 : 0) : (n === 3 ? 1 : 0);
      }
    grid = next;
  }

  const cw = W / COLS, ch = H / ROWS;
  let frameN = 0, raf: number;

  function draw() {
    ctx.fillStyle = '#0c0c1e';
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = '#6ee7b7';
    for (let r = 0; r < ROWS; r++)
      for (let c = 0; c < COLS; c++)
        if (grid[ci(r, c)])
          ctx.fillRect(
            Math.round(c*cw)+1, Math.round(r*ch)+1,
            Math.max(1, Math.round(cw)-1), Math.max(1, Math.round(ch)-1),
          );

    frameN++;
    if (frameN % 4 === 0) step();
    raf = requestAnimationFrame(draw);
  }

  raf = requestAnimationFrame(draw);
  return () => cancelAnimationFrame(raf);
};

// ─── Cellular Automata (Langton's Ant mini) ────────────────────────────────

const cellularRenderer: Renderer = (canvas) => {
  const dpr = window.devicePixelRatio || 1;
  const W = (canvas.width = Math.round(canvas.offsetWidth * dpr));
  const H = (canvas.height = Math.round(canvas.offsetHeight * dpr));
  const ctx = canvas.getContext('2d')!;

  const COLS = 50, ROWS = 30;
  const grid = new Uint8Array(COLS * ROWS);
  const ant = { x: Math.floor(COLS / 2), y: Math.floor(ROWS / 2), dir: 0 };
  const cw = W / COLS;
  const ch = H / ROWS;
  let raf = 0;

  const idx = (x: number, y: number) => y * COLS + x;
  const wrap = (value: number, max: number) => ((value % max) + max) % max;

  function step() {
    const i = idx(ant.x, ant.y);
    const state = grid[i];
    ant.dir = wrap(ant.dir + (state === 0 ? 1 : -1), 4);
    grid[i] = state === 0 ? 1 : 0;

    if (ant.dir === 0) ant.y = wrap(ant.y - 1, ROWS);
    if (ant.dir === 1) ant.x = wrap(ant.x + 1, COLS);
    if (ant.dir === 2) ant.y = wrap(ant.y + 1, ROWS);
    if (ant.dir === 3) ant.x = wrap(ant.x - 1, COLS);
  }

  function frame() {
    for (let i = 0; i < 2; i++) step();

    ctx.fillStyle = '#0c0c1e';
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = '#c084fc';
    for (let y = 0; y < ROWS; y++) {
      for (let x = 0; x < COLS; x++) {
        if (!grid[idx(x, y)]) continue;
        ctx.fillRect(
          Math.round(x * cw),
          Math.round(y * ch),
          Math.max(1, Math.ceil(cw)),
          Math.max(1, Math.ceil(ch)),
        );
      }
    }

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(
      Math.round(ant.x * cw),
      Math.round(ant.y * ch),
      Math.max(2, Math.ceil(cw)),
      Math.max(2, Math.ceil(ch)),
    );

    raf = requestAnimationFrame(frame);
  }

  raf = requestAnimationFrame(frame);
  return () => cancelAnimationFrame(raf);
};

// ─── Three Body Problem (figure-8 orbit) ────────────────────────────────────

const threebodyRenderer: Renderer = (canvas) => {
  const dpr = window.devicePixelRatio || 1;
  const W = (canvas.width  = Math.round(canvas.offsetWidth  * dpr));
  const H = (canvas.height = Math.round(canvas.offsetHeight * dpr));
  const ctx = canvas.getContext('2d')!;

  // Chenciner & Montgomery figure-8 initial conditions
  const bodies = [
    { x: -0.97000436, y:  0.24308753, vx:  0.4662036850, vy:  0.4323657300 },
    { x:  0.97000436, y: -0.24308753, vx:  0.4662036850, vy:  0.4323657300 },
    { x:  0,          y:  0,          vx: -0.93240737,   vy: -0.86473146   },
  ];
  const COLORS = ['#f43f5e', '#fb923c', '#fbbf24'];
  const scale  = Math.min(W * 0.34, H * 0.34);
  const cx = W / 2, cy = H / 2;
  let raf: number;

  const toScreen = (x: number, y: number): [number, number] =>
    [cx + x * scale, cy - y * scale];

  ctx.fillStyle = '#0b0b18';
  ctx.fillRect(0, 0, W, H);

  function gravStep(dt: number) {
    const fx = [0, 0, 0], fy = [0, 0, 0];
    for (let i = 0; i < 3; i++)
      for (let j = i+1; j < 3; j++) {
        const dx = bodies[j].x - bodies[i].x;
        const dy = bodies[j].y - bodies[i].y;
        const r2 = dx*dx + dy*dy, r = Math.sqrt(r2) + 1e-4;
        const f  = 1 / (r2 * r); // G=1, M=1
        fx[i] += f*dx; fy[i] += f*dy;
        fx[j] -= f*dx; fy[j] -= f*dy;
      }
    for (let i = 0; i < 3; i++) {
      bodies[i].vx += fx[i] * dt; bodies[i].vy += fy[i] * dt;
      bodies[i].x  += bodies[i].vx * dt; bodies[i].y += bodies[i].vy * dt;
    }
  }

  function frame() {
    // Fading trail effect
    ctx.fillStyle = 'rgba(11,11,24,0.12)';
    ctx.fillRect(0, 0, W, H);

    for (let s = 0; s < 3; s++) gravStep(0.005);

    for (let i = 0; i < 3; i++) {
      const [sx, sy] = toScreen(bodies[i].x, bodies[i].y);
      const gr = ctx.createRadialGradient(sx, sy, 0, sx, sy, 7 * dpr);
      gr.addColorStop(0, COLORS[i]);
      gr.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = gr;
      ctx.beginPath();
      ctx.arc(sx, sy, 7 * dpr, 0, Math.PI * 2);
      ctx.fill();
    }

    raf = requestAnimationFrame(frame);
  }

  raf = requestAnimationFrame(frame);
  return () => cancelAnimationFrame(raf);
};

// ─── Reaction-Diffusion (Gray-Scott, animated) ───────────────────────────────

const reactionRenderer: Renderer = (canvas) => {
  const dpr = window.devicePixelRatio || 1;
  const W = (canvas.width  = Math.round(canvas.offsetWidth  * dpr));
  const H = (canvas.height = Math.round(canvas.offsetHeight * dpr));
  const ctx = canvas.getContext('2d')!;

  const SW = 96, SH = 96;
  const N  = SW * SH;

  const u  = new Float32Array(N).fill(1);
  const v  = new Float32Array(N);
  const nu = new Float32Array(N);
  const nv = new Float32Array(N);
  const imgData = new ImageData(SW, SH);
  const off = new OffscreenCanvas(SW, SH);
  const offCtx = off.getContext('2d')!;

  // Spots preset
  const F = 0.035, K = 0.065, DU = 0.2097, DV = 0.105;

  // Seed a few blobs
  const seeds = [[48,48,7],[20,70,5],[75,25,5],[25,25,4],[70,70,4]];
  for (const [cx, cy, r] of seeds) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (dx*dx + dy*dy > r*r) continue;
        const x = ((cx+dx)%SW+SW)%SW;
        const y = ((cy+dy)%SH+SH)%SH;
        const i = y*SW+x;
        u[i] = 0.5; v[i] = 0.25;
      }
    }
  }

  function step() {
    for (let y = 0; y < SH; y++) {
      const yW  = y * SW;
      const ynW = (y === 0 ? SH-1 : y-1) * SW;
      const ysW = (y === SH-1 ? 0 : y+1) * SW;
      for (let x = 0; x < SW; x++) {
        const i  = yW + x;
        const xL = x === 0    ? SW-1 : x-1;
        const xR = x === SW-1 ? 0    : x+1;
        const ui = u[i], vi = v[i];
        const lapU = u[yW+xL] + u[yW+xR] + u[ynW+x] + u[ysW+x] - 4*ui;
        const lapV = v[yW+xL] + v[yW+xR] + v[ynW+x] + v[ysW+x] - 4*vi;
        const uvv = ui * vi * vi;
        let nu_ = ui + DU*lapU - uvv + F*(1-ui);
        let nv_ = vi + DV*lapV + uvv - (F+K)*vi;
        if (nu_ < 0) nu_ = 0; else if (nu_ > 1) nu_ = 1;
        if (nv_ < 0) nv_ = 0; else if (nv_ > 1) nv_ = 1;
        nu[i] = nu_; nv[i] = nv_;
      }
    }
    u.set(nu); v.set(nv);
  }

  const { data } = imgData;
  let raf: number, frame = 0;

  function draw() {
    for (let s = 0; s < 6; s++) step();

    for (let i = 0; i < N; i++) {
      const t = v[i];
      const idx = i << 2;
      data[idx]   = Math.round(3   + t * 20);
      data[idx+1] = Math.round(13  + t * 195);
      data[idx+2] = Math.round(13  + t * 178);
      data[idx+3] = 255;
    }
    offCtx.putImageData(imgData, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, W, H);

    frame++;
    // After ~800 gens, re-seed a random blob to keep things evolving
    if (frame % 133 === 0) {
      const cx = 10 + Math.floor(Math.random() * (SW - 20));
      const cy = 10 + Math.floor(Math.random() * (SH - 20));
      for (let dy = -4; dy <= 4; dy++) {
        for (let dx = -4; dx <= 4; dx++) {
          if (dx*dx+dy*dy > 16) continue;
          const x = ((cx+dx)%SW+SW)%SW;
          const y = ((cy+dy)%SH+SH)%SH;
          const i = y*SW+x;
          u[i] = 0.5; v[i] = 0.25;
        }
      }
    }

    raf = requestAnimationFrame(draw);
  }

  raf = requestAnimationFrame(draw);
  return () => cancelAnimationFrame(raf);
};

// ─── Registry ────────────────────────────────────────────────────────────────

const RENDERERS: Record<PreviewType, Renderer> = {
  lorenz:      lorenzRenderer,
  mandelbrot:  mandelbrotRenderer,
  cardioid:    cardioidRenderer,
  bifurcation: bifurcationRenderer,
  koch:        kochRenderer,
  pendulum:    pendulumRenderer,
  conway:      conwayRenderer,
  cellular:    cellularRenderer,
  threebody:   threebodyRenderer,
  reaction:    reactionRenderer,
};

// ─── Component ───────────────────────────────────────────────────────────────

export function MiniPreview({ type }: { type: PreviewType }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || canvas.offsetWidth === 0) return;
    const stop = RENDERERS[type](canvas);
    return stop;
  }, [type]);

  return <canvas ref={ref} className={styles.canvas} />;
}
