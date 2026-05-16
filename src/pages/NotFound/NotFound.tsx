import { useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import styles from './NotFound.module.css';

function useLorenz(ref: React.RefObject<HTMLCanvasElement | null>) {
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;

    const dpr = window.devicePixelRatio || 1;
    const W = (canvas.width  = Math.round(canvas.offsetWidth  * dpr));
    const H = (canvas.height = Math.round(canvas.offsetHeight * dpr));
    const ctx = canvas.getContext('2d')!;

    const N = 8000;
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
    const pad = 0.08, xR = xMax - xMin, zR = zMax - zMin;
    const mapX = (v: number) => (pad + (v - xMin) / xR * (1 - 2*pad)) * W;
    const mapY = (v: number) => (pad + (v - zMin) / zR * (1 - 2*pad)) * H;

    // Pre-draw the ghost trail
    const bg = new OffscreenCanvas(W, H);
    const bgCtx = bg.getContext('2d')!;
    bgCtx.fillStyle = '#0b0b18';
    bgCtx.fillRect(0, 0, W, H);
    bgCtx.strokeStyle = 'rgba(129,140,248,0.08)';
    bgCtx.lineWidth = 0.8 * dpr;
    bgCtx.beginPath();
    bgCtx.moveTo(mapX(xs[0]), mapY(zs[0]));
    for (let i = 1; i < N; i++) bgCtx.lineTo(mapX(xs[i]), mapY(zs[i]));
    bgCtx.stroke();

    let head = 0, raf: number;
    const TRAIL = 900;

    function frame() {
      ctx.drawImage(bg, 0, 0);

      const tStart = Math.max(0, head - TRAIL);
      ctx.lineWidth = 1.5 * dpr;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      ctx.moveTo(mapX(xs[tStart]), mapY(zs[tStart]));
      for (let i = tStart + 1; i <= head; i++) ctx.lineTo(mapX(xs[i % N]), mapY(zs[i % N]));
      ctx.strokeStyle = 'rgba(165,180,252,0.6)';
      ctx.stroke();

      const hx = mapX(xs[head]), hy = mapY(zs[head]);
      const g = ctx.createRadialGradient(hx, hy, 0, hx, hy, 10 * dpr);
      g.addColorStop(0, 'rgba(224,231,255,0.95)');
      g.addColorStop(1, 'rgba(129,140,248,0)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(hx, hy, 10 * dpr, 0, Math.PI * 2);
      ctx.fill();

      head = (head + 1) % N;
      raf = requestAnimationFrame(frame);
    }

    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [ref]);
}

export default function NotFound() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useLorenz(canvasRef);

  return (
    <div className={styles.page}>
      <canvas ref={canvasRef} className={styles.canvas} aria-hidden="true" />
      <div className={styles.overlay}>
        <p className={styles.eyebrow}>Error 404</p>
        <h1 className={styles.code}>Lost in chaos</h1>
        <p className={styles.description}>
          This page has diverged into an unknown trajectory.<br />
          Even tiny perturbations can lead here, but the attractor always has a home.
        </p>
        <Link to="/" className={styles.homeLink}>
          ← Return to the attractor
        </Link>
      </div>
    </div>
  );
}
