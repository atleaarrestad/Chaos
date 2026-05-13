import { useLocation } from 'react-router-dom';
import styles from './ComingSoon.module.css';

const LABELS: Record<string, { symbol: string; title: string; color: string }> = {
  '/lorenz':          { symbol: 'σ', title: 'Lorenz Attractor',   color: 'var(--col-lorenz)' },
  '/mandelbrot':      { symbol: '∞', title: 'Mandelbrot Set',      color: 'var(--col-mandelbrot)' },
  '/julia':           { symbol: 'ℂ', title: 'Julia Sets',          color: 'var(--col-julia)' },
  '/cardioid':        { symbol: 'θ', title: 'Cardioid',            color: 'var(--col-cardioid)' },
  '/bifurcation':     { symbol: 'δ', title: 'Bifurcation Diagram', color: 'var(--col-bifurcation)' },
  '/double-pendulum': { symbol: 'g', title: 'Double Pendulum',     color: 'var(--col-pendulum)' },
};

export default function ComingSoon() {
  const { pathname } = useLocation();
  const meta = LABELS[pathname] ?? { symbol: '?', title: 'Exploration', color: 'var(--accent)' };

  return (
    <div className={styles.page}>
      <span
        className={styles.symbol}
        style={{ color: meta.color } as React.CSSProperties}
      >
        {meta.symbol}
      </span>
      <h1 className={styles.title}>{meta.title}</h1>
      <p className={styles.message}>This exploration is coming soon.</p>
    </div>
  );
}
