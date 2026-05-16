import { Link } from 'react-router-dom';
import styles from './Home.module.css';

interface Exploration {
  path: string;
  title: string;
  category: string;
  description: string;
  symbol: string;
  color: string;
}

const EXPLORATIONS: Exploration[] = [
  {
    path: '/lorenz',
    title: 'Strange Attractors',
    category: 'Attractor',
    description:
      'Six strange attractors: Lorenz, Rössler, Halvorsen, Thomas, Aizawa, Dadras. Each born from three equations yet producing infinitely complex, never-repeating orbits.',
    symbol: '∿',
    color: 'var(--col-lorenz)',
  },
  {
    path: '/mandelbrot',
    title: 'Mandelbrot Set',
    category: 'Fractal',
    description:
      'The most famous object in mathematics. An infinitely complex boundary emerges from iterating a single quadratic equation on the complex plane.',
    symbol: '∞',
    color: 'var(--col-mandelbrot)',
  },
  {
    path: '/cardioid',
    title: 'Cardioid',
    category: 'Curve',
    description:
      'A heart-shaped curve traced by a point on a circle rolling around an equal circle. Also the main bulb boundary of the Mandelbrot set.',
    symbol: 'θ',
    color: 'var(--col-cardioid)',
  },
  {
    path: '/bifurcation',
    title: 'Bifurcation Diagram',
    category: 'Diagram',
    description:
      'The logistic map x → r·x·(1−x) reveals the road to chaos through period doubling. The Feigenbaum constant δ ≈ 4.669 governs the cascade.',
    symbol: 'δ',
    color: 'var(--col-bifurcation)',
  },
  {
    path: '/koch',
    title: 'Koch Snowflake',
    category: 'Fractal',
    description:
      'An infinite perimeter enclosing a finite area. Each edge of an equilateral triangle repeatedly sprouts smaller triangles, revealing self-similarity at every scale.',
    symbol: '❄',
    color: 'var(--col-koch)',
  },
  {
    path: '/double-pendulum',
    title: 'Double Pendulum',
    category: 'Attractor',
    description:
      'Two linked pendulums with unpredictable, swirling motion. A deceptively simple mechanical system that produces genuinely chaotic trajectories.',
    symbol: 'g',
    color: 'var(--col-pendulum)',
  },
  {
    path: '/conway',
    title: "Conway's Game of Life",
    category: 'Cellular Automaton',
    description:
      'Zero-player game on an infinite grid. Four simple rules applied to cells (alive or dead) produce emergent complexity: oscillators, spaceships, and unbounded growth.',
    symbol: '⬛',
    color: 'var(--col-conway)',
  },
  {
    path: '/three-body',
    title: 'Three Body Problem',
    category: 'N-Body',
    description:
      'Three massive bodies attracting each other through gravity. While the two-body problem is solvable, the three-body problem is generically chaotic — tiny changes in initial conditions lead to wildly different outcomes.',
    symbol: '⊙',
    color: 'var(--col-three-body)',
  },
];

export default function Home() {
  return (
    <div className={styles.page}>
      <header className={styles.hero}>
        <p className={styles.eyebrow}>Interactive Explorer</p>
        <h1 className={styles.title}>Chaos &amp; Fractals</h1>
        <p className={styles.subtitle}>
          Visualize the beautiful complexity hidden in simple equations.
          From strange attractors to infinite fractal boundaries, explore the mathematics of chaos.
        </p>
      </header>

      <section className={styles.grid}>
        {EXPLORATIONS.map((exp) => (
          <Link
            key={exp.path}
            to={exp.path}
            className={styles.card}
            style={{ '--card-color': exp.color } as React.CSSProperties}
          >
            <div className={styles.cardVisual}>
              <span className={styles.cardSymbol}>{exp.symbol}</span>
            </div>
            <div className={styles.cardBody}>
              <span className={styles.cardCategory}>{exp.category}</span>
              <h2 className={styles.cardTitle}>{exp.title}</h2>
              <p className={styles.cardDesc}>{exp.description}</p>
            </div>
          </Link>
        ))}
      </section>
    </div>
  );
}
