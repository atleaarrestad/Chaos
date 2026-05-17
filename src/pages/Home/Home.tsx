import { Link } from 'react-router-dom';
import { MiniPreview, type PreviewType } from '../../components/MiniPreview/MiniPreview';
import styles from './Home.module.css';

interface Exploration {
  path: string;
  title: string;
  category: string;
  description: string;
  previewType: PreviewType;
  color: string;
}

const EXPLORATIONS: Exploration[] = [
  {
    path: '/lorenz',
    title: 'Strange Attractors',
    category: 'Attractor',
    description:
      'Six strange attractors: Lorenz, Rössler, Halvorsen, Thomas, Aizawa, Dadras. Each born from three equations yet producing infinitely complex, never-repeating orbits.',
    previewType: 'lorenz',
    color: 'var(--col-lorenz)',
  },
  {
    path: '/mandelbrot',
    title: 'Mandelbrot Set',
    category: 'Fractal',
    description:
      'The most famous object in mathematics. An infinitely complex boundary emerges from iterating a single quadratic equation on the complex plane.',
    previewType: 'mandelbrot',
    color: 'var(--col-mandelbrot)',
  },
  {
    path: '/cardioid',
    title: 'Cardioid',
    category: 'Curve',
    description:
      'A heart-shaped curve traced by a point on a circle rolling around an equal circle. Also the main bulb boundary of the Mandelbrot set.',
    previewType: 'cardioid',
    color: 'var(--col-cardioid)',
  },
  {
    path: '/bifurcation',
    title: 'Bifurcation Diagram',
    category: 'Diagram',
    description:
      'The logistic map x → r·x·(1−x) reveals the road to chaos through period doubling. The Feigenbaum constant δ ≈ 4.669 governs the cascade.',
    previewType: 'bifurcation',
    color: 'var(--col-bifurcation)',
  },
  {
    path: '/koch',
    title: 'Koch Snowflake',
    category: 'Fractal',
    description:
      'An infinite perimeter enclosing a finite area. Each edge of an equilateral triangle repeatedly sprouts smaller triangles, revealing self-similarity at every scale.',
    previewType: 'koch',
    color: 'var(--col-koch)',
  },
  {
    path: '/double-pendulum',
    title: 'Double Pendulum',
    category: 'Attractor',
    description:
      'Two linked pendulums with unpredictable, swirling motion. A deceptively simple mechanical system that produces genuinely chaotic trajectories.',
    previewType: 'pendulum',
    color: 'var(--col-pendulum)',
  },
  {
    path: '/cellular-automata',
    title: 'Cellular Automata',
    category: 'Cellular Automaton',
    description:
      "Twelve rulesets - Conway's Life, Langton's Ant, elementary 1D rules, Brian's Brain, and 2D outer-totalistic automata. Simple local rules, endlessly varied emergent behavior.",
    previewType: 'cellular',
    color: 'var(--col-ca)',
  },
  {
    path: '/three-body',
    title: 'Three Body Problem',
    category: 'N-Body',
    description:
      'Three massive bodies attracting each other through gravity. While the two-body problem is solvable, the three-body problem is generically chaotic - tiny changes in initial conditions lead to wildly different outcomes.',
    previewType: 'threebody',
    color: 'var(--col-three-body)',
  },
  {
    path: '/reaction-diffusion',
    title: 'Reaction Diffusion',
    category: 'Turing Pattern',
    description:
      'Two chemicals diffusing and reacting via the Gray-Scott equations spontaneously break symmetry into spots, stripes, coral, and mazes - a mathematical model of animal markings and biological morphogenesis.',
    previewType: 'reaction',
    color: 'var(--col-reaction)',
  },
  {
    path: '/barnsley-fern',
    title: 'Barnsley Fern',
    category: 'IFS / Chaos Game',
    description:
      'Five Iterated Function Systems rendered via the chaos game: Barnsley Fern, Sierpiński Triangle, Heighway Dragon, Lévy C Curve, and Fractal Tree. Organic self-similarity from a handful of affine transforms.',
    previewType: 'fern',
    color: 'var(--col-fern)',
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
              <MiniPreview type={exp.previewType} />
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
