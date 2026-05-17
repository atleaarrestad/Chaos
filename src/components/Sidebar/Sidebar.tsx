import { NavLink } from 'react-router-dom';
import styles from './Sidebar.module.css';

interface NavItem {
  path: string;
  label: string;
  symbol: string;
  color?: string;
}

interface NavSection {
  section: string;
  items: NavItem[];
}

const NAV: NavSection[] = [
  {
    section: 'Attractors',
    items: [
      { path: '/lorenz',          label: 'Strange Attractors', symbol: '🌀', color: 'var(--col-lorenz)' },
      { path: '/double-pendulum', label: 'Double Pendulum',    symbol: '⇌',  color: 'var(--col-pendulum)' },
      { path: '/three-body',      label: 'Three Body Problem', symbol: '🪐', color: 'var(--col-three-body)' },
    ],
  },
  {
    section: 'Fractals',
    items: [
      { path: '/mandelbrot',    label: 'Mandelbrot Set', symbol: '∞',  color: 'var(--col-mandelbrot)' },
      { path: '/koch',          label: 'Koch Snowflake', symbol: '❄️', color: 'var(--col-koch)' },
      { path: '/barnsley-fern', label: 'Barnsley Fern',  symbol: '🌿', color: 'var(--col-fern)' },
    ],
  },
  {
    section: 'Curves & Diagrams',
    items: [
      { path: '/cardioid',    label: 'Cardioid',     symbol: '♥', color: 'var(--col-cardioid)' },
      { path: '/bifurcation', label: 'Bifurcation',  symbol: '⌥', color: 'var(--col-bifurcation)' },
    ],
  },
  {
    section: 'Cellular Automata',
    items: [
      { path: '/cellular-automata', label: 'Cellular Automata', symbol: '⊞', color: 'var(--col-ca)' },
    ],
  },
  {
    section: 'Emergence',
    items: [
      { path: '/reaction-diffusion', label: 'Reaction Diffusion', symbol: '⬡', color: 'var(--col-reaction)' },
    ],
  },
];

export default function Sidebar() {
  return (
    <aside className={styles.sidebar}>
      <NavLink to="/" className={styles.logo}>
        <span className={styles.logoSymbol}>✦</span>
        Chaos
      </NavLink>

      <nav className={styles.nav}>
        <NavLink
          to="/"
          end
          className={({ isActive }) =>
            [styles.link, isActive ? styles.active : ''].join(' ')
          }
        >
          <span className={styles.symbol}>🏠</span>
          Home
        </NavLink>

        {NAV.map(({ section, items }) => (
          <div key={section} className={styles.group}>
            <span className={styles.sectionLabel}>{section}</span>
            {items.map(({ path, label, symbol, color }) => (
              <NavLink
                key={path}
                to={path}
                className={({ isActive }) =>
                  [styles.link, isActive ? styles.active : ''].join(' ')
                }
              >
                <span className={styles.symbol} style={color ? { color } : undefined}>{symbol}</span>
                {label}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>
    </aside>
  );
}

