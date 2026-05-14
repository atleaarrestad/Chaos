import { NavLink } from 'react-router-dom';
import styles from './Sidebar.module.css';

interface NavItem {
  path: string;
  label: string;
  symbol: string;
}

interface NavSection {
  section: string;
  items: NavItem[];
}

const NAV: NavSection[] = [
  {
    section: 'Attractors',
    items: [
      { path: '/lorenz', label: 'Lorenz Attractor', symbol: 'σ' },
      { path: '/double-pendulum', label: 'Double Pendulum', symbol: 'g' },
    ],
  },
  {
    section: 'Fractals',
    items: [
      { path: '/mandelbrot',      label: 'Mandelbrot Set',     symbol: '∞' },
      { path: '/koch',            label: 'Koch Snowflake',     symbol: '❄' },
    ],
  },
  {
    section: 'Curves & Diagrams',
    items: [
      { path: '/cardioid', label: 'Cardioid', symbol: 'θ' },
      { path: '/bifurcation', label: 'Bifurcation', symbol: 'δ' },
    ],
  },
  {
    section: 'Development',
    items: [
      { path: '/controls-test', label: 'Controls Test', symbol: '⚙' },
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
          <span className={styles.symbol}>⌂</span>
          Home
        </NavLink>

        {NAV.map(({ section, items }) => (
          <div key={section} className={styles.group}>
            <span className={styles.sectionLabel}>{section}</span>
            {items.map(({ path, label, symbol }) => (
              <NavLink
                key={path}
                to={path}
                className={({ isActive }) =>
                  [styles.link, isActive ? styles.active : ''].join(' ')
                }
              >
                <span className={styles.symbol}>{symbol}</span>
                {label}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>
    </aside>
  );
}
