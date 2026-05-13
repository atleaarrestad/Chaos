import { useState } from 'react';
import {
  Slider, NumericInput, Toggle, SelectControl,
  ControlPanel, ControlGroup,
} from '@/components/Controls';
import styles from './ControlsTest.module.css';

type ColorScheme = 'classic' | 'heat' | 'grayscale' | 'plasma';
type Integrator  = 'rk4' | 'euler' | 'midpoint';

export default function ControlsTest() {
  /* ── Equation params ── */
  const [sigma, setSigma]       = useState(10);
  const [rho, setRho]           = useState(28);
  const [beta, setBeta]         = useState(2.667);
  const [dt, setDt]             = useState(0.005);

  /* ── Simulation params ── */
  const [iterations, setIterations] = useState(2000);
  const [trailLength, setTrailLength] = useState(500);
  const [integrator, setIntegrator]   = useState<Integrator>('rk4');

  /* ── Display params ── */
  const [animate, setAnimate]       = useState(true);
  const [showTrail, setShowTrail]   = useState(true);
  const [showAxes, setShowAxes]     = useState(false);
  const [colorScheme, setColorScheme] = useState<ColorScheme>('classic');

  const state = { sigma, rho, beta, dt, iterations, trailLength, integrator, animate, showTrail, showAxes, colorScheme };

  return (
    <div className={styles.page}>
      <header className={styles.hero}>
        <p className={styles.eyebrow}>Development</p>
        <h1 className={styles.title}>Controls Test</h1>
        <p className={styles.subtitle}>
          Reusable parameter panels — Slider, NumericInput, Toggle, SelectControl, ControlPanel, ControlGroup.
        </p>
      </header>

      <div className={styles.layout}>
        {/* ── Left: panels ── */}
        <div className={styles.panels}>
          <ControlPanel title="Equation Parameters">
            <ControlGroup label="Lorenz Coefficients">
              <Slider
                label="σ — sigma"
                value={sigma} onChange={setSigma}
                min={0} max={20} step={0.1}
              />
              <Slider
                label="ρ — rho"
                value={rho} onChange={setRho}
                min={0} max={60} step={0.5}
              />
              <Slider
                label="β — beta"
                value={beta} onChange={setBeta}
                min={0} max={10} step={0.001}
                format={v => v.toFixed(3)}
              />
            </ControlGroup>
            <ControlGroup label="Time Step">
              <Slider
                label="dt"
                value={dt} onChange={setDt}
                min={0.0001} max={0.02} step={0.0001}
                format={v => v.toFixed(4)}
              />
            </ControlGroup>
          </ControlPanel>

          <ControlPanel title="Simulation">
            <ControlGroup>
              <NumericInput
                label="Iterations per frame"
                value={iterations} onChange={setIterations}
                min={100} max={50000} step={100}
              />
              <NumericInput
                label="Trail length"
                value={trailLength} onChange={setTrailLength}
                min={10} max={5000} step={10}
              />
              <SelectControl
                label="Integrator"
                value={integrator} onChange={setIntegrator}
                options={[
                  { value: 'rk4',     label: 'Runge–Kutta 4 (RK4)' },
                  { value: 'midpoint', label: 'Midpoint method'      },
                  { value: 'euler',   label: 'Euler (forward)'      },
                ]}
              />
            </ControlGroup>
          </ControlPanel>

          <ControlPanel title="Display" defaultOpen={false}>
            <ControlGroup>
              <Toggle
                label="Animate"
                value={animate} onChange={setAnimate}
                description="Run the continuous simulation loop"
              />
              <Toggle
                label="Show trail"
                value={showTrail} onChange={setShowTrail}
                description="Draw the trajectory history"
              />
              <Toggle
                label="Show axes"
                value={showAxes} onChange={setShowAxes}
                description="Overlay X / Y / Z reference lines"
              />
              <SelectControl
                label="Color scheme"
                value={colorScheme} onChange={setColorScheme}
                options={[
                  { value: 'classic',   label: 'Classic'   },
                  { value: 'heat',      label: 'Heat map'  },
                  { value: 'grayscale', label: 'Grayscale' },
                  { value: 'plasma',    label: 'Plasma'    },
                ]}
              />
            </ControlGroup>
          </ControlPanel>
        </div>

        {/* ── Right: live state preview ── */}
        <div className={styles.preview}>
          <div className={styles.previewTitle}>Live State</div>
          <pre className={styles.json}>{JSON.stringify(state, null, 2)}</pre>
        </div>
      </div>
    </div>
  );
}
