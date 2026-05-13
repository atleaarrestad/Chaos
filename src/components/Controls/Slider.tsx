import type { CSSProperties } from 'react';
import styles from './Slider.module.css';

interface SliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  format?: (v: number) => string;
}

function autoDecimals(step: number): number {
  return step < 1 ? Math.max(0, Math.ceil(-Math.log10(step))) : 0;
}

export default function Slider({
  label, value, onChange, min, max, step = 1, unit, format,
}: SliderProps) {
  const fill = (value - min) / (max - min);
  const display = format ? format(value) : value.toFixed(autoDecimals(step));

  return (
    <div className={styles.control}>
      <div className={styles.header}>
        <span className={styles.label}>{label}</span>
        <span className={styles.value}>
          {display}
          {unit && <span className={styles.unit}> {unit}</span>}
        </span>
      </div>
      <input
        type="range"
        className={styles.range}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ '--fill': fill } as CSSProperties}
      />
    </div>
  );
}
