import { type CSSProperties, useState, useEffect } from 'react';
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
  manualInput?: boolean;
}

function autoDecimals(step: number): number {
  return step < 1 ? Math.max(0, Math.ceil(-Math.log10(step))) : 0;
}

export default function Slider({
  label, value, onChange, min, max, step = 1, unit, format, manualInput,
}: SliderProps) {
  const fill = Math.min(1, Math.max(0, (value - min) / (max - min)));
  const decimals = autoDecimals(step);
  const display = format ? format(value) : value.toFixed(decimals);

  const [raw, setRaw] = useState(display);
  const [editing, setEditing] = useState(false);

  useEffect(() => {
    if (!editing) setRaw(format ? format(value) : value.toFixed(decimals));
  }, [value, editing, format, decimals]);

  function commitRaw(s: string) {
    const n = parseFloat(s);
    if (!isNaN(n)) onChange(Math.min(max, Math.max(min, n)));
    setEditing(false);
  }

  return (
    <div className={styles.control}>
      <div className={styles.header}>
        <span className={styles.label}>{label}</span>
        {manualInput ? (
          <div className={styles.manualWrap}>
            <input
              type="number"
              className={styles.manualInput}
              value={raw}
              min={min}
              max={max}
              step={step}
              onChange={e => { setEditing(true); setRaw(e.target.value); }}
              onBlur={e => commitRaw(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') commitRaw((e.target as HTMLInputElement).value); }}
            />
            {unit && <span className={styles.unit}> {unit}</span>}
          </div>
        ) : (
          <span className={styles.value}>
            {display}
            {unit && <span className={styles.unit}> {unit}</span>}
          </span>
        )}
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
