import { useId } from 'react';
import styles from './NumericInput.module.css';

interface NumericInputProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
}

function clamp(v: number, min?: number, max?: number): number {
  let n = v;
  if (min !== undefined) n = Math.max(min, n);
  if (max !== undefined) n = Math.min(max, n);
  return n;
}

export default function NumericInput({
  label, value, onChange, min, max, step = 1, unit,
}: NumericInputProps) {
  const id = useId();

  return (
    <div className={styles.control}>
      <label htmlFor={id} className={styles.label}>{label}</label>
      <div className={styles.row}>
        <div className={styles.inputWrap}>
          <input
            id={id}
            type="number"
            className={styles.input}
            value={value}
            min={min}
            max={max}
            step={step}
            onChange={e => {
              const v = Number(e.target.value);
              if (!isNaN(v)) onChange(clamp(v, min, max));
            }}
          />
          {unit && <span className={styles.unit}>{unit}</span>}
        </div>
        {(min !== undefined || max !== undefined) && (
          <span className={styles.range}>
            {min !== undefined && <span>{min}</span>}
            <span className={styles.sep}>–</span>
            {max !== undefined && <span>{max}</span>}
          </span>
        )}
      </div>
    </div>
  );
}
