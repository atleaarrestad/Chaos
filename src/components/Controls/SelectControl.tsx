import { useId } from 'react';
import styles from './SelectControl.module.css';

interface Option<T extends string> {
  value: T;
  label: string;
}

interface SelectControlProps<T extends string> {
  label: string;
  value: T;
  onChange: (value: T) => void;
  options: Option<T>[];
}

export default function SelectControl<T extends string>({
  label, value, onChange, options,
}: SelectControlProps<T>) {
  const id = useId();

  return (
    <div className={styles.control}>
      <label htmlFor={id} className={styles.label}>{label}</label>
      <div className={styles.selectWrap}>
        <select
          id={id}
          className={styles.select}
          value={value}
          onChange={e => onChange(e.target.value as T)}
        >
          {options.map(opt => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
        <span className={styles.arrow}>▾</span>
      </div>
    </div>
  );
}
