import styles from './Toggle.module.css';

interface ToggleProps {
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
  description?: string;
}

export default function Toggle({ label, value, onChange, description }: ToggleProps) {
  return (
    <div className={styles.control}>
      <div className={styles.text}>
        <span className={styles.label}>{label}</span>
        {description && <span className={styles.desc}>{description}</span>}
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={value}
        className={[styles.track, value ? styles.on : ''].join(' ')}
        onClick={() => onChange(!value)}
      >
        <span className={styles.thumb} />
      </button>
    </div>
  );
}
