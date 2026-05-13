import styles from './Toggle.module.css';

interface ToggleProps {
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
  description?: string;
  disabled?: boolean;
}

export default function Toggle({ label, value, onChange, description, disabled }: ToggleProps) {
  return (
    <div className={[styles.control, disabled ? styles.disabled : ''].join(' ')}>
      <div className={styles.text}>
        <span className={styles.label}>{label}</span>
        {description && <span className={styles.desc}>{description}</span>}
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={value}
        disabled={disabled}
        className={[styles.track, value ? styles.on : ''].join(' ')}
        onClick={() => onChange(!value)}
      >
        <span className={styles.thumb} />
      </button>
    </div>
  );
}
