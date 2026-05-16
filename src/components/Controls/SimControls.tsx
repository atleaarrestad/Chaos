import styles from './SimControls.module.css';

interface SimControlsProps {
  /** Whether the simulation is currently running/animating. Omit for static sims. */
  running?: boolean;
  /** Called when the play/pause button is clicked. Required when `running` is provided. */
  onToggle?: () => void;
  /** Called when the reset button is clicked. */
  onReset: () => void;
  /** Disable the play/pause button (e.g. feature requires GPU). */
  toggleDisabled?: boolean;
}

export default function SimControls({ running, onToggle, onReset, toggleDisabled }: SimControlsProps) {
  const hasAnimation = onToggle !== undefined;

  return (
    <div className={styles.wrap}>
      {hasAnimation && (
        <div className={styles.btnCol}>
          <button
            type="button"
            className={[styles.btn, running ? styles.btnRunning : ''].join(' ')}
            onClick={onToggle}
            disabled={toggleDisabled}
            title={running ? 'Pause (Space)' : 'Play (Space)'}
          >
            <span className={styles.icon}>{running ? '⏸' : '▶'}</span>
            <span className={styles.label}>{running ? 'Pause' : 'Play'}</span>
          </button>
          <kbd className={styles.key}>Space</kbd>
        </div>
      )}
      <div className={styles.btnCol}>
        <button
          type="button"
          className={styles.btn}
          onClick={onReset}
          title="Reset (R)"
        >
          <span className={styles.icon}>↺</span>
          <span className={styles.label}>Reset</span>
        </button>
        <kbd className={styles.key}>R</kbd>
      </div>
    </div>
  );
}
