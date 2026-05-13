import { useState, type ReactNode } from 'react';
import styles from './ControlPanel.module.css';

interface ControlPanelProps {
  title: string;
  children: ReactNode;
  defaultOpen?: boolean;
}

export default function ControlPanel({ title, children, defaultOpen = true }: ControlPanelProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className={styles.panel}>
      <button
        type="button"
        className={styles.header}
        onClick={() => setOpen(v => !v)}
        aria-expanded={open}
      >
        <span className={styles.title}>{title}</span>
        <span className={[styles.chevron, open ? styles.open : ''].join(' ')}>▾</span>
      </button>
      <div className={[styles.body, open ? styles.bodyOpen : ''].join(' ')}>
        <div className={styles.bodyInner}>{children}</div>
      </div>
    </div>
  );
}
