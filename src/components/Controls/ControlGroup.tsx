import type { ReactNode } from 'react';
import styles from './ControlGroup.module.css';

interface ControlGroupProps {
  label?: string;
  children: ReactNode;
}

export default function ControlGroup({ label, children }: ControlGroupProps) {
  return (
    <div className={styles.group}>
      {label && <span className={styles.label}>{label}</span>}
      <div className={styles.controls}>{children}</div>
    </div>
  );
}
