import type { ReactNode } from 'react';
import styles from './InfoDialog.module.css';

interface Props {
  title: string;
  onClose: () => void;
  children: ReactNode;
}

export function InfoDialog({ title, onClose, children }: Props) {
  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.dialog} onClick={e => e.stopPropagation()}>
        <div className={styles.header}>
          <span className={styles.title}>{title}</span>
          <button className={styles.closeBtn} onClick={onClose}>×</button>
        </div>
        <div className={styles.body}>{children}</div>
      </div>
    </div>
  );
}
