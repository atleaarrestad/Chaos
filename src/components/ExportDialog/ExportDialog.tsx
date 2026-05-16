import { useState, useEffect } from 'react';
import type { ExportFormat } from '../../lib/exportImage';
import styles from './ExportDialog.module.css';

const SIZES = [
  { label: 'HD',  w: 1280, h: 720  },
  { label: 'FHD', w: 1920, h: 1080 },
  { label: 'QHD', w: 2560, h: 1440 },
  { label: '4K',  w: 3840, h: 2160 },
  { label: '8K',  w: 7680, h: 4320 },
] as const;

const FORMATS: { label: string; value: ExportFormat }[] = [
  { label: 'PNG',  value: 'png'  },
  { label: 'WebP', value: 'webp' },
  { label: 'JPEG', value: 'jpeg' },
  { label: 'BMP',  value: 'bmp'  },
];

export interface ExportOpts {
  width:  number;
  height: number;
  format: ExportFormat;
}

interface Props {
  onClose:      () => void;
  onDownload:   (opts: ExportOpts) => void | Promise<void>;
  isRendering?: boolean;
}

export default function ExportDialog({ onClose, onDownload, isRendering = false }: Props) {
  const [size,   setSize]   = useState<(typeof SIZES)[number]>(SIZES[1]); // FHD default
  const [format, setFormat] = useState<ExportFormat>('png');

  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [onClose]);

  const handleDownload = async () => {
    await onDownload({ width: size.w, height: size.h, format });
    onClose();
  };

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.dialog} onClick={e => e.stopPropagation()}>

        <div className={styles.header}>
          <span className={styles.title}>Export Image</span>
          <button className={styles.closeBtn} onClick={onClose}>×</button>
        </div>

        <div className={styles.body}>
          <div className={styles.section}>
            <span className={styles.sectionLabel}>Size</span>
            <div className={styles.optionRow}>
              {SIZES.map(s => (
                <button
                  key={s.label}
                  className={[styles.optBtn, size.label === s.label ? styles.optBtnActive : ''].join(' ')}
                  onClick={() => setSize(s)}
                >
                  <span className={styles.optBtnLabel}>{s.label}</span>
                  <span className={styles.optBtnSub}>{s.w}×{s.h}</span>
                </button>
              ))}
            </div>
          </div>

          <div className={styles.section}>
            <span className={styles.sectionLabel}>Format</span>
            <div className={styles.optionRow}>
              {FORMATS.map(f => (
                <button
                  key={f.value}
                  className={[styles.optBtn, format === f.value ? styles.optBtnActive : ''].join(' ')}
                  onClick={() => setFormat(f.value)}
                >
                  <span className={styles.optBtnLabel}>{f.label}</span>
                </button>
              ))}
            </div>
          </div>

          <button
            className={styles.downloadBtn}
            onClick={handleDownload}
            disabled={isRendering}
          >
            {isRendering ? '⏳ Rendering…' : `↓ Download ${format.toUpperCase()}`}
          </button>
        </div>

      </div>
    </div>
  );
}
