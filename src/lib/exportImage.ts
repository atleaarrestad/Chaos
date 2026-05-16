export type ExportFormat = 'png' | 'webp' | 'jpeg' | 'bmp';

const MIME: Record<ExportFormat, string> = {
  png:  'image/png',
  webp: 'image/webp',
  jpeg: 'image/jpeg',
  bmp:  'image/bmp',
};

/** Scale `source` to `width × height`, encode as `format`, and trigger a download. */
export function exportImage(
  source: HTMLCanvasElement,
  width: number,
  height: number,
  format: ExportFormat,
  basename: string,
): void {
  const off = document.createElement('canvas');
  off.width  = width;
  off.height = height;
  const ctx = off.getContext('2d')!;
  ctx.drawImage(source, 0, 0, width, height);
  triggerDownload(off, format, `${basename}-${width}x${height}`);
}

/** Encode an already-sized canvas and trigger download (used by Mandelbrot). */
export function downloadCanvas(
  canvas: HTMLCanvasElement,
  format: ExportFormat,
  basename: string,
): void {
  triggerDownload(canvas, format, `${basename}-${canvas.width}x${canvas.height}`);
}

function triggerDownload(canvas: HTMLCanvasElement, format: ExportFormat, filename: string): void {
  const quality = format === 'jpeg' ? 0.95 : undefined;
  const url = canvas.toDataURL(MIME[format], quality);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${filename}.${format}`;
  a.click();
}
