/** Detect WebGL2 support without throwing. */
export function detectWebGL2(): boolean {
  try {
    const c = document.createElement('canvas');
    return !!c.getContext('webgl2');
  } catch {
    return false;
  }
}

/** Create a WebGL2 context with the given options, returns null on failure. */
export function createWebGL2Context(
  canvas: HTMLCanvasElement,
  opts?: WebGLContextAttributes,
): WebGL2RenderingContext | null {
  try {
    return canvas.getContext('webgl2', opts) as WebGL2RenderingContext | null;
  } catch {
    return null;
  }
}
