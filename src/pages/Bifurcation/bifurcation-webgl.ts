// bifurcation-webgl.ts — WebGL2 renderer for the logistic-map bifurcation diagram.
// Each fragment independently iterates x → r·x·(1−x) and counts how many
// iterated values land in that pixel's y-bucket, so the whole diagram is drawn
// in a single draw call on the GPU.

export type ColorSchemeId = 'cyan' | 'heat' | 'plasma' | 'mono';

export interface GLRenderParams {
  rMin: number;
  rMax: number;
  yMin: number;
  yMax: number;
  iterations: number;
  burnin: number;
  colorScheme: ColorSchemeId;
  logScale: boolean;
}

export interface WebGLBifurcationRenderer {
  render(params: GLRenderParams): void;
  dispose(): void;
}

// ─── Shaders ──────────────────────────────────────────────────────────────────

const VS_SRC = /* glsl */`#version 300 es
in vec2 a_pos;
void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }
`;

// Fragment shader: each fragment = one pixel.
// Computes r from its x-column, burns in, then counts iterations that land in
// its y-row, and maps that count to a colour.
const FS_SRC = /* glsl */`#version 300 es
precision highp float;
precision highp int;

uniform float u_rMin;
uniform float u_rMax;
uniform float u_yMin;
uniform float u_yMax;
uniform int   u_burnin;
uniform int   u_iterations;
uniform int   u_colorScheme; // 0=cyan 1=heat 2=plasma 3=mono
uniform int   u_logScale;    // 0=false 1=true
uniform vec2  u_resolution;

out vec4 fragColor;

const vec3 BG = vec3(7.0 / 255.0, 7.0 / 255.0, 18.0 / 255.0);

void main() {
  float W  = u_resolution.x;
  float H  = u_resolution.y;
  // gl_FragCoord is in pixel-centre coords, bottom-left origin.
  float px = gl_FragCoord.x - 0.5;
  float py = gl_FragCoord.y - 0.5;

  // r value for this column
  float r = u_rMin + (px / max(W - 1.0, 1.0)) * (u_rMax - u_rMin);

  // Logistic-map x-value this row represents.
  // py=0 (bottom) → u_yMin,  py=H-1 (top) → u_yMax
  float yRange   = u_yMax - u_yMin;
  float xCenter  = u_yMin + (py / max(H - 1.0, 1.0)) * yRange;
  float halfPx   = 0.5 * yRange / max(H - 1.0, 1.0);

  // Burn-in (fixed upper bound, break on condition — safe on all WebGL2 drivers)
  float x = 0.5;
  for (int i = 0; i < 1024; i++) {
    if (i >= u_burnin) break;
    x = r * x * (1.0 - x);
  }

  // Count how many iterated values land in this pixel's y-band
  int count = 0;
  for (int i = 0; i < 1024; i++) {
    if (i >= u_iterations) break;
    x = r * x * (1.0 - x);
    if (abs(x - xCenter) < halfPx) count++;
  }

  if (count == 0) {
    fragColor = vec4(BG, 1.0);
    return;
  }

  // Normalise — GPU can't do a global-max reduction cheaply, so we normalise
  // against u_iterations (max possible count). With log-scale this is visually
  // indistinguishable from the CPU path's global-max normalisation.
  float maxC = float(u_iterations);
  float t = (u_logScale == 1)
    ? log(float(count) + 1.0) / log(maxC + 1.0)
    : float(count) / maxC;

  vec3 col;
  if (u_colorScheme == 0) {
    // cyan: background → dim cyan → bright cyan-white
    float s = 0.15 + 0.85 * t;
    col = BG + (vec3(34.0, 211.0, 238.0) / 255.0 - BG) * s;

  } else if (u_colorScheme == 1) {
    // heat: black → red → yellow → white
    if (t < 0.33) {
      float s = t / 0.33;
      col = vec3(180.0 / 255.0 * s, 0.0, 0.0);
    } else if (t < 0.67) {
      float s = (t - 0.33) / 0.34;
      col = vec3((180.0 + 75.0 * s) / 255.0, 180.0 / 255.0 * s, 0.0);
    } else {
      float s = (t - 0.67) / 0.33;
      col = vec3(1.0, (180.0 + 75.0 * s) / 255.0, 240.0 / 255.0 * s);
    }

  } else if (u_colorScheme == 2) {
    // plasma: dark-purple → magenta → orange → yellow
    if (t < 0.5) {
      float s = t * 2.0;
      col = vec3((100.0 + 155.0 * s) / 255.0, 0.0, 200.0 / 255.0 * (1.0 - s));
    } else {
      float s = (t - 0.5) * 2.0;
      col = vec3(1.0, 200.0 / 255.0 * s, 0.0);
    }

  } else {
    // mono
    col = vec3(t);
  }

  fragColor = vec4(col, 1.0);
}
`;

// ─── Helpers ──────────────────────────────────────────────────────────────────

function compileShader(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Bifurcation shader compile error:\n${log}`);
  }
  return shader;
}

const CS_INDEX: Record<string, number> = { cyan: 0, heat: 1, plasma: 2, mono: 3 };

// ─── Public API ───────────────────────────────────────────────────────────────

export function detectWebGL(): boolean {
  try {
    const c = document.createElement('canvas');
    const gl = c.getContext('webgl2');
    if (!gl) return false;
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return true;
  } catch {
    return false;
  }
}

export function createWebGLRenderer(canvas: HTMLCanvasElement): WebGLBifurcationRenderer | null {
  const gl = canvas.getContext('webgl2', {
    alpha: false, antialias: false, depth: false, stencil: false,
  });
  if (!gl) return null;

  let vs: WebGLShader, fs: WebGLShader, prog: WebGLProgram;
  try {
    vs   = compileShader(gl, gl.VERTEX_SHADER,   VS_SRC);
    fs   = compileShader(gl, gl.FRAGMENT_SHADER, FS_SRC);
    prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS))
      throw new Error(gl.getProgramInfoLog(prog) ?? 'link error');
  } catch (err) {
    console.error('[bifurcation-webgl]', err);
    return null;
  }

  // Full-screen quad (two triangles)
  const quadBuf = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1,   1, -1,  -1,  1,
    -1,  1,   1, -1,   1,  1,
  ]), gl.STATIC_DRAW);

  const aPos    = gl.getAttribLocation(prog,  'a_pos');
  const uRMin   = gl.getUniformLocation(prog, 'u_rMin');
  const uRMax   = gl.getUniformLocation(prog, 'u_rMax');
  const uYMin   = gl.getUniformLocation(prog, 'u_yMin');
  const uYMax   = gl.getUniformLocation(prog, 'u_yMax');
  const uBurnin = gl.getUniformLocation(prog, 'u_burnin');
  const uIter   = gl.getUniformLocation(prog, 'u_iterations');
  const uCS     = gl.getUniformLocation(prog, 'u_colorScheme');
  const uLog    = gl.getUniformLocation(prog, 'u_logScale');
  const uRes    = gl.getUniformLocation(prog, 'u_resolution');

  return {
    render({ rMin, rMax, yMin, yMax, iterations, burnin, colorScheme, logScale }) {
      const W = canvas.width, H = canvas.height;
      if (W === 0 || H === 0) return;
      gl.viewport(0, 0, W, H);
      gl.useProgram(prog);
      gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
      gl.uniform1f(uRMin,   rMin);
      gl.uniform1f(uRMax,   rMax);
      gl.uniform1f(uYMin,   yMin);
      gl.uniform1f(uYMax,   yMax);
      gl.uniform1i(uBurnin, burnin);
      gl.uniform1i(uIter,   iterations);
      gl.uniform1i(uCS,     CS_INDEX[colorScheme] ?? 0);
      gl.uniform1i(uLog,    logScale ? 1 : 0);
      gl.uniform2f(uRes,    W, H);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    },
    dispose() {
      gl.deleteBuffer(quadBuf);
      gl.deleteProgram(prog);
      gl.deleteShader(fs);
      gl.deleteShader(vs);
    },
  };
}
