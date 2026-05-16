import { createProgram } from '../../lib/gpu/shader';

// ─── GLSL sources ─────────────────────────────────────────────────────────────

const VERT_SRC = /* glsl */`#version 300 es
// Full-screen triangle pair — no vertex buffer needed, positions come from gl_VertexID.
const vec2 CORNERS[6] = vec2[](
  vec2(-1.,-1.), vec2(1.,-1.), vec2(-1.,1.),
  vec2(-1., 1.), vec2(1.,-1.), vec2( 1.,1.)
);
void main() {
  gl_Position = vec4(CORNERS[gl_VertexID], 0.0, 1.0);
}
`;

const STEP_FRAG_SRC = /* glsl */`#version 300 es
precision highp float;
precision highp int;
uniform sampler2D u_prev;
uniform ivec2 u_size;   // (cols, rows)
out vec4 fragColor;

void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  int live = 0;
  for (int dr = -1; dr <= 1; dr++) {
    for (int dc = -1; dc <= 1; dc++) {
      if (dr == 0 && dc == 0) continue;
      ivec2 nc = ((coord + ivec2(dc, dr)) % u_size + u_size) % u_size;
      if (texelFetch(u_prev, nc, 0).r > 0.5) live++;
    }
  }
  vec4 cur   = texelFetch(u_prev, coord, 0);
  bool was   = cur.r > 0.5;
  bool now_  = was ? (live == 2 || live == 3) : (live == 3);
  float age  = now_ ? min(cur.g + 1.0/255.0, 1.0) : 0.0;
  fragColor  = vec4(now_ ? 1.0 : 0.0, age, 0.0, 1.0);
}
`;

const RENDER_FRAG_SRC = /* glsl */`#version 300 es
precision highp float;
precision highp int;
uniform sampler2D u_state;
uniform vec2  u_viewOrigin;   // (viewX, viewY) in cell-space
uniform float u_cellSizePx;   // physical px per cell (CSS px × dpr)
uniform float u_canvasH;      // canvas height in physical px
uniform int   u_cols;
uniform int   u_rows;
uniform bool  u_showGrid;
uniform bool  u_useAgeColor;
out vec4 fragColor;

vec4 ageColor(float normAge) {
  float a = normAge * 255.0;
  if (a <= 1.0) return vec4(0.941, 0.992, 0.976, 1.0); // #f0fdf9
  if (a <= 3.0) return vec4(0.655, 0.953, 0.816, 1.0); // #a7f3d0
  if (a <= 9.0) return vec4(0.431, 0.906, 0.718, 1.0); // #6ee7b7
  return          vec4(0.204, 0.831, 0.600, 1.0);       // #34d399
}

void main() {
  // Convert gl_FragCoord (y=0 at bottom) to screen-top-down coords
  float px = gl_FragCoord.x;
  float py = u_canvasH - gl_FragCoord.y;

  float cellFx = u_viewOrigin.x + px / u_cellSizePx;
  float cellFy = u_viewOrigin.y + py / u_cellSizePx;

  ivec2 cell = ((ivec2(floor(cellFx), floor(cellFy)) % ivec2(u_cols, u_rows))
                + ivec2(u_cols, u_rows)) % ivec2(u_cols, u_rows);

  float fracX   = fract(cellFx);
  float fracY   = fract(cellFy);
  float gapFrac = 1.0 / u_cellSizePx;
  bool  zoomed  = u_cellSizePx >= 6.0;

  // 1-pixel gap on the left and top edges of each cell when zoomed in
  if (zoomed && (fracX < gapFrac || fracY < gapFrac)) {
    if (u_showGrid && u_cellSizePx >= 8.0) {
      // rgba(255,255,255,0.06) blended over #070712
      fragColor = vec4(0.087, 0.087, 0.131, 1.0);
    } else {
      fragColor = vec4(0.027, 0.027, 0.071, 1.0); // #070712 BG_COLOR
    }
    return;
  }

  vec4 texel = texelFetch(u_state, cell, 0);
  bool alive = texel.r > 0.5;

  if (!alive) {
    fragColor = vec4(0.047, 0.047, 0.118, 1.0); // #0c0c1e DEAD_COLOR
    return;
  }

  fragColor = u_useAgeColor ? ageColor(texel.g) : vec4(0.431, 0.906, 0.718, 1.0);
}
`;

// ─── ConwayGPU ────────────────────────────────────────────────────────────────

export class ConwayGPU {
  private gl: WebGL2RenderingContext;
  private cols: number;
  private rows: number;

  private stepProg:   WebGLProgram;
  private renderProg: WebGLProgram;
  private vao:        WebGLVertexArrayObject;

  private texA: WebGLTexture;
  private texB: WebGLTexture;
  private fboA: WebGLFramebuffer;
  private fboB: WebGLFramebuffer;

  // Which texture is the current live state
  private current: 'A' | 'B' = 'A';

  // Step uniforms
  private su_prev: WebGLUniformLocation;
  private su_size: WebGLUniformLocation;

  // Render uniforms
  private ru_state:      WebGLUniformLocation;
  private ru_viewOrigin: WebGLUniformLocation;
  private ru_cellSizePx: WebGLUniformLocation;
  private ru_canvasH:    WebGLUniformLocation;
  private ru_cols:       WebGLUniformLocation;
  private ru_rows:       WebGLUniformLocation;
  private ru_showGrid:   WebGLUniformLocation;
  private ru_useAgeColor:WebGLUniformLocation;

  constructor(gl: WebGL2RenderingContext, cols: number, rows: number) {
    this.gl   = gl;
    this.cols = cols;
    this.rows = rows;

    this.stepProg   = createProgram(gl, VERT_SRC, STEP_FRAG_SRC);
    this.renderProg = createProgram(gl, VERT_SRC, RENDER_FRAG_SRC);

    // Step uniforms
    this.su_prev = gl.getUniformLocation(this.stepProg, 'u_prev')!;
    this.su_size = gl.getUniformLocation(this.stepProg, 'u_size')!;

    // Render uniforms
    this.ru_state       = gl.getUniformLocation(this.renderProg, 'u_state')!;
    this.ru_viewOrigin  = gl.getUniformLocation(this.renderProg, 'u_viewOrigin')!;
    this.ru_cellSizePx  = gl.getUniformLocation(this.renderProg, 'u_cellSizePx')!;
    this.ru_canvasH     = gl.getUniformLocation(this.renderProg, 'u_canvasH')!;
    this.ru_cols        = gl.getUniformLocation(this.renderProg, 'u_cols')!;
    this.ru_rows        = gl.getUniformLocation(this.renderProg, 'u_rows')!;
    this.ru_showGrid    = gl.getUniformLocation(this.renderProg, 'u_showGrid')!;
    this.ru_useAgeColor = gl.getUniformLocation(this.renderProg, 'u_useAgeColor')!;

    // Empty VAO — no vertex attributes needed (gl_VertexID only)
    this.vao = gl.createVertexArray()!;

    this.texA = this.makeTexture();
    this.texB = this.makeTexture();
    this.fboA = this.makeFbo(this.texA);
    this.fboB = this.makeFbo(this.texB);
  }

  private makeTexture(): WebGLTexture {
    const { gl, cols, rows } = this;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, cols, rows, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    return tex;
  }

  private makeFbo(tex: WebGLTexture): WebGLFramebuffer {
    const { gl } = this;
    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return fbo;
  }

  private get curTex():  WebGLTexture     { return this.current === 'A' ? this.texA : this.texB; }
  private get curFbo():  WebGLFramebuffer  { return this.current === 'A' ? this.fboA : this.fboB; }
  private get nextFbo(): WebGLFramebuffer  { return this.current === 'A' ? this.fboB : this.fboA; }

  /** Upload CPU grid state into the current texture. */
  uploadState(grid: Uint8Array, ages: Uint16Array): void {
    const { gl, cols, rows } = this;
    const data = new Uint8Array(cols * rows * 4);
    for (let i = 0; i < cols * rows; i++) {
      data[i*4+0] = grid[i] ? 255 : 0;
      data[i*4+1] = Math.min(ages[i], 255);
      data[i*4+3] = 255;
    }
    gl.bindTexture(gl.TEXTURE_2D, this.curTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, cols, rows, gl.RGBA, gl.UNSIGNED_BYTE, data);
  }

  /** Toggle a single cell; alive=1/0, age in generations. */
  setCell(col: number, row: number, alive: number, age: number): void {
    const { gl } = this;
    const pixel = new Uint8Array([alive ? 255 : 0, Math.min(age, 255), 0, 255]);
    gl.bindTexture(gl.TEXTURE_2D, this.curTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, col, row, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
  }

  /** Read whether cell (col, row) is alive in the current state. */
  readCell(col: number, row: number): boolean {
    const { gl } = this;
    const pixel = new Uint8Array(4);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.curFbo);
    gl.readPixels(col, row, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    return pixel[0] > 127;
  }

  /** Count alive cells by reading back the full texture. */
  readPopulation(): number {
    const { gl, cols, rows } = this;
    const data = new Uint8Array(cols * rows * 4);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.curFbo);
    gl.readPixels(0, 0, cols, rows, gl.RGBA, gl.UNSIGNED_BYTE, data);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    let count = 0;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i] > 127) count++;
    }
    return count;
  }

  /** Advance one generation via ping-pong FBO. */
  step(): void {
    const { gl, cols, rows } = this;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.nextFbo);
    gl.viewport(0, 0, cols, rows);
    gl.useProgram(this.stepProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.curTex);
    gl.uniform1i(this.su_prev, 0);
    gl.uniform2i(this.su_size, cols, rows);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    this.current = this.current === 'A' ? 'B' : 'A';
  }

  /**
   * Render the current state to the canvas default framebuffer.
   * @param viewX     viewport left in cell-space
   * @param viewY     viewport top in cell-space
   * @param cellSizeCss cell size in CSS logical pixels
   * @param showGrid  show grid lines when paused and zoomed
   * @param useAgeColor age-based colour tiers vs flat green
   */
  render(
    viewX: number,
    viewY: number,
    cellSizeCss: number,
    showGrid: boolean,
    useAgeColor: boolean,
  ): void {
    const { gl } = this;
    const dpr = window.devicePixelRatio || 1;
    const W   = gl.drawingBufferWidth;
    const H   = gl.drawingBufferHeight;

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, W, H);
    gl.useProgram(this.renderProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.curTex);
    gl.uniform1i (this.ru_state,       0);
    gl.uniform2f (this.ru_viewOrigin,  viewX, viewY);
    gl.uniform1f (this.ru_cellSizePx,  cellSizeCss * dpr);
    gl.uniform1f (this.ru_canvasH,     H);
    gl.uniform1i (this.ru_cols,        this.cols);
    gl.uniform1i (this.ru_rows,        this.rows);
    gl.uniform1i (this.ru_showGrid,    showGrid    ? 1 : 0);
    gl.uniform1i (this.ru_useAgeColor, useAgeColor ? 1 : 0);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  destroy(): void {
    const { gl } = this;
    gl.deleteProgram(this.stepProg);
    gl.deleteProgram(this.renderProg);
    gl.deleteVertexArray(this.vao);
    gl.deleteTexture(this.texA);
    gl.deleteTexture(this.texB);
    gl.deleteFramebuffer(this.fboA);
    gl.deleteFramebuffer(this.fboB);
  }
}
