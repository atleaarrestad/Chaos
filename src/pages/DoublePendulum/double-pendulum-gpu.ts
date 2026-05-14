/**
 * GPU-accelerated double pendulum ensemble using WebGL2 transform feedback.
 *
 * Architecture (per frame):
 *   1. Integration pass – vertex shader integrates N pendulums with RK4.
 *      Transform feedback captures: v_newState (th1, om1, th2, om2).
 *      RASTERIZER_DISCARD on: no drawing happens.
 *   2. Scatter pass – reads current state, computes second-bob (x2,y2),
 *      renders colored points to the WebGL canvas using additive blending.
 *      Background is transparent (alpha=0) so the 2D trail canvas can
 *      accumulate frames with source-over compositing.
 *
 * Physical model: equal masses m=1, equal rods l=1.
 * Coordinates: θ measured from downward vertical (θ=0 → hanging straight down).
 * Second bob: x2 = sin(θ1)+sin(θ2),  y2 = -(cos(θ1)+cos(θ2)).
 */

import { createWebGL2Context } from '@/lib/gpu/context';
import { createProgram } from '@/lib/gpu/shader';

export const N_PENDULUMS = 16384;
const STEPS_PER_FRAME = 4;

// ─── Shaders ──────────────────────────────────────────────────────────────────

const VERT_INTEGRATE = /* glsl */`#version 300 es
precision highp float;

in vec4 a_state; // (th1, om1, th2, om2)

uniform float u_dt;
uniform float u_g;
uniform int   u_steps;

out vec4 v_newState;

// Double pendulum derivatives: m1=m2=1, l1=l2=1
vec4 dpDeriv(vec4 s) {
  float th1 = s.x, om1 = s.y, th2 = s.z, om2 = s.w;
  float d    = th1 - th2;
  float sd   = sin(d), cd = cos(d);
  float denom = 3.0 - cos(2.0 * d);
  float a1 = (-3.0*u_g*sin(th1) - u_g*sin(th1 - 2.0*th2)
              - 2.0*sd*(om2*om2 + om1*om1*cd)) / denom;
  float a2 = (2.0*sd*(2.0*om1*om1 + 2.0*u_g*cos(th1) + om2*om2*cd)) / denom;
  return vec4(om1, a1, om2, a2);
}

vec4 rk4(vec4 s) {
  vec4 k1 = dpDeriv(s);
  vec4 k2 = dpDeriv(s + k1 * (u_dt * 0.5));
  vec4 k3 = dpDeriv(s + k2 * (u_dt * 0.5));
  vec4 k4 = dpDeriv(s + k3 * u_dt);
  return s + (k1 + 2.0*k2 + 2.0*k3 + k4) * (u_dt / 6.0);
}

#define MAX_STEPS 16
void main() {
  vec4 state = a_state;
  for (int i = 0; i < MAX_STEPS; i++) {
    if (i >= u_steps) break;
    state = rk4(state);
  }
  v_newState = state;
  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}
`;

const FRAG_NOOP = /* glsl */`#version 300 es
precision mediump float;
out vec4 fragColor;
void main() { fragColor = vec4(0.0); }
`;

const VERT_SCATTER = /* glsl */`#version 300 es
precision highp float;

in vec4 a_state; // (th1, om1, th2, om2)

uniform float u_pointSize;
uniform int   u_nParticles;
uniform int   u_colorMode; // 0=heat, 1=rainbow, 2=green

out vec4 v_color; // pre-multiplied RGBA

// Heat gradient: blue → cyan → green → yellow → red
vec3 heat(float t) {
  if (t < 0.25) return mix(vec3(0.0, 0.2, 1.0), vec3(0.0, 0.9, 1.0), t * 4.0);
  if (t < 0.50) return mix(vec3(0.0, 0.9, 1.0), vec3(0.3, 1.0, 0.0), (t-0.25)*4.0);
  if (t < 0.75) return mix(vec3(0.3, 1.0, 0.0), vec3(1.0, 0.9, 0.0), (t-0.50)*4.0);
  return             mix(vec3(1.0, 0.9, 0.0), vec3(1.0, 0.1, 0.0), (t-0.75)*4.0);
}

// Full HSV rainbow
vec3 rainbow(float t) {
  float h = t * 6.0;
  float x = 1.0 - abs(mod(h, 2.0) - 1.0);
  if (h < 1.0) return vec3(1.0, x,   0.0);
  if (h < 2.0) return vec3(x,   1.0, 0.0);
  if (h < 3.0) return vec3(0.0, 1.0, x  );
  if (h < 4.0) return vec3(0.0, x,   1.0);
  if (h < 5.0) return vec3(x,   0.0, 1.0);
  return              vec3(1.0, 0.0, x  );
}

// Cyan → green → lime
vec3 greenPalette(float t) {
  return mix(vec3(0.0, 0.7, 1.0), vec3(0.6, 1.0, 0.1), t);
}

void main() {
  float th1 = a_state.x;
  float th2 = a_state.z;

  // Second-bob position (l1=l2=1), NDC range [-2.2, 2.2]
  float x2 = sin(th1) + sin(th2);
  float y2 = -(cos(th1) + cos(th2));

  gl_Position  = vec4(x2 / 2.2, y2 / 2.2, 0.0, 1.0);
  gl_PointSize = u_pointSize;

  float t = float(gl_VertexID) / float(u_nParticles - 1);
  vec3 col;
  if      (u_colorMode == 1) col = rainbow(t);
  else if (u_colorMode == 2) col = greenPalette(t);
  else                       col = heat(t);

  // Pre-multiplied alpha for additive blending
  float alpha = 0.55;
  v_color = vec4(col * alpha, alpha);
}
`;

const FRAG_SCATTER = /* glsl */`#version 300 es
precision mediump float;

in  vec4 v_color;
out vec4 fragColor;

void main() {
  // Soft circular point (discard corners)
  vec2  c  = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(c, c);
  if (r2 > 1.0) discard;
  float softness = pow(1.0 - r2, 1.5);
  fragColor = v_color * softness;
}
`;

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Fill state buffer: linear spread of θ₁ across [theta1-spread/2, theta1+spread/2]. */
function generateInitialState(p: DoublePendulumParams): Float32Array {
  const data = new Float32Array(N_PENDULUMS * 4);
  for (let i = 0; i < N_PENDULUMS; i++) {
    const t = i / (N_PENDULUMS - 1);
    data[i * 4 + 0] = p.theta1 + (t - 0.5) * p.spread;
    data[i * 4 + 1] = p.omega1;
    data[i * 4 + 2] = p.theta2;
    data[i * 4 + 3] = p.omega2;
  }
  return data;
}

// ─── Public types ─────────────────────────────────────────────────────────────

export type ColorMode = 'heat' | 'rainbow' | 'green';

export interface DoublePendulumParams {
  theta1: number;  // rad
  omega1: number;  // rad/s
  theta2: number;  // rad
  omega2: number;  // rad/s
  g: number;       // m/s²
  dt: number;
  spread: number;  // rad – initial θ₁ spread across ensemble
  pointSize: number;
  colorMode: ColorMode;
}

// ─── GPU class ────────────────────────────────────────────────────────────────

export class DoublePendulumGPU {
  readonly canvas: HTMLCanvasElement;
  private gl: WebGL2RenderingContext;
  private tf: WebGLTransformFeedback;

  private progIntegrate: WebGLProgram;
  private progScatter:   WebGLProgram;

  private bufStateA: WebGLBuffer;
  private bufStateB: WebGLBuffer;

  /** VAOs for the integration pass (reads state, outputs via TF) */
  private vaoIntA: WebGLVertexArrayObject;
  private vaoIntB: WebGLVertexArrayObject;

  /** VAOs for the scatter render pass (reads state, rasterises positions) */
  private vaoScatterA: WebGLVertexArrayObject;
  private vaoScatterB: WebGLVertexArrayObject;

  // Integration uniforms
  private uDt:    WebGLUniformLocation;
  private uG:     WebGLUniformLocation;
  private uSteps: WebGLUniformLocation;

  // Scatter uniforms
  private uPointSize:  WebGLUniformLocation;
  private uNParticles: WebGLUniformLocation;
  private uColorMode:  WebGLUniformLocation;

  /** Ping-pong: false → A is current, true → B is current */
  private ping = false;
  totalFrames = 0;

  constructor(params: DoublePendulumParams) {
    const canvas = document.createElement('canvas');
    canvas.width  = 512;
    canvas.height = 512;
    this.canvas = canvas;

    // alpha:true so transparent background passes through to 2D trail canvas.
    const gl = createWebGL2Context(canvas, { alpha: true, antialias: false });
    if (!gl) throw new Error('WebGL2 not available');
    this.gl = gl;

    this.progIntegrate = createProgram(gl, VERT_INTEGRATE, FRAG_NOOP, ['v_newState']);
    this.progScatter   = createProgram(gl, VERT_SCATTER,   FRAG_SCATTER);

    const pi = this.progIntegrate;
    this.uDt    = gl.getUniformLocation(pi, 'u_dt')!;
    this.uG     = gl.getUniformLocation(pi, 'u_g')!;
    this.uSteps = gl.getUniformLocation(pi, 'u_steps')!;

    const ps = this.progScatter;
    this.uPointSize  = gl.getUniformLocation(ps, 'u_pointSize')!;
    this.uNParticles = gl.getUniformLocation(ps, 'u_nParticles')!;
    this.uColorMode  = gl.getUniformLocation(ps, 'u_colorMode')!;

    const byteSize = N_PENDULUMS * 4 * 4;
    this.bufStateA = this.makeBuffer(generateInitialState(params), null);
    this.bufStateB = this.makeBuffer(null, byteSize);

    this.tf = gl.createTransformFeedback()!;

    const aInt     = gl.getAttribLocation(pi, 'a_state');
    this.vaoIntA   = this.makeVAO(this.bufStateA, aInt);
    this.vaoIntB   = this.makeVAO(this.bufStateB, aInt);

    const aSct         = gl.getAttribLocation(ps, 'a_state');
    this.vaoScatterA   = this.makeVAO(this.bufStateA, aSct);
    this.vaoScatterB   = this.makeVAO(this.bufStateB, aSct);
  }

  private makeBuffer(data: Float32Array | null, size: number | null): WebGLBuffer {
    const gl = this.gl;
    const buf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    if (data) gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_COPY);
    else      gl.bufferData(gl.ARRAY_BUFFER, size!, gl.DYNAMIC_COPY);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return buf;
  }

  private makeVAO(buf: WebGLBuffer, attribLoc: number): WebGLVertexArrayObject {
    const gl = this.gl;
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(attribLoc);
    gl.vertexAttribPointer(attribLoc, 4, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    return vao;
  }

  /** Run STEPS_PER_FRAME integration steps for all pendulums via transform feedback. */
  step(p: DoublePendulumParams): void {
    const gl = this.gl;

    // ping=false: read A, write B → flip to true
    // ping=true : read B, write A → flip to false
    const srcVAO = this.ping ? this.vaoIntB  : this.vaoIntA;
    const dstBuf = this.ping ? this.bufStateA : this.bufStateB;

    gl.useProgram(this.progIntegrate);
    gl.uniform1f(this.uDt,    p.dt);
    gl.uniform1f(this.uG,     p.g);
    gl.uniform1i(this.uSteps, STEPS_PER_FRAME);

    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.tf);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, dstBuf);

    gl.enable(gl.RASTERIZER_DISCARD);
    gl.bindVertexArray(srcVAO);
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS, 0, N_PENDULUMS);
    gl.endTransformFeedback();
    gl.disable(gl.RASTERIZER_DISCARD);

    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
    gl.bindVertexArray(null);

    this.ping = !this.ping;
    this.totalFrames++;
  }

  /**
   * Render current second-bob positions as additive colored scatter to this.canvas.
   * Background is cleared to transparent so 2D drawImage can accumulate trails.
   */
  renderScatter(p: DoublePendulumParams): void {
    const gl = this.gl;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.enable(gl.BLEND);
    gl.blendEquation(gl.FUNC_ADD);
    gl.blendFunc(gl.ONE, gl.ONE); // additive

    gl.useProgram(this.progScatter);
    gl.uniform1f(this.uPointSize, p.pointSize);
    gl.uniform1i(this.uNParticles, N_PENDULUMS);
    gl.uniform1i(this.uColorMode,
      p.colorMode === 'rainbow' ? 1 : p.colorMode === 'green' ? 2 : 0);

    // After step(), ping was toggled. Current state lives in the buffer
    // written during the last step:
    //   ping=true  → last write was to bufStateB → use vaoScatterB
    //   ping=false → last write was to bufStateA → use vaoScatterA
    const vao = this.ping ? this.vaoScatterB : this.vaoScatterA;
    gl.bindVertexArray(vao);
    gl.drawArrays(gl.POINTS, 0, N_PENDULUMS);
    gl.bindVertexArray(null);

    gl.disable(gl.BLEND);
  }

  /** Re-upload initial state and reset counters. */
  reset(params: DoublePendulumParams): void {
    const gl = this.gl;
    const data = generateInitialState(params);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufStateA);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    this.ping = false;
    this.totalFrames = 0;
  }

  dispose(): void {
    const gl = this.gl;
    gl.deleteProgram(this.progIntegrate);
    gl.deleteProgram(this.progScatter);
    for (const b of [this.bufStateA, this.bufStateB])             gl.deleteBuffer(b);
    for (const v of [this.vaoIntA, this.vaoIntB, this.vaoScatterA, this.vaoScatterB]) gl.deleteVertexArray(v);
    gl.deleteTransformFeedback(this.tf);
  }
}
