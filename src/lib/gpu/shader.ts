/** Compile a vertex or fragment shader; throws on error. */
export function compileShader(
  gl: WebGL2RenderingContext,
  type: number,
  src: string,
): WebGLShader {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error(`Shader compile error:\n${log}`);
  }
  return sh;
}

/**
 * Compile + link a WebGL2 program.
 * Pass transformFeedbackVaryings to create a transform-feedback program.
 * The mode is SEPARATE_ATTRIBS (each varying goes to its own buffer slot).
 */
export function createProgram(
  gl: WebGL2RenderingContext,
  vertSrc: string,
  fragSrc: string,
  transformFeedbackVaryings?: readonly string[],
): WebGLProgram {
  const vert = compileShader(gl, gl.VERTEX_SHADER, vertSrc);
  const frag = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
  const prog = gl.createProgram()!;
  gl.attachShader(prog, vert);
  gl.attachShader(prog, frag);
  if (transformFeedbackVaryings?.length) {
    gl.transformFeedbackVaryings(prog, [...transformFeedbackVaryings], gl.SEPARATE_ATTRIBS);
  }
  gl.linkProgram(prog);
  gl.deleteShader(vert);
  gl.deleteShader(frag);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog);
    gl.deleteProgram(prog);
    throw new Error(`Program link error:\n${log}`);
  }
  return prog;
}
