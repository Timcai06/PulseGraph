import { useEffect, useRef } from "react";
import VERT from "./shaders/Dither.vert.glsl?raw";
import FRAG from "./shaders/Dither.frag.glsl?raw";

type Props = {
  /** Quantization levels — lower = chunkier dither banding. Default 5. */
  colorSteps?: number;
  /** Noise field zoom. Default 3.2. */
  scale?: number;
  /** Drift speed. Default 0.5. */
  speed?: number;
  /** Retro dither block size in device px — larger = chunkier pixel dots. Default 3. */
  pixelSize?: number;
  className?: string;
};

// Two-tone deep-navy↔blue palette — PulseGraph cockpit tokens (--bg family,
// accent hue)。high 用 #1a3fae（比 --accent 暗两档）：场保持黑占主导，
// 白字/角落 mono 标签压上去才有对比度，亮蓝只在抖动亮带浮出。
const COLOR_LOW: [number, number, number] = [0.008, 0.016, 0.039]; // #02040a near-black navy
const COLOR_HIGH: [number, number, number] = [0.102, 0.247, 0.682]; // #1a3fae deep accent blue

/**
 * Dither 开屏背景 —— 单 pass 的「5-octave FBM 波动噪声 + 8×8 Bayer 有序抖动」着色器。
 * 只作为 intro 面板的动态背景层，面板退场后随 Loader 卸载，不进站点稳态 GPU 预算。
 *
 * 原生 WebGL1 实现：这里只需要一个全屏三角形 + 一段 fragment shader，
 * three.js 在此纯属样板（曾用 RawShaderMaterial 等价实现），去掉后省下
 * 整个 three chunk 的下载。shader 源文件保持与 TTTIM 原版逐字一致。
 *
 * perf: DPR 上限 1.5；visibilitychange 后台暂停；挂载 1s uFade 淡入；
 *   卸载完整清理 + loseContext。降动用户直接早退（不挂 canvas，
 *   露出 intro 的静态渐变兜底）。
 */
export default function DitherBackground({
  colorSteps = 5,
  scale = 3.2,
  speed = 0.5,
  pixelSize = 3,
  className = ""
}: Props) {
  const mountRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl", {
      antialias: false,
      alpha: false,
      depth: false,
      stencil: false,
      powerPreference: "high-performance",
      premultipliedAlpha: false,
      preserveDrawingBuffer: false
    });
    if (!gl) return; // WebGL 不可用：静态渐变兜底

    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.style.display = "block";
    mount.appendChild(canvas);

    const compile = (type: number, source: string) => {
      const shader = gl.createShader(type);
      if (!shader) return null;
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        gl.deleteShader(shader);
        return null;
      }
      return shader;
    };
    const vert = compile(gl.VERTEX_SHADER, VERT);
    const frag = compile(gl.FRAGMENT_SHADER, FRAG);
    const program = gl.createProgram();
    if (!vert || !frag || !program) {
      mount.removeChild(canvas);
      return;
    }
    gl.attachShader(program, vert);
    gl.attachShader(program, frag);
    gl.linkProgram(program);
    gl.useProgram(program);

    // 全屏大三角：覆盖裁剪空间，比两个三角少一条对角线插值缝
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]),
      gl.STATIC_DRAW
    );
    const positionLoc = gl.getAttribLocation(program, "position");
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 3, gl.FLOAT, false, 0, 0);

    const uniform = (name: string) => gl.getUniformLocation(program, name);
    const uTime = uniform("iTime");
    const uResolution = uniform("iResolution");
    const uFade = uniform("uFade");
    gl.uniform3f(uniform("uColorLow"), ...COLOR_LOW);
    gl.uniform3f(uniform("uColorHigh"), ...COLOR_HIGH);
    gl.uniform1f(uniform("uColorSteps"), colorSteps);
    gl.uniform1f(uniform("uScale"), scale);
    gl.uniform1f(uniform("uSpeed"), speed);
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    gl.uniform1f(uniform("uPixelSize"), Math.max(1, pixelSize * dpr));

    const setSize = () => {
      const w = mount.clientWidth || 1;
      const h = mount.clientHeight || 1;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.uniform2f(uResolution, canvas.width, canvas.height);
    };
    setSize();
    const ro = new ResizeObserver(setSize);
    ro.observe(mount);

    let paused = false;
    const onVis = () => {
      paused = document.hidden;
    };
    document.addEventListener("visibilitychange", onVis, { passive: true });

    let raf = 0;
    let fade = 0;
    const start = performance.now();
    const animate = (now: number) => {
      raf = requestAnimationFrame(animate);
      if (paused) return;
      gl.uniform1f(uTime, (now - start) / 1000);
      if (fade < 1) {
        fade = Math.min(1, fade + 1 / 60); // ~1s ease-in over frames
        gl.uniform1f(uFade, fade);
      }
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    };
    raf = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      document.removeEventListener("visibilitychange", onVis);
      gl.deleteBuffer(buffer);
      gl.deleteProgram(program);
      gl.deleteShader(vert);
      gl.deleteShader(frag);
      gl.getExtension("WEBGL_lose_context")?.loseContext();
      if (mount.contains(canvas)) mount.removeChild(canvas);
    };
  }, [colorSteps, scale, speed, pixelSize]);

  return <div ref={mountRef} className={`intro__dither ${className}`.trim()} aria-hidden="true" />;
}
