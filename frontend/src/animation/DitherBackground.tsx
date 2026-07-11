import { useEffect, useRef } from "react";
import * as THREE from "three";
import VERT from "./shaders/Dither.vert.glsl?raw";
import FRAG from "./shaders/Dither.frag.glsl?raw";

interface DitherUniforms {
  iTime: THREE.IUniform<number>;
  iResolution: THREE.IUniform<THREE.Vector2>;
  uColorLow: THREE.IUniform<THREE.Vector3>;
  uColorHigh: THREE.IUniform<THREE.Vector3>;
  uColorSteps: THREE.IUniform<number>;
  uScale: THREE.IUniform<number>;
  uSpeed: THREE.IUniform<number>;
  uPixelSize: THREE.IUniform<number>;
  uFade: THREE.IUniform<number>;
  [uniform: string]: THREE.IUniform;
}

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
const COLOR_LOW = new THREE.Vector3(0.008, 0.016, 0.039); // #02040a near-black navy
const COLOR_HIGH = new THREE.Vector3(0.102, 0.247, 0.682); // #1a3fae deep accent blue

/**
 * Dither 开屏背景 —— 单 pass 的「5-octave FBM 波动噪声 + 8×8 Bayer 有序抖动」着色器。
 * 只作为 intro 面板的动态背景层，面板退场后随 Loader 卸载，不进站点稳态 GPU 预算。
 *
 * perf: DPR 上限 1.5；visibilitychange 后台暂停；挂载 1s uFade 淡入；
 *   卸载完整清理 + forceContextLoss。降动用户直接早退（不挂 canvas，
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

    const renderer = new THREE.WebGLRenderer({
      antialias: false,
      alpha: false,
      depth: false,
      stencil: false,
      powerPreference: "high-performance",
      premultipliedAlpha: false,
      preserveDrawingBuffer: false
    });
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    renderer.setPixelRatio(dpr);
    renderer.setClearColor(0x000000, 1);
    const canvas = renderer.domElement;
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.style.display = "block";
    mount.appendChild(canvas);

    const scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    // 全屏大三角：覆盖裁剪空间，比两个三角少一条对角线插值缝
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3)
    );

    const uniforms: DitherUniforms = {
      iTime: { value: 0 },
      iResolution: { value: new THREE.Vector2(1, 1) },
      uColorLow: { value: COLOR_LOW.clone() },
      uColorHigh: { value: COLOR_HIGH.clone() },
      uColorSteps: { value: colorSteps },
      uScale: { value: scale },
      uSpeed: { value: speed },
      uPixelSize: { value: Math.max(1, pixelSize * dpr) },
      uFade: { value: 0 }
    };

    const material = new THREE.RawShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms,
      depthTest: false,
      depthWrite: false
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false;
    scene.add(mesh);

    const setSize = () => {
      const w = mount.clientWidth || 1;
      const h = mount.clientHeight || 1;
      renderer.setSize(w, h, false);
      uniforms.iResolution.value.set(w * dpr, h * dpr);
    };
    setSize();
    const ro = new ResizeObserver(setSize);
    ro.observe(mount);

    let paused = false;
    const onVis = () => {
      paused = document.hidden;
    };
    document.addEventListener("visibilitychange", onVis, { passive: true });

    const clock = new THREE.Clock();
    let raf = 0;
    let fade = 0;
    const animate = () => {
      raf = requestAnimationFrame(animate);
      if (paused) return;
      uniforms.iTime.value = clock.getElapsedTime();
      if (fade < 1) {
        fade = Math.min(1, fade + 1 / 60); // ~1s ease-in over frames
        uniforms.uFade.value = fade;
      }
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      document.removeEventListener("visibilitychange", onVis);
      geometry.dispose();
      material.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      if (mount.contains(canvas)) mount.removeChild(canvas);
    };
  }, [colorSteps, scale, speed, pixelSize]);

  return <div ref={mountRef} className={`intro__dither ${className}`.trim()} aria-hidden="true" />;
}
