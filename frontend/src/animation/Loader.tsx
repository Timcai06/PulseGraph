import { lazy, Suspense, useEffect, useRef, useState } from "react";
import gsap from "gsap";
import { INTRO_EXIT_EVENT, introEnabled, markIntroSeen } from "./introGate";
import { usePreloadGate } from "./usePreloadGate";
import {
  displayedProgressValue,
  introCharGroup,
  introRiseStagger,
  stepDisplayedProgress
} from "./loaderTiming";

// Lazy so the intro Dither shader (+ three) stays out of the eager index chunk;
// the static `.intro` gradient is the fallback until it streams in.
const DitherBackground = lazy(() => import("./DitherBackground"));

const TITLE = "PulseGraph.";
const BAFFLE_CHARS = "!<>-_\\/[]{}—=+*^?#█▓▒░█";

function randomBaffleChar() {
  return BAFFLE_CHARS[Math.floor(Math.random() * BAFFLE_CHARS.length)] ?? "";
}

/**
 * 会话门：每 tab 只播一次（sessionStorage），`?intro` 强制重播。
 * 不播时整个 Loader 不挂载 —— 预加载请求、GSAP、shader chunk 都不发生；
 * introGate.onIntroHandoff 会立即放行 App 的入场动画。
 */
export default function Loader() {
  if (!introEnabled) return null;
  return <LoaderPanel />;
}

/**
 * Loader 全屏加载页 —— 应用入口动画的 start-to-end 编排。
 *   阶段 1 (intro): "PulseGraph." 字符从遮罩边缘升起 + Baffle 乱码 (42ms × 15 帧) → 扫描线掠过
 *   阶段 2 (hand-off): introReady (字符落地) 且 preload.ready (字体 + 后端 API 全就绪，
 *     进度条 100% = 首屏依赖真的加载完) → 标题上浮出遮罩 → 计数器/进度条淡出 → 整屏上滑揭幕
 *   阶段 3 (complete): done 置 true，组件返回 null，彻底从 DOM 卸载
 *
 * 节奏参数全部来自 loaderTiming.ts（可单测）；乱码用 setInterval(42ms) ≈ 24fps，
 * 抖动感刚好且不需要 60fps 精度；退场 yPercent:-100 用 expo.inOut(1.15s) 制造卷帘门感。
 *
 * prefers-reduced-motion：跳过乱码/升起/卷帘（字符直接就位，退场改 0.35s 淡出），
 * 但资源门照走 —— 降动用户同样等到真实资源就绪。
 */
function LoaderPanel() {
  const panelRef = useRef<HTMLDivElement>(null);
  const textRef = useRef<HTMLDivElement>(null);
  const countRef = useRef<HTMLSpanElement>(null);
  const barRef = useRef<HTMLSpanElement>(null);
  const exitStarted = useRef(false);
  const preloadRef = useRef<ReturnType<typeof usePreloadGate> | null>(null);
  const [done, setDone] = useState(false);
  const [introReady, setIntroReady] = useState(false);
  const preload = usePreloadGate();
  const stageText = preload.ready ? "ready" : preload.label;
  // 组件生命周期内不变，读一次即可（切系统偏好需刷新，与原版一致）
  const reducedMotion = useRef(
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  ).current;

  useEffect(() => {
    preloadRef.current = preload;
  }, [preload]);

  /* ── 入场：乱码 + 字符成组升起 + 扫描线 ── */
  useEffect(() => {
    if (!panelRef.current) return;
    const intervals: number[] = [];

    const ctx = gsap.context(() => {
      const charEls = textRef.current?.querySelectorAll<HTMLElement>(".intro__char");
      if (!charEls?.length) return;

      if (reducedMotion) {
        // 快速路径：字符直接就位，跳过乱码与扫描线
        gsap.set(charEls, { opacity: 1, yPercent: 0 });
        setIntroReady(true);
        return;
      }

      const tl = gsap.timeline();

      gsap.set(charEls, { opacity: 0, yPercent: 120 });
      gsap.set(".intro__scan-line", { opacity: 0, scaleX: 0 });

      charEls.forEach((el) => {
        const glyph = el.querySelector<HTMLElement>(".intro__char-glyph") ?? el;
        const final = glyph.getAttribute("data-final") || glyph.textContent || "";
        let frame = 0;
        const interval = window.setInterval(() => {
          if (frame < 11) {
            glyph.textContent = randomBaffleChar();
          } else if (frame < 15) {
            glyph.textContent = frame % 2 === 0 ? randomBaffleChar() : final;
          } else {
            glyph.textContent = final;
            clearInterval(interval);
          }
          frame++;
        }, 42);
        intervals.push(interval);
      });

      tl.to(
        charEls,
        {
          opacity: 1,
          yPercent: 0,
          duration: 1.25,
          stagger: (index, target) => {
            const group = Number((target as HTMLElement).dataset.introGroup ?? index);
            return introRiseStagger(group, index);
          },
          ease: "expo.out"
        },
        0.1
      );

      tl.to(
        ".intro__scan-line",
        { opacity: 0.9, scaleX: 1, duration: 0.42, ease: "power3.out" },
        ">-0.28"
      );
      tl.to(".intro__scan-line", { opacity: 0, duration: 0.5, ease: "power2.out" }, ">-0.08");

      tl.call(() => setIntroReady(true));
    }, panelRef);

    return () => {
      intervals.forEach((interval) => window.clearInterval(interval));
      ctx.revert();
    };
  }, []);

  /* ── 进度条：rAF 阻尼追踪真实 preload 进度 ── */
  useEffect(() => {
    let frame = 0;
    let displayedProgress = 0;

    const renderProgress = () => {
      const current = preloadRef.current;
      if (!current) return;

      const actualProgress = current.total > 0 ? current.completed / current.total : 0;
      const target = current.ready ? 1 : actualProgress;

      displayedProgress = stepDisplayedProgress(displayedProgress, target, current.ready);
      const displayValue = displayedProgressValue(displayedProgress, current.ready);

      if (countRef.current) countRef.current.textContent = String(displayValue).padStart(2, "0");
      if (barRef.current) barRef.current.style.transform = `scaleX(${displayedProgress.toFixed(4)})`;

      if (!current.ready || displayedProgress < 0.999) {
        frame = window.requestAnimationFrame(renderProgress);
      }
    };

    frame = window.requestAnimationFrame(renderProgress);
    return () => window.cancelAnimationFrame(frame);
  }, []);

  /* ── 退场：introReady && preload.ready 双门同时满足才开闸 ── */
  useEffect(() => {
    if (!introReady || !preload.ready || exitStarted.current || !panelRef.current) return;
    exitStarted.current = true;

    const ctx = gsap.context(() => {
      if (reducedMotion) {
        // 快速路径：交接 + 0.35s 淡出，不跑字符/卷帘编排
        markIntroSeen();
        window.dispatchEvent(new CustomEvent(INTRO_EXIT_EVENT));
        gsap.to(panelRef.current, {
          opacity: 0,
          duration: 0.35,
          ease: "power2.out",
          onComplete: () => setDone(true)
        });
        return;
      }

      const charEls = textRef.current?.querySelectorAll<HTMLElement>(".intro__char");
      if (!charEls?.length) return;
      const titleChars = textRef.current?.querySelectorAll<HTMLElement>(
        ".intro__char:not(.intro__dot)"
      );
      const dot = textRef.current?.querySelector<HTMLElement>(".intro__dot");

      const tl = gsap.timeline();

      /* hold a beat */
      tl.to({}, { duration: 0.35 });

      /* 角落细节（计数器 + 发丝线）退场 */
      tl.to(
        ".intro__counter, .intro__bar-track, .intro__status",
        { opacity: 0, duration: 0.5, ease: "power2.in" },
        ">-0.1"
      );

      /* 标题升出遮罩 */
      tl.to(
        titleChars ?? charEls,
        { yPercent: -120, duration: 0.8, stagger: 0.04, ease: "power3.in" },
        "<"
      );

      if (dot) {
        // GSAP 颜色补间需要具体值 —— 退场时解析设计 token，不硬编码
        const rootStyles = getComputedStyle(document.documentElement);
        const popColor = rootStyles.getPropertyValue("--cyan").trim() || "#38bdf8";
        tl.to(
          dot,
          {
            color: popColor,
            textShadow: "0 0 28px rgba(56, 189, 248, 0.72)",
            yPercent: -70,
            duration: 0.52,
            ease: "power2.in"
          },
          "<0.18"
        );
        tl.to(dot, { opacity: 0, duration: 0.24, ease: "power2.in" }, ">-0.08");
      }

      /* 面板清场前一瞬把控制权交给底下的 App（此刻起算"完整看过"） */
      tl.call(
        () => {
          markIntroSeen();
          window.dispatchEvent(new CustomEvent(INTRO_EXIT_EVENT));
        },
        [],
        ">-0.15"
      );

      /* 卷帘门：整屏上滑，露出已组装好的驾驶舱 */
      tl.to(panelRef.current, { yPercent: -100, duration: 1.15, ease: "expo.inOut" }, ">-0.05");

      /* 浅色主题下深色幕布掀开会有明暗跳变：wipe 后半程叠加淡出，软化落差 */
      if (document.documentElement.dataset.theme === "light") {
        tl.to(panelRef.current, { opacity: 0, duration: 0.7, ease: "power2.in" }, "<0.3");
      }

      tl.call(() => setDone(true));
    }, panelRef);

    return () => ctx.revert();
  }, [introReady, preload.ready]);

  if (done) return null;

  // Inter 是比例字体：按字形宽度分档给固定槽位，乱码换字整行不抖
  const charClassName = (ch: string) => {
    if (ch === ".") return "intro__char intro__dot";
    if (ch === "l") return "intro__char intro__char--narrow";
    if (ch === "G") return "intro__char intro__char--wide";
    return "intro__char";
  };

  const chars = TITLE.split("").map((ch, i) => (
    <span key={i} className={charClassName(ch)} data-intro-group={introCharGroup(i)}>
      <span className="intro__char-glyph" data-final={ch}>
        {ch}
      </span>
    </span>
  ));

  return (
    <div className="intro" ref={panelRef}>
      <Suspense fallback={null}>
        <DitherBackground />
      </Suspense>
      <div className="intro__meta">// PulseGraph · Training Cockpit</div>

      <div className="intro__text-wrap">
        <div className="intro__text" ref={textRef}>
          {chars}
        </div>
      </div>
      <div className="intro__scan-line" aria-hidden="true" />

      <div className="intro__status" aria-hidden="true">
        <span>Runtime gate</span>
        <span>fonts · api · model</span>
      </div>
      <div className="intro__counter">
        <span ref={countRef}>00</span>
        <span className="intro__counter-sep">/ 100</span>
        <span className="intro__stage">{stageText}</span>
      </div>

      <div className="intro__bar-track">
        <span className="intro__bar" ref={barRef} />
      </div>
    </div>
  );
}
