/**
 * 开屏动画的会话门与交接订阅 —— Loader 和 App 共享的唯一事实源。
 * 模块加载时一次性判定（先于任何 render/effect，两边拿到同一结果）。
 */

/** Loader 面板清场前一瞬 dispatch；App 侧监听它接管入场动画。 */
export const INTRO_EXIT_EVENT = "pulsegraph:intro-exit";

const SEEN_KEY = "pg:intro-seen";

function computeEnabled(): boolean {
  if (typeof window === "undefined") return false;
  // ?intro 强制重播（调动画用）：http://127.0.0.1:5173/?intro
  if (new URLSearchParams(window.location.search).has("intro")) return true;
  try {
    return window.sessionStorage.getItem(SEEN_KEY) !== "1";
  } catch {
    return true;
  }
}

/** 本次页面加载是否播放开屏 —— 每会话（tab）只播一次。 */
export const introEnabled = computeEnabled();

/** 在退场时标记（而非挂载时）：播到一半刷新会重播，完整看过才算看过。 */
export function markIntroSeen() {
  try {
    window.sessionStorage.setItem(SEEN_KEY, "1");
  } catch {
    /* 隐私模式等 sessionStorage 不可写时静默放过 */
  }
}

/**
 * intro → App 的交接：intro 会播时在面板清场前一瞬回调；
 * 本会话不播时立即回调（App 入场动画照常直接跑）。
 * 返回取消订阅函数（配合 effect cleanup / StrictMode 双挂载）。
 */
export function onIntroHandoff(callback: () => void): () => void {
  if (!introEnabled) {
    callback();
    return () => {};
  }
  const handler = () => callback();
  window.addEventListener(INTRO_EXIT_EVENT, handler, { once: true });
  return () => window.removeEventListener(INTRO_EXIT_EVENT, handler);
}
