/**
 * 开屏动画的会话门与交接订阅 —— Loader 和 App 共享的唯一事实源。
 * 模块加载时一次性判定（先于任何 render/effect，两边拿到同一结果）。
 */

/** Loader 面板清场前一瞬 dispatch；App 侧监听它接管入场动画。 */
export const INTRO_EXIT_EVENT = "pulsegraph:intro-exit";

function computeEnabled(): boolean {
  if (typeof window === "undefined") return false;
  // 每次刷新都播；调试/E2E 想跳过时加 ?nointro：http://127.0.0.1:5173/?nointro
  return !new URLSearchParams(window.location.search).has("nointro");
}

/** 本次页面加载是否播放开屏。 */
export const introEnabled = computeEnabled();

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
