import { useEffect, useState } from "react";
import { getDemoModel, getHealth, listRuns } from "../api/client";

export type PreloadState = {
  completed: number;
  total: number;
  ready: boolean;
  /** 最近完成的任务名，intro 角落的 stage 文案。 */
  label: string;
};

/** 单任务超时：挂掉的资源（如后端未启动且连接悬挂）不能把 intro 困住。 */
const TASK_TIMEOUT_MS = 12_000;

/** 进度条的最短陪跑时间：即使全部资源秒回，也留出字符落地的呼吸感。 */
const MIN_HOLD_MS = 900;

type PreloadTask = { label: string; run: () => Promise<unknown> };

/**
 * Loader 的资源门 —— 原版 useWholeSitePreload 的 PulseGraph 版。
 * 进度条 100% = 驾驶舱首屏依赖的真实资源全部就绪：字体、后端心跳、
 * run 列表、demo 模型图。请求失败也计完成（fast-fail：后端没起时
 * intro 照常走完，App 自己会展示连接错误态）。
 */
export function usePreloadGate(): PreloadState {
  const [state, setState] = useState<PreloadState>({
    completed: 0,
    total: 1,
    ready: false,
    label: "boot"
  });

  useEffect(() => {
    let alive = true;

    const tasks: PreloadTask[] = [
      { label: "fonts", run: () => document.fonts.ready },
      { label: "api · health", run: () => getHealth() },
      { label: "api · runs", run: () => listRuns() },
      { label: "model graph", run: () => getDemoModel() },
      { label: "compose", run: () => new Promise((r) => setTimeout(r, MIN_HOLD_MS)) }
    ];

    let completed = 0;
    setState({ completed: 0, total: tasks.length, ready: false, label: "boot" });

    for (const task of tasks) {
      const timeout = new Promise((r) => setTimeout(r, TASK_TIMEOUT_MS));
      Promise.race([task.run().catch(() => undefined), timeout]).then(() => {
        if (!alive) return;
        completed += 1;
        setState({
          completed,
          total: tasks.length,
          ready: completed >= tasks.length,
          label: task.label
        });
      });
    }

    return () => {
      alive = false;
    };
  }, []);

  return state;
}
