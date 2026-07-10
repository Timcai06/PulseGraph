import { useRef } from "react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import type { MetricPoint } from "../hooks/useRunStream";
import type { MetricSchema } from "../api/types";
import { useReducedMotion } from "../hooks/useReducedMotion";
import { motionDuration, motionEase } from "../lib/motion";
import { derivePrimaryMetricSignal, latestMetricValue } from "../lib/metricSeries";

gsap.registerPlugin(useGSAP);

function useCountUp(value: number | undefined, format: (current: number) => string) {
  const ref = useRef<HTMLElement | null>(null);
  const tracked = useRef({ current: 0 });
  const reducedMotion = useReducedMotion();

  useGSAP(() => {
    const el = ref.current;
    if (!el || value === undefined) return;
    if (reducedMotion) {
      tracked.current.current = value;
      el.textContent = format(value);
      return;
    }
    gsap.to(tracked.current, {
      current: value,
      duration: motionDuration("count", reducedMotion),
      ease: motionEase.standard,
      overwrite: true,
      onUpdate: () => {
        el.textContent = format(tracked.current.current);
      }
    });
  }, [value, reducedMotion]);

  return ref;
}

type Props = {
  metrics: MetricPoint[];
  task?: string;
  metricSchema?: MetricSchema | null;
};

export function StageStats({ metrics, task, metricSchema }: Props) {
  const loss = latestMetricValue(metrics, "loss");
  const primarySignal = derivePrimaryMetricSignal(metrics, { task, metricSchema });
  const primaryValue = primarySignal ? latestMetricValue(metrics, primarySignal.key) : undefined;
  const step = metrics.length ? metrics[metrics.length - 1].step : undefined;

  const lossRef = useCountUp(loss, (current) => current.toFixed(4));
  const primaryRef = useCountUp(primaryValue, (current) =>
    primarySignal?.format === "percent" ? `${(current * 100).toFixed(1)}%` : current.toFixed(4)
  );
  const stepRef = useCountUp(step, (current) => String(Math.round(current)));

  if (!metrics.length) return null;

  return (
    <aside className="stage-stats" aria-hidden="true">
      <span className="stat-label">Loss</span>
      <strong className="stat-value" ref={lossRef}>–</strong>
      <div className="stat-row">
        <span>
          {primarySignal?.label ?? "signal"} <b ref={primaryRef}>–</b>
        </span>
        <span>
          step <b ref={stepRef}>–</b>
        </span>
      </div>
    </aside>
  );
}
