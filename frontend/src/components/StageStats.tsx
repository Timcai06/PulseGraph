import { useRef } from "react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import type { MetricPoint } from "../hooks/useRunStream";
import { useReducedMotion } from "../hooks/useReducedMotion";
import { motionDuration, motionEase } from "../lib/motion";

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
};

export function StageStats({ metrics }: Props) {
  let loss: number | undefined;
  let accuracy: number | undefined;
  for (let index = metrics.length - 1; index >= 0; index -= 1) {
    if (loss === undefined && metrics[index].loss != null) loss = metrics[index].loss;
    if (accuracy === undefined && metrics[index].accuracy != null) accuracy = metrics[index].accuracy;
    if (loss !== undefined && accuracy !== undefined) break;
  }
  const step = metrics.length ? metrics[metrics.length - 1].step : undefined;

  const lossRef = useCountUp(loss, (current) => current.toFixed(4));
  const accuracyRef = useCountUp(accuracy, (current) => `${(current * 100).toFixed(1)}%`);
  const stepRef = useCountUp(step, (current) => String(Math.round(current)));

  if (!metrics.length) return null;

  return (
    <aside className="stage-stats" aria-hidden="true">
      <span className="stat-label">Loss</span>
      <strong className="stat-value" ref={lossRef}>–</strong>
      <div className="stat-row">
        <span>
          acc <b ref={accuracyRef}>–</b>
        </span>
        <span>
          step <b ref={stepRef}>–</b>
        </span>
      </div>
    </aside>
  );
}
