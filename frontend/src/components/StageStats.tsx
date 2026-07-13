import { useRef } from "react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import type { MetricPoint } from "../hooks/useRunStream";
import type { MetricSchema } from "../api/types";
import type { RunDetail } from "../api/client";
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
  detail?: RunDetail;
  device?: string;
  metrics: MetricPoint[];
  runId?: string;
  status?: string;
  task?: string;
  metricSchema?: MetricSchema | null;
};

export function StageStats({ detail, device, metrics, runId, status, task, metricSchema }: Props) {
  const loss = latestMetricValue(metrics, "loss");
  const primarySignal = derivePrimaryMetricSignal(metrics, { task, metricSchema });
  const primaryValue = primarySignal ? latestMetricValue(metrics, primarySignal.key) : undefined;
  const step = metrics.length ? metrics[metrics.length - 1].step : undefined;
  const throughput = latestMetricValue(metrics, "samples_per_sec");
  const model = typeof detail?.config?.model === "string" ? detail.config.model : detail?.entry_class ?? "model";
  const datasetSpec = detail?.config?.dataset_spec;
  const datasetRecord = datasetSpec && typeof datasetSpec === "object" && !Array.isArray(datasetSpec)
    ? datasetSpec as Record<string, unknown>
    : undefined;
  const dataset = typeof datasetRecord?.name === "string"
    ? datasetRecord.name
    : task ?? "training";

  const lossRef = useCountUp(loss, (current) => current.toFixed(4));
  const primaryRef = useCountUp(primaryValue, (current) =>
    primarySignal?.format === "percent" ? `${(current * 100).toFixed(1)}%` : current.toFixed(4)
  );
  const stepRef = useCountUp(step, (current) => String(Math.round(current)));

  if (!metrics.length && !runId) return null;

  return (
    <aside className="stage-stats" aria-label="current training run">
      <div className="stage-run-identity">
        <span>{status ?? "idle"}</span>
        <strong title={runId}>{runId ?? "local run"}</strong>
        <em>{model} · {dataset}</em>
      </div>
      <div className="stage-stat-grid">
        <span><small>Loss</small><b ref={lossRef}>–</b></span>
        <span><small>{primarySignal?.label ?? "signal"}</small><b ref={primaryRef}>–</b></span>
        <span><small>Step</small><b ref={stepRef}>–</b></span>
        <span><small>Samples/s</small><b>{throughput == null ? "–" : throughput.toFixed(1)}</b></span>
        <span><small>Device</small><b>{device ?? "unknown"}</b></span>
        <span><small>Checkpoint</small><b>{detail?.checkpoints.length ?? 0}</b></span>
      </div>
    </aside>
  );
}
