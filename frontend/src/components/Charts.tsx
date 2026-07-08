import { useEffect, useMemo, useRef, useState } from "react";
import * as echarts from "echarts";

export type MetricPoint = {
  step: number;
  loss?: number;
  accuracy?: number;
  stepTimeMs?: number;
  memoryPeakMb?: number;
};

type MetricsProps = {
  points: MetricPoint[];
};

function useReducedMotion() {
  const [reducedMotion, setReducedMotion] = useState(false);

  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReducedMotion(query.matches);
    const onChange = () => setReducedMotion(query.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);

  return reducedMotion;
}

export function MetricChart({ points }: MetricsProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);
  const reducedMotion = useReducedMotion();

  useEffect(() => {
    if (!ref.current) return;
    const chart = echarts.init(ref.current, "dark");
    chartRef.current = chart;
    const onResize = () => chart.resize();
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    chart.setOption({
      backgroundColor: "transparent",
      tooltip: { trigger: "axis" },
      legend: { textStyle: { color: "#9ca3af" } },
      grid: { left: 42, right: 18, top: 36, bottom: 36 },
      xAxis: { type: "category", data: points.map((point) => point.step), axisLabel: { color: "#9ca3af" } },
      yAxis: [
        { type: "value", axisLabel: { color: "#9ca3af" }, splitLine: { lineStyle: { color: "#1f2937" } } },
        { type: "value", axisLabel: { color: "#9ca3af" }, splitLine: { show: false } }
      ],
      series: [
        { name: "loss", type: "line", smooth: true, data: points.map((point) => point.loss ?? null), color: "#f59e0b" },
        { name: "accuracy", type: "line", smooth: true, yAxisIndex: 1, data: points.map((point) => point.accuracy ?? null), color: "#22c55e" },
        { name: "step ms", type: "line", smooth: true, data: points.map((point) => point.stepTimeMs ?? null), color: "#38bdf8" }
      ],
      animationDurationUpdate: reducedMotion ? 0 : 220
    });
  }, [points, reducedMotion]);

  return <div className="chart" ref={ref} />;
}

export function ProbabilityChart({ probabilities }: { probabilities: number[] }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);
  const reducedMotion = useReducedMotion();
  const labels = useMemo(() => probabilities.map((_, index) => index), [probabilities]);

  useEffect(() => {
    if (!ref.current) return;
    const chart = echarts.init(ref.current, "dark");
    chartRef.current = chart;
    const onResize = () => chart.resize();
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    chart.setOption({
      backgroundColor: "transparent",
      grid: { left: 32, right: 12, top: 20, bottom: 30 },
      xAxis: { type: "category", data: labels, axisLabel: { color: "#9ca3af" } },
      yAxis: { type: "value", max: 1, axisLabel: { color: "#9ca3af" }, splitLine: { lineStyle: { color: "#1f2937" } } },
      series: [{ type: "bar", data: probabilities, itemStyle: { color: "#8b5cf6" } }],
      animationDurationUpdate: reducedMotion ? 0 : 240
    });
  }, [labels, probabilities, reducedMotion]);

  return <div className="prob-chart" ref={ref} />;
}
