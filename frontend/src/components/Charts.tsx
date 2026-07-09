import { useEffect, useMemo, useRef } from "react";
import * as echarts from "echarts";
import { chartPalette, withAlpha, type Theme } from "../lib/chartTheme";
import { chartProbabilityRows } from "../lib/inferenceView";
import { useReducedMotion } from "../hooks/useReducedMotion";
import type { MetricPoint, StreamStatus } from "../hooks/useRunStream";

export type { MetricPoint };

function useChart(ref: React.RefObject<HTMLDivElement | null>) {
  const chartRef = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    const chart = echarts.init(ref.current);
    chartRef.current = chart;
    const onResize = () => chart.resize();
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chart.dispose();
      chartRef.current = null;
    };
  }, [ref]);

  return chartRef;
}

type MetricsProps = {
  points: MetricPoint[];
  status?: StreamStatus;
  theme?: Theme;
  runKind?: string;
  selectedStep?: number;
};

function emptyMetricMessage(status: StreamStatus, runKind?: string) {
  if (runKind === "source-import") {
    return "This run is an inference replay. Use Run Training to create loss, accuracy, and infra telemetry.";
  }
  if (status === "streaming") {
    return "Waiting for training metrics from the current run.";
  }
  return "Run training or watch a recorded training run to populate telemetry.";
}

export function MetricChart({ points, status = "idle", theme = "dark", runKind, selectedStep }: MetricsProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const chartRef = useChart(ref);
  const reducedMotion = useReducedMotion();

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    const palette = chartPalette();
    const area = (color: string) => ({
      color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
        { offset: 0, color: withAlpha(color, 0.22) },
        { offset: 1, color: withAlpha(color, 0) }
      ])
    });
    chart.setOption(
      {
        backgroundColor: "transparent",
        textStyle: { color: palette.text },
        tooltip: {
          trigger: "axis",
          backgroundColor: palette.tooltipBg,
          borderColor: palette.grid,
          textStyle: { color: palette.text }
        },
        legend: {
          top: 2,
          left: 8,
          itemWidth: 10,
          itemHeight: 6,
          textStyle: { color: palette.text, fontSize: 11 }
        },
        grid: { left: 42, right: 24, top: 52, bottom: 44, containLabel: true },
        xAxis: {
          type: "category",
          data: points.map((point) => point.step),
          name: "step",
          nameLocation: "middle",
          nameGap: 28,
          axisLabel: { color: palette.text, hideOverlap: true, margin: 10 },
          axisLine: { lineStyle: { color: palette.grid } }
        },
        yAxis: [
          { type: "value", axisLabel: { color: palette.text }, splitLine: { lineStyle: { color: palette.grid } } },
          { type: "value", axisLabel: { color: palette.text }, splitLine: { show: false } }
        ],
        series: [
          {
            name: "loss",
            type: "line",
            smooth: true,
            showSymbol: false,
            data: points.map((point) => point.loss ?? null),
            color: palette.amber,
            areaStyle: area(palette.amber)
          },
          {
            name: "accuracy",
            type: "line",
            smooth: true,
            showSymbol: false,
            yAxisIndex: 1,
            data: points.map((point) => point.accuracy ?? null),
            color: palette.green,
            areaStyle: area(palette.green)
          },
          {
            name: "step ms",
            type: "line",
            smooth: true,
            showSymbol: false,
            data: points.map((point) => point.stepTimeMs ?? null),
            color: palette.cyan
          },
          {
            name: "selected step",
            type: "line",
            data: [],
            markLine: selectedStep == null ? undefined : {
              symbol: "none",
              label: { formatter: `step ${selectedStep}`, color: palette.text },
              lineStyle: { color: palette.red, width: 1.5, type: "dashed" },
              data: [{ xAxis: selectedStep }]
            }
          }
        ],
        animationDurationUpdate: reducedMotion ? 0 : 220
      },
      { notMerge: true }
    );
  }, [chartRef, points, reducedMotion, selectedStep, theme]);

  return (
    <div className="chart-wrap">
      <div className="chart" ref={ref} />
      {points.length === 0 && (
        <div className="chart-empty">
          {emptyMetricMessage(status, runKind)}
        </div>
      )}
    </div>
  );
}

type ProbabilityProps = {
  probabilities: number[];
  label?: number;
  prediction?: number;
  classNames?: string[] | null;
  theme?: Theme;
};

export function ProbabilityChart({ probabilities, label, prediction, classNames, theme = "dark" }: ProbabilityProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const chartRef = useChart(ref);
  const reducedMotion = useReducedMotion();
  const rows = useMemo(() => chartProbabilityRows(probabilities, classNames), [classNames, probabilities]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    const palette = chartPalette();
    const colorFor = (index: number) => {
      if (prediction !== undefined && index === prediction) {
        return prediction === label ? palette.green : palette.red;
      }
      if (label !== undefined && index === label) return palette.amber;
      return withAlpha(palette.violet, 0.7);
    };
    chart.setOption(
      {
        backgroundColor: "transparent",
        grid: { left: 32, right: 12, top: 20, bottom: 30 },
        xAxis: {
          type: "category",
          data: rows.map((row) => row.label),
          axisLabel: { color: palette.text, hideOverlap: true },
          axisLine: { lineStyle: { color: palette.grid } }
        },
        yAxis: {
          type: "value",
          max: 1,
          axisLabel: { color: palette.text },
          splitLine: { lineStyle: { color: palette.grid } }
        },
        series: [
          {
            type: "bar",
            data: rows.map((row) => ({ value: row.value, itemStyle: { color: colorFor(row.index) } })),
            barCategoryGap: "28%"
          }
        ],
        animationDuration: reducedMotion ? 0 : 300,
        animationDurationUpdate: reducedMotion ? 0 : 240
      },
      { notMerge: true }
    );
  }, [chartRef, rows, label, prediction, reducedMotion, theme]);

  return <div className="prob-chart" ref={ref} />;
}
