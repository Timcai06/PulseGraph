import type { MetricSchema, OutputSchema } from "../api/types";

export type RunContract = {
  task?: string;
  outputSchema?: OutputSchema;
  metricSchema?: MetricSchema;
};

function recordValue(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : undefined;
}

export function runContractFromConfig(config?: Record<string, unknown> | null): RunContract | undefined {
  if (!config) return undefined;
  const task = typeof config.task === "string" && config.task.trim() ? config.task.trim().toLowerCase() : undefined;
  const outputSchema = recordValue(config.output_schema) as OutputSchema | undefined;
  const metricSchema = recordValue(config.metric_schema) as MetricSchema | undefined;
  if (!task && !outputSchema && !metricSchema) return undefined;
  return { task, outputSchema, metricSchema };
}
