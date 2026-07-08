import type { InspectionResponse, ModelGraph, PredictionResponse, RunSummary } from "./types";

export async function getHealth(): Promise<{ status: string; product: string }> {
  const response = await fetch("/health");
  if (!response.ok) throw new Error("Backend health check failed");
  return response.json();
}

export async function getDemoModel(): Promise<ModelGraph> {
  const response = await fetch("/api/demo/model");
  if (!response.ok) throw new Error("Failed to load demo model");
  return response.json();
}

export async function getDemoForward(index: number): Promise<PredictionResponse> {
  const response = await fetch(`/api/demo/forward?index=${index}`);
  if (!response.ok) throw new Error("Failed to run trusted demo forward");
  return response.json();
}

export async function listRuns(): Promise<RunSummary[]> {
  const response = await fetch("/api/runs");
  if (!response.ok) throw new Error("Failed to list runs");
  return response.json();
}

export async function inspectFile(file: File): Promise<InspectionResponse> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch("/api/inspect/upload", { method: "POST", body: form });
  if (!response.ok) {
    const detail = await response.json().then((body) => body?.detail).catch(() => undefined);
    throw new Error(typeof detail === "string" ? detail : `Inspection failed (HTTP ${response.status})`);
  }
  return response.json();
}
