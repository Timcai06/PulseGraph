import type {
  AttachSourceResult,
  ImportArtifactResult,
  InspectionResponse,
  ModelGraph,
  PredictionResponse,
  RunDetail,
  RunReport,
  RunSummary,
  SourceCandidate,
  SourceImportResult
} from "./types";

export type NamedSourceFile = { file: File; path: string };

function sourceForm(files: NamedSourceFile[]): FormData {
  const form = new FormData();
  for (const { file, path } of files) form.append("files", file, path);
  return form;
}

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

export async function deleteRun(runId: string): Promise<{ run_id: string; deleted: boolean }> {
  const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`, { method: "DELETE" });
  if (!response.ok) {
    const detail = await response.json().then((body) => body?.detail).catch(() => undefined);
    throw new Error(typeof detail === "string" ? detail : "Deleting the run failed");
  }
  return response.json();
}

export async function getRunDetail(runId: string): Promise<RunDetail> {
  const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/detail`);
  if (!response.ok) throw new Error("Failed to load run detail");
  return response.json();
}

export async function getRunReport(runId: string): Promise<RunReport> {
  const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/report`);
  if (!response.ok) throw new Error("Failed to build run report");
  return response.json();
}

export async function runForward(runId: string, checkpointStep = 0, index = 0): Promise<PredictionResponse> {
  const response = await fetch(
    `/api/runs/${encodeURIComponent(runId)}/forward?checkpoint_step=${checkpointStep}&index=${index}`
  );
  if (!response.ok) {
    const detail = await response.json().then((body) => body?.detail).catch(() => undefined);
    throw new Error(typeof detail === "string" ? detail : "Forward replay failed");
  }
  return response.json();
}

export async function analyzeSourceCandidates(
  files: NamedSourceFile[]
): Promise<{ files: string[]; candidates: SourceCandidate[] }> {
  const response = await fetch("/api/inspect/source/candidates", { method: "POST", body: sourceForm(files) });
  if (!response.ok) throw new Error("Could not analyze the uploaded source files");
  return response.json();
}

export async function attachRunSource(
  runId: string,
  files: NamedSourceFile[],
  entryFile: string,
  entryClass: string
): Promise<AttachSourceResult> {
  const form = sourceForm(files);
  form.append("entry_file", entryFile);
  form.append("entry_class", entryClass);
  const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/source`, { method: "POST", body: form });
  if (!response.ok) {
    const detail = await response.json().then((body) => body?.detail).catch(() => undefined);
    throw new Error(typeof detail === "string" ? detail : "Attaching source failed");
  }
  return response.json();
}

export async function importArtifact(file: File): Promise<ImportArtifactResult> {
  const response = await fetch("/api/runs/import", { method: "POST", body: file });
  if (!response.ok) {
    const detail = await response.json().then((body) => body?.detail).catch(() => undefined);
    throw new Error(typeof detail === "string" ? detail : "Importing the artifact failed");
  }
  return response.json();
}

export async function importSourceRun(
  files: NamedSourceFile[],
  entryFile: string,
  entryClass: string
): Promise<SourceImportResult> {
  const form = sourceForm(files);
  form.append("entry_file", entryFile);
  form.append("entry_class", entryClass);
  const response = await fetch("/api/runs/from-source", { method: "POST", body: form });
  if (!response.ok) {
    const detail = await response.json().then((body) => body?.detail).catch(() => undefined);
    throw new Error(typeof detail === "string" ? detail : "Creating a run from source failed");
  }
  return response.json();
}

export async function trainSourceRun(
  files: NamedSourceFile[],
  entryFile: string,
  entryClass: string,
  steps = 8
): Promise<SourceImportResult> {
  const form = sourceForm(files);
  form.append("entry_file", entryFile);
  form.append("entry_class", entryClass);
  form.append("steps", String(steps));
  const response = await fetch("/api/runs/train-source", { method: "POST", body: form });
  if (!response.ok) {
    const detail = await response.json().then((body) => body?.detail).catch(() => undefined);
    throw new Error(typeof detail === "string" ? detail : "Training the source failed");
  }
  return response.json();
}

export async function trainResourceRun(
  files: NamedSourceFile[],
  entryFile: string,
  steps = 100,
  telemetryStride = 5
): Promise<SourceImportResult> {
  const form = sourceForm(files);
  form.append("entry_file", entryFile);
  form.append("steps", String(steps));
  form.append("telemetry_stride", String(telemetryStride));
  const response = await fetch("/api/runs/train-resource", { method: "POST", body: form });
  if (!response.ok) {
    const detail = await response.json().then((body) => body?.detail).catch(() => undefined);
    throw new Error(typeof detail === "string" ? detail : "Training the resource failed");
  }
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
