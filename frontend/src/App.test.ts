import { describe, expect, it } from "vitest";
import appSource from "./App.tsx?raw";
import controlRailSource from "./components/ControlRail.tsx?raw";
import digitPreviewSource from "./components/DigitPreview.tsx?raw";
import inferenceProbeSource from "./components/InferenceProbe.tsx?raw";
import modelGraphSource from "./components/ModelGraphPanel.tsx?raw";
import chartsSource from "./components/Charts.tsx?raw";

describe("App workspace layout", () => {
  it("does not render the right-side layer inspector panel", () => {
    expect(appSource).not.toContain("<Layer" + "Inspector");
  });

  it("keeps the monitor free of explanatory demo copy", () => {
    const combined = [appSource, controlRailSource, inferenceProbeSource, digitPreviewSource, modelGraphSource].join("\n");

    expect(combined).not.toContain("Netron-like structure with training pulses");
    expect(combined).not.toContain("Upload source, then run forward or train a short local recipe.");
    expect(combined).not.toContain("Secondary path: inspect weights");
    expect(combined).not.toContain("No active training runs. Completed runs are in History.");
    expect(combined).not.toContain("recorded probe");
    expect(combined).not.toContain("tensor");
  });

  it("waits for source training completion before running forward", () => {
    expect(appSource).toContain("pendingForwardRun");
    expect(appSource).not.toContain("applyPredictionResult(await runForward(result.run_id, result.checkpoint.step));");
  });

  it("uses a single training resource workflow instead of source and pt side paths", () => {
    const combined = [appSource, controlRailSource].join("\n");

    expect(combined).toContain("Training Resource");
    expect(combined).toContain("Run Training");
    expect(combined).toContain("Run Inference");
    expect(combined).not.toContain("Weights File");
    expect(combined).not.toContain("Inspect .pt");
    expect(combined).not.toContain("Train source");
    expect(appSource).not.toContain("importArtifact");
    expect(appSource).not.toContain("inspectFile");
    expect(appSource).not.toContain("importSourceRun");
  });

  it("exposes deletion from history", () => {
    expect(appSource).toContain("handleDeleteRun");
    expect(controlRailSource).not.toContain("Weights File");
  });

  it("labels inference by data source instead of pretending every probe is recognition", () => {
    expect(inferenceProbeSource).toContain("sourceBadge");
    expect(inferenceProbeSource).toContain("Synthetic probe");
    expect(inferenceProbeSource).toContain("Probe output");
    expect(inferenceProbeSource).toContain("Recognized");
  });

  it("makes training steps configurable from the main workflow", () => {
    expect(appSource).toContain("trainingSteps");
    expect(controlRailSource).toContain("Training Steps");
    expect(controlRailSource).toContain("onTrainingStepsChange");
  });

  it("makes telemetry stride configurable separately from training steps", () => {
    expect(appSource).toContain("telemetryStride");
    expect(controlRailSource).toContain("Telemetry Stride");
    expect(controlRailSource).toContain("onTelemetryStrideChange");
    expect(appSource).toContain("trainResourceRun(sourceRecipe.files, sourceRecipe.entryFile, steps, stride)");
  });

  it("keeps telemetry legend and x-axis labels from overlapping", () => {
    expect(chartsSource).toContain("containLabel: true");
    expect(chartsSource).toContain("hideOverlap: true");
    expect(chartsSource).toContain("legend:");
    expect(chartsSource).toContain("top: 2");
  });
});
