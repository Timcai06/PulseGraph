import { describe, expect, it } from "vitest";
import appSource from "./App.tsx?raw";
import controlRailSource from "./components/ControlRail.tsx?raw";
import imagePreviewSource from "./components/ImagePreview.tsx?raw";
import inferenceProbeSource from "./components/InferenceProbe.tsx?raw";
import modelGraphSource from "./components/ModelGraphPanel.tsx?raw";
import runDetailPanelSource from "./components/RunDetailPanel.tsx?raw";
import historyPageSource from "./components/HistoryPage.tsx?raw";
import chartsSource from "./components/Charts.tsx?raw";
import apiHttpSource from "./api/http.ts?raw";
import stageStatsSource from "./components/StageStats.tsx?raw";
import motionSource from "./lib/motion.ts?raw";
import layerHealthSource from "./lib/layerHealth.ts?raw";
import trainingLoopSource from "./lib/trainingLoop.ts?raw";
import trainingLoopStripSource from "./components/TrainingLoopStrip.tsx?raw";
import layerInspectorSource from "./components/LayerInspector.tsx?raw";
import graphStylesSource from "./styles/modules/graph.css?raw";
import panelStylesSource from "./styles/modules/panels.css?raw";

describe("App workspace layout", () => {
  it("keeps layer details out of the control rail", () => {
    expect(controlRailSource).not.toContain("LayerInspector");
    expect(controlRailSource).not.toContain("layer-inspector");
    expect(appSource).toContain("graph-layer-detail-drawer");
    expect(appSource).toContain("<Layer" + "Inspector");
  });

  it("keeps the monitor free of explanatory demo copy", () => {
    const combined = [appSource, controlRailSource, inferenceProbeSource, imagePreviewSource, modelGraphSource].join("\n");

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

  it("keeps inference source badges while showing class-name results", () => {
    expect(inferenceProbeSource).toContain("sourceBadge");
    expect(inferenceProbeSource).toContain("Synthetic probe");
    expect(inferenceProbeSource).toContain("Recognized");
    expect(inferenceProbeSource).toContain("displayClassName");
    expect(inferenceProbeSource).toContain("ImagePreview");
  });

  it("shows resource preview samples and named report mistakes", () => {
    expect(controlRailSource).toContain("resource-samples");
    expect(controlRailSource).toContain("label_name");
    expect(controlRailSource).toContain("ImagePreview");
    expect(runDetailPanelSource).toContain("displayClassName");
    expect(runDetailPanelSource).toContain("image_shape");
    expect(runDetailPanelSource).toContain("prediction_name");
  });

  it("can export and copy a shareable run report", () => {
    expect(apiHttpSource).toContain("downloadRunReportMarkdown");
    expect(apiHttpSource).toContain("/report/export.md");
    expect(apiHttpSource).toContain("runReportHtmlUrl");
    expect(apiHttpSource).toContain("/report/export.html");
    expect(runDetailPanelSource).toContain("Download report");
    expect(runDetailPanelSource).toContain("Printable HTML");
    expect(runDetailPanelSource).toContain("Print / PDF");
    expect(runDetailPanelSource).toContain("window.open");
    expect(runDetailPanelSource).toContain("Copy link");
    expect(runDetailPanelSource).toContain("navigator.clipboard.writeText");
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

  it("centralizes GSAP motion decisions", () => {
    expect(motionSource).toContain("motionDurations");
    expect(motionSource).toContain("motionEase");
    expect(motionSource).toContain("configureMotionDefaults");
    expect(motionSource).toContain("gsap.defaults");
    expect(motionSource).toContain("gsap.matchMedia");
    expect(motionSource).toContain("prefers-reduced-motion");
    expect([appSource, modelGraphSource, runDetailPanelSource, historyPageSource, stageStatsSource].join("\n")).toContain("../lib/motion");
  });

  it("uses GSAP Flip for high-level view transitions", () => {
    expect(modelGraphSource).toContain("gsap/Flip");
    expect(modelGraphSource).toContain("Flip.getState");
    expect(modelGraphSource).toContain("Flip.from");
    expect(runDetailPanelSource).toContain("gsap/Flip");
    expect(runDetailPanelSource).toContain("shared-detail");
  });

  it("turns history into a filterable run library", () => {
    expect(historyPageSource).toContain("Run Library");
    expect(historyPageSource).toContain("library-controls");
    expect(historyPageSource).toContain("Search runs");
    expect(historyPageSource).toContain("statusFilter");
    expect(historyPageSource).toContain("groupedRuns");
    expect(historyPageSource).toContain("library-section");
  });

  it("adds report navigation and confusion drilldown", () => {
    expect(runDetailPanelSource).toContain("report-nav");
    expect(runDetailPanelSource).toContain("Summary");
    expect(runDetailPanelSource).toContain("Layer Health");
    expect(runDetailPanelSource).toContain("selectedConfusion");
    expect(runDetailPanelSource).toContain("misclassified.filter");
  });

  it("derives operator health from layer snapshots", () => {
    expect(layerHealthSource).toContain("deriveLayerHealth");
    expect(layerHealthSource).toContain("activation_sparsity");
    expect(layerHealthSource).toContain("gradient_norm");
    expect(layerHealthSource).toContain("possible dead layer");
    expect(layerHealthSource).toContain("possible vanishing gradient");
  });

  it("shows richer Ops node health metadata", () => {
    expect(modelGraphSource).toContain("layerSnapshots");
    expect(modelGraphSource).toContain("deriveLayerHealth");
    expect(modelGraphSource).toContain("formatNodeShape");
    expect(modelGraphSource).toContain("node-health");
    expect(modelGraphSource).toContain("node-shape");
  });

  it("keeps the training loop in an integrated top drawer", () => {
    expect(trainingLoopSource).toContain("deriveTrainingLoopStages");
    expect(trainingLoopSource).toContain("Data");
    expect(trainingLoopSource).toContain("Forward");
    expect(trainingLoopSource).toContain("Backward");
    expect(trainingLoopStripSource).toContain("gsap");
    expect(trainingLoopStripSource).toContain("useGSAP");
    expect(trainingLoopStripSource).toContain("drawerRef");
    expect(trainingLoopStripSource).toContain("body.scrollHeight");
    expect(trainingLoopStripSource).toContain("maxExpandedHeight");
    expect(trainingLoopStripSource).toContain("overflowY");
    expect(trainingLoopStripSource).toContain("motionDuration");
    expect(trainingLoopStripSource).toContain("top-training-drawer");
    expect(trainingLoopStripSource).toContain("training-loop-handle");
    expect(trainingLoopStripSource).toContain("training-loop-drawer-body");
    expect(trainingLoopStripSource).toContain("drawerOpen");
    expect(trainingLoopStripSource).toContain("aria-expanded");
    expect(trainingLoopStripSource).not.toContain("stage-detail-panel");
    expect(graphStylesSource).not.toContain("display: none;");
    expect(trainingLoopStripSource).toContain("activeStageId");
    expect(appSource).toContain("TrainingLoopStrip");
  });

  it("turns the left control rail into a GSAP-backed drawer layer", () => {
    expect(controlRailSource).toContain("gsap");
    expect(controlRailSource).toContain("useGSAP");
    expect(controlRailSource).toContain("railDrawerOpen");
    expect(controlRailSource).toContain("drawerRef");
    expect(controlRailSource).toContain("left-control-drawer");
    expect(controlRailSource).toContain("rail-drawer-handle");
    expect(controlRailSource).toContain("motionDuration");
    expect(controlRailSource).toContain("aria-expanded");
    expect(appSource).toContain(".left-control-drawer");
    expect(graphStylesSource).not.toContain("left: calc(var(--rail-w)");
    expect(panelStylesSource).not.toContain("left: calc(var(--rail-w)");
  });

  it("adds a selected layer inspector to Ops", () => {
    expect(layerInspectorSource).toContain("LayerInspector");
    expect(layerInspectorSource).toContain("activation_sparsity");
    expect(layerInspectorSource).toContain("gradient_norm");
    expect(layerInspectorSource).toContain("onClose");
    expect(appSource).toContain("selectedLayerHistory");
    expect(appSource).toContain("graph-layer-detail-drawer");
  });
});
