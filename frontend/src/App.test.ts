import { describe, expect, it } from "vitest";
import appSource from "./App.tsx?raw";
import controlRailSource from "./components/ControlRail.tsx?raw";
import detectionOverlaySource from "./components/DetectionOverlay.tsx?raw";
import imagePreviewSource from "./components/ImagePreview.tsx?raw";
import inferenceProbeSource from "./components/InferenceProbe.tsx?raw";
import modelGraphSource from "./components/ModelGraphPanel.tsx?raw";
import runDetailPanelSource from "./components/RunDetailPanel.tsx?raw";
import runDetailSectionsSource from "./components/RunDetailSections.tsx?raw";
import runDetailSharedSource from "./components/RunDetailShared.tsx?raw";
import runReportSource from "./components/RunReportView.tsx?raw";
import historyPageSource from "./components/HistoryPage.tsx?raw";
import chartsSource from "./components/Charts.tsx?raw";
import apiHttpSource from "./api/http.ts?raw";
import stageStatsSource from "./components/StageStats.tsx?raw";
import motionSource from "./lib/motion.ts?raw";
import layerHealthSource from "./lib/layerHealth.ts?raw";
import trainingLoopSource from "./lib/trainingLoop.ts?raw";
import trainingLoopStripSource from "./components/TrainingLoopStrip.tsx?raw";
import layerInspectorSource from "./components/LayerInspector.tsx?raw";
import timelineSource from "./lib/timeline.ts?raw";
import timelineScrubberSource from "./components/TimelineScrubber.tsx?raw";
import graphTopologySource from "./lib/graphTopology.ts?raw";
import graphPortsSource from "./lib/graphPorts.ts?raw";
import graphStylesSource from "./styles/modules/graph.css?raw";
import panelStylesSource from "./styles/modules/panels.css?raw";
import apiTypesSource from "./api/types.ts?raw";
import diagnosticsTraySource from "./components/DiagnosticsTray.tsx?raw";
import telemetryPanelSource from "./components/TelemetryPanel.tsx?raw";

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

  it("marks every Python execution request as trusted local code", () => {
    expect(apiHttpSource).toContain('"X-PulseGraph-Trust": "trusted-local-code"');
    expect(apiHttpSource.match(/headers: trustedExecutionHeaders/g)?.length).toBeGreaterThanOrEqual(6);
  });

  it("uses a single training resource workflow instead of source and pt side paths", () => {
    const combined = [appSource, controlRailSource].join("\n");

    expect(combined).toContain("control-tabs");
    expect(combined).toContain("Resource");
    expect(combined).toContain("Train");
    expect(combined).toContain("Run");
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

  it("keeps inference source badges while showing task-neutral classification output", () => {
    expect(appSource).toContain("Inference Output");
    expect(appSource).not.toContain("Recognition Result");
    expect(apiTypesSource).toContain("InferenceOutput");
    expect(apiTypesSource).toContain("task?:");
    expect(apiTypesSource).toContain("output?:");
    expect(inferenceProbeSource).toContain("sourceBadge");
    expect(inferenceProbeSource).toContain("Synthetic probe");
    expect(inferenceProbeSource).toContain("Top Prediction");
    expect(inferenceProbeSource).not.toContain("Recognized");
    expect(inferenceProbeSource).toContain("displayClassName");
    expect(inferenceProbeSource).toContain("ImagePreview");
    expect(inferenceProbeSource).toContain("classification-output");
  });

  it("routes detection rendering through the inference overlay", () => {
    expect(inferenceProbeSource).toContain("resolveInferenceRenderer");
    expect(inferenceProbeSource).toContain("detection-output");
    expect(inferenceProbeSource).toContain("detection-results");
    expect(imagePreviewSource).toContain("DetectionOverlay");
    expect(imagePreviewSource).toContain("image-preview-surface");
    expect(detectionOverlaySource).toContain("tabIndex={0}");
    expect(detectionOverlaySource).toContain("detection-box");
  });

  it("keeps resource import and named report mistakes", () => {
    expect(controlRailSource).toContain("webkitRelativePath");
    expect(controlRailSource).toContain("Folder");
    expect(controlRailSource).toContain("control-mini-action");
    expect(appSource).toContain("preferredResourceEntry");
    expect(controlRailSource).not.toContain("resource-samples");
    expect(controlRailSource).not.toContain("ImagePreview");
    expect(runDetailSharedSource).toContain("displayClassName");
    expect(runReportSource).toContain("image_shape");
    expect(runReportSource).toContain("prediction_name");
  });

  it("keeps vision task metadata out of the control surface", () => {
    expect(apiTypesSource).toContain("DatasetSpec");
    expect(apiTypesSource).toContain("OutputSchema");
    expect(apiTypesSource).toContain("MetricSchema");
    expect(appSource).toContain("datasetSpec: preview.resource.dataset_spec");
    expect(appSource).toContain("outputSchema: preview.resource.output_schema");
    expect(appSource).toContain("metricSchema: preview.resource.metric_schema");
    expect(controlRailSource).not.toContain("resourceContractRows");
    expect(controlRailSource).not.toContain("resource-contract-grid");
    expect(controlRailSource).not.toContain("Preview samples");
  });

  it("passes task-aware metric context into the generic telemetry views", () => {
    expect(appSource).toContain("getRunDetail(stream.runId)");
    expect(appSource).toContain("runContractFromConfig(detail.config)");
    expect(appSource).toContain("const metricTask = activeRunContract?.task ?? prediction?.task ?? sourceRecipe?.summary?.task");
    expect(appSource).toContain("activeRunContract?.metricSchema ?? prediction?.metric_schema");
    expect(appSource).toContain("<StageStats metrics={stream.metrics} task={metricTask} metricSchema={metricSchema} />");
    expect(appSource).toContain("task={metricTask}");
    expect(appSource).toContain("metricSchema={metricSchema}");
  });

  it("can export and copy a shareable run report", () => {
    expect(apiHttpSource).toContain("downloadRunReportMarkdown");
    expect(apiHttpSource).toContain("/report/export.md");
    expect(apiHttpSource).toContain("runReportHtmlUrl");
    expect(apiHttpSource).toContain("/report/export.html");
    expect(runReportSource).toContain("Download report");
    expect(runReportSource).toContain("Printable HTML");
    expect(runReportSource).toContain("Print / PDF");
    expect(runDetailPanelSource).toContain("window.open");
    expect(runReportSource).toContain("Copy link");
    expect(runDetailPanelSource).toContain("navigator.clipboard.writeText");
  });

  it("does not cancel the run report request when loading begins", () => {
    expect(runDetailPanelSource).toContain("const reportAvailable = Boolean");
    expect(runDetailPanelSource).toContain("[report, reportAvailable, runId, tab]");
    expect(runDetailPanelSource).not.toContain('report || reportLoading ||');
  });

  it("renders detection report evidence with the shared image overlay", () => {
    expect(runDetailSharedSource).toContain("evidenceDetection(sample)");
    expect(runDetailSharedSource).toContain("Checkpoint detections for sample");
    expect(runReportSource).toContain("report.detection_analysis.evidence");
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

  it("prioritizes grouped telemetry and demotes raw runtime events into diagnostics", () => {
    expect(appSource).toContain("TelemetryPanel");
    expect(appSource).toContain("DiagnosticsTray");
    expect(appSource).not.toContain("Runtime Events");
    expect(telemetryPanelSource).toContain("Optimization");
    expect(telemetryPanelSource).toContain("Quality");
    expect(telemetryPanelSource).toContain("Infra");
    expect(telemetryPanelSource).toContain("etaSec");
    expect(diagnosticsTraySource).toContain("No warnings or failures");
    expect(diagnosticsTraySource).not.toContain("layer_snapshot");
    expect(appSource).not.toContain("DockSize");
    expect(appSource).not.toContain("dock-size-control");
    expect(appSource).not.toContain("Compact telemetry");
    expect(appSource).not.toContain("Expanded telemetry");
    expect(apiTypesSource).toContain('type: "run_status"');
    expect(appSource).toContain("progress={stream.progress}");
  });

  it("can cancel the active local training run and blocks duplicate starts", () => {
    expect(apiHttpSource).toContain("cancelRun");
    expect(apiHttpSource).toContain("/cancel");
    expect(appSource).toContain("handleCancelRun");
    expect(appSource).toContain('currentRunKind === "resource-training" && stream.status === "streaming"');
    expect(telemetryPanelSource).toContain("onCancel");
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
    expect(runReportSource).toContain("report-nav");
    expect(runReportSource).toContain("Summary");
    expect(runReportSource).toContain("Layer Health");
    expect(runDetailPanelSource).toContain("selectedConfusion");
    expect(runReportSource).toContain("misclassified.filter");
  });

  it("turns live run detail into a polling focus workspace", () => {
    expect(runDetailPanelSource).toContain("POLL_MS");
    expect(runDetailPanelSource).toContain("shouldPollRunDetail");
    expect(runDetailPanelSource).toContain("RunDetailMetricsView");
    expect(runDetailPanelSource).toContain("RunDetailLayersView");
    expect(runDetailPanelSource).toContain("RunDetailArtifactsView");
    expect(runDetailPanelSource).toContain("RunDetailEventLogView");
    expect(runDetailSectionsSource).toContain("detail-config-disclosure");
    expect(runReportSource).toContain("report-export-menu");
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
    expect(controlRailSource).toContain("control-disclosure");
    expect(controlRailSource).toContain("motionDuration");
    expect(controlRailSource).toContain("aria-expanded");
    expect(appSource).toContain(".left-control-drawer");
    expect(graphStylesSource).not.toContain("left: calc(var(--rail-w)");
    expect(panelStylesSource).not.toContain("left: calc(var(--rail-w)");
  });

  it("routes overflow live runs into the filtered run library", () => {
    expect(controlRailSource).toContain("onViewAllRuns");
    expect(appSource).toContain('setHistoryMode("live")');
    expect(appSource).toContain('runs={historyMode === "live" ? runBuckets.active : runBuckets.history}');
    expect(appSource).toContain('initialStatusFilter={historyMode === "live" ? "live" : "all"}');
    expect(appSource).toContain('statusFilters={historyMode === "live" ? ["live"] : undefined}');
    expect(historyPageSource).toContain("initialStatusFilter");
    expect(historyPageSource).toContain("statusFilters.length > 1");
    expect(historyPageSource).toContain("{title}");
  });

  it("adds a selected layer inspector to Ops", () => {
    expect(layerInspectorSource).toContain("LayerInspector");
    expect(layerInspectorSource).toContain("activation_sparsity");
    expect(layerInspectorSource).toContain("gradient_norm");
    expect(layerInspectorSource).toContain("onClose");
    expect(appSource).toContain("selectedLayerHistory");
    expect(appSource).toContain("graph-layer-detail-drawer");
  });

  it("adds a time scrubber for replaying Ops telemetry frames", () => {
    expect(timelineSource).toContain("deriveTimelineFrames");
    expect(timelineSource).toContain("layerSnapshotsAtStep");
    expect(timelineSource).toContain("eventsAtTimelineStep");
    expect(timelineScrubberSource).toContain("timeline-scrubber");
    expect(timelineScrubberSource).toContain("timeline-summary");
    expect(timelineScrubberSource).toContain("aria-expanded={open}");
    expect(timelineScrubberSource).toContain('motionDuration("drawer"');
    expect(timelineScrubberSource).toContain("Telemetry timeline");
    expect(appSource).toContain("selectedTimelineStep");
    expect(appSource).toContain("TimelineScrubber");
    expect(appSource).toContain("replayLayerSnapshots");
    expect(chartsSource).toContain("selectedStep");
    expect(chartsSource).toContain("markLine");
  });

  it("surfaces causal debugging focus from timeline replay state", () => {
    expect(timelineSource).toContain("deriveCausalFocus");
    expect(timelineSource).toContain("peakLossStep");
    expect(timelineScrubberSource).toContain("causal-focus");
    expect(appSource).toContain("causalFocus");
    expect(appSource).toContain("replayPulseNodeId");
  });

  it("marks non-linear Ops topology instead of assuming one long line", () => {
    expect(graphTopologySource).toContain("deriveGraphTopology");
    expect(graphTopologySource).toContain("hasBranching");
    expect(graphTopologySource).toContain("hasSkipConnections");
    expect(modelGraphSource).toContain("deriveGraphTopology");
    expect(modelGraphSource).toContain("topology-summary");
    expect(modelGraphSource).toContain("edge-skip");
    expect(modelGraphSource).toContain("topology-");
    expect(modelGraphSource).toContain("edgeClassNameByKind");
  });

  it("keeps optimizer telemetry wired through the stream", () => {
    expect(trainingLoopSource).toContain("latestLearningRate");
    expect(appSource).toContain("deriveTrainingLoopStages");
    expect(appSource).toContain("metrics: stream.metrics");
  });

  it("turns exposed node handles into intentional Composer ports", () => {
    expect(graphPortsSource).toContain("deriveGraphPorts");
    expect(graphPortsSource).toContain("assessGhostEdge");
    expect(graphPortsSource).toContain("wouldCreateCycle");
    expect(modelGraphSource).toContain("composer");
    expect(modelGraphSource).toContain("ghost-edge");
    expect(modelGraphSource).toContain("port-");
    expect(modelGraphSource).toContain("onGhostEdgeSelect");
  });
});
