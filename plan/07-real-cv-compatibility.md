# Real CV Compatibility

## Decision

PulseGraph already supports classification and object detection across training, telemetry, replay, inference, and reports. This stage validates that architecture against a mainstream detector and real images instead of adding another task family.

The reference workload is Penn-Fudan pedestrian detection with `torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn`. Full data remains under the workspace `data/` boundary; the uploadable resource remains under `model_repo/pulsegraph_resources/real_detection/`.

## Product Boundary

In scope:

- fast handoff from validated resource preview to an observable training run;
- resource-owned optimizer, training-step, and evaluation hooks;
- explicit startup phases before the first training metric;
- honest AP@0.50, precision@0.50, recall@0.50, and mean-IoU telemetry;
- backward compatibility for existing classification and detection resources;
- preservation of the current Ops, telemetry, replay, inference, and report UI.

Deferred:

- segmentation, keypoints, OCR, and embedding task runtimes;
- COCO-style AP averaged across IoU 0.50:0.95;
- dependency installation from uploaded resources;
- cloud sandboxing and distributed training.

## Stage Tasks

### Task 1: Remove Duplicate Training Startup Work

- [x] Fingerprint validated previews and keep a bounded in-memory preflight cache.
- [x] Reuse cached resource metadata and graph when the same upload starts training.
- [x] Return a run ID before the model is rebuilt for execution.
- [x] Preserve the direct API path when no preview cache exists.

### Task 2: Add Resource Runtime Hooks

- [x] Support an optional `build_optimizer(model)` hook.
- [x] Support an optional resource-owned `training_step(...)` hook.
- [x] Support an optional `evaluation_metrics(...)` hook.
- [x] Validate hook outputs and preserve default runtime behavior when hooks are absent.

### Task 3: Make Startup Observable

- [x] Publish loading, model-building, data-preparation, and optimizer-initialization phases.
- [x] Persist each phase in run configuration.
- [x] Keep cancellation and failure behavior valid before step one.
- [x] Surface the current phase and message in Training Telemetry without adding a new panel.

### Task 4: Validate A Mainstream Real Detector

- [x] Use Faster R-CNN MobileNetV3 FPN with real Penn-Fudan images.
- [x] Use the resource optimizer hook for a detector-appropriate SGD configuration.
- [x] Publish AP@0.50, precision@0.50, recall@0.50, and mean IoU.
- [x] Rebuild the governed upload zip without moving raw data into the repository.

### Task 5: Verify The Full Workbench

- [x] Run focused contract and API tests for hooks and startup lifecycle.
- [x] Run the complete backend and frontend suites.
- [x] Execute a real-resource smoke training run.
- [x] Verify dark and light UI states without reducing the current visual quality.

## Acceptance Criteria

This stage is complete when:

- [x] starting a previously previewed real resource no longer rebuilds it before returning the run ID;
- [x] legacy resources train unchanged without implementing new hooks;
- [x] hook failures become explicit run failures with actionable messages;
- [x] Faster R-CNN trains from real images and emits honest task metrics;
- [x] the operator graph, telemetry, Live timeline, inference overlay, replay, and report remain usable;
- [x] generated runs, datasets, checkpoints, builds, and screenshots remain outside Git.
