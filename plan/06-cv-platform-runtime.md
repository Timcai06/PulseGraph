# CV Platform Runtime

## Decision

PulseGraph should continue growing into a local-first general CV training platform while preserving the current visualization and observability cockpit.

The platform should not become a plain script launcher. Each supported task must remain inspectable through graph structure, training telemetry, runtime events, inference output, replay, and reports.

## Product Boundary

Short-term:

- local trusted execution only;
- PyTorch-first resources;
- classification remains fully supported;
- object detection becomes the first non-classification task;
- data is added per task instead of building a generic dataset center first.

Deferred:

- cloud execution and multi-tenant sandboxing;
- code generation from Composer;
- large dataset management;
- distributed training;
- full plugin marketplace.

## Why Object Detection First

Detection is the right first pressure test because it breaks the assumptions still hidden in the current classification path:

- output is a set of boxes, scores, and labels, not one class index;
- metrics are IoU and mAP-like, not simple accuracy;
- inference UI must render overlays on top of the image;
- reports need sample-level visual evidence;
- replay must preserve structured outputs without forcing probabilities.

If detection works cleanly, segmentation, keypoints, OCR, and embeddings can reuse most of the same task-runtime pattern.

## Architecture Target

The next architecture should separate five contracts:

1. `TaskSpec`: task kind and capabilities.
2. `DatasetSpec`: sample shape, label schema, splits, and source.
3. `OutputSchema`: structured model output and renderer hint.
4. `MetricSchema`: task-specific metric names, primary metric, and chart grouping.
5. `RendererRegistry`: frontend mapping from output kind to visual renderer.

Classification should become one implementation of this pattern, not the platform default that every task must imitate.

## Stage Tasks

### Task 1: Make Prediction Output Task-Neutral

- [x] Add a task-neutral inference response shape that can carry structured outputs without requiring `label`, `prediction`, and `probabilities`.
- [x] Keep backward-compatible classification fields while making non-classification outputs valid.
- [x] Add backend tests for classification compatibility and detection-shaped responses.
- [x] Add frontend tests that confirm classification still uses the rich probability view.

### Task 2: Introduce Task Runtime Dispatch

- [x] Replace hard-coded classification branches with a task runtime dispatcher.
- [x] Keep `classification` as the first runtime implementation.
- [x] Add a trainable `detection` runtime that validates resource metadata and preserves structured outputs.
- [x] Make unsupported tasks fail with a clear platform-capability error.

### Task 3: Add Synthetic Detection Resource

- [x] Create a tiny local detection resource with generated images, boxes, labels, and a simple model.
- [x] Use synthetic data first to avoid dataset download friction.
- [x] Ensure preview samples show visible object boxes and label names.
- [x] Keep the resource small enough for fast local verification.

### Task 4: Build Detection Inference Renderer

- [x] Add an image overlay renderer for boxes, scores, and labels.
- [x] Keep the existing classification probability card unchanged.
- [x] Add renderer registry logic keyed by `output.kind` or `output_schema.renderer`.
- [x] Verify dark and light themes visually.

### Task 5: Extend Metrics And Reports

- [x] Add task-specific metric grouping so detection can show loss components and IoU-style signals.
- [x] Keep generic metric charts for unknown metric names.
- [x] Add report sections that can summarize structured outputs and visual evidence.
- [x] Avoid claiming mAP until the evaluator is real.

### Task 6: Preserve Ops Observability

- [x] Ensure graph, layer health, training loop stages, timeline scrubber, and runtime events still work for classification.
- [x] Ensure detection runs still publish graph, infra, checkpoint, and completion events.
- [x] Add visual checks so the UI does not regress while task support expands.

## Acceptance Criteria

This stage is complete when:

- [x] A classification resource still imports, trains, replays, and renders exactly as before.
- [x] A synthetic detection resource imports and previews with boxes.
- [x] Detection inference renders boxes in the bottom output panel.
- [x] The API can return structured detection output without pretending it is classification.
- [x] Tests cover task-neutral response handling and renderer selection.
- [x] Dark and light theme screenshots show no layout or color regression.

## Repository Hygiene During This Stage

Do not reorganize source directories while changing runtime semantics. Keep functional changes small and verifiable.

Allowed cleanup:

- delete ignored generated artifacts when they are no longer useful;
- move reusable external model resources outside this Git repository under the workspace model repository;
- document new runtime boundaries before moving files.

Avoid:

- mixing repo cleanup with task-runtime refactors in the same commit;
- moving backend and frontend files at the same time as changing API contracts;
- committing generated runs, screenshots, builds, caches, or downloaded datasets.
