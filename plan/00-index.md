# PulseGraph Direction

This directory does not archive completed or abandoned stage plans. New numbered plans are not created by default; development follows the current product direction and real workload evidence.

## Product Boundary

PulseGraph is a local-first PyTorch training investigation workbench: a training glass box and, eventually, a training APM.

It should answer:

- where a run first changed or failed;
- which phase, step, operation, layer, sample, or system event was involved;
- which code, configuration, data, environment, and checkpoint produced the observed weights;
- what evidence supports a diagnosis and what experiment should verify it.

PulseGraph is not becoming a generic cloud trainer, cluster orchestrator, dataset platform, or drag-and-drop model builder.

## Current Direction

- Continue deepening computer vision; LLM workloads are paused.
- Prefer one real CV vertical slice over adding many shallow task labels.
- Use a real model, real images, a healthy run, and a controlled faulty run to prove that PulseGraph can locate the first meaningful difference.
- Keep classification and detection working while real workloads determine when segmentation, keypoints, OCR, or embedding support is justified.
- Treat AMP, compilation, profiling, and external-script attachment as supporting mechanisms rather than the product objective.

The development loop is:

`real CV workload -> observable failure -> recorded evidence -> focused product improvement -> regression proof`

## Working Rules

1. Preserve the user's real training script and add low-friction instrumentation around it.
2. Separate lightweight continuous telemetry from opt-in deep capture.
3. Every diagnosis must link to recorded evidence; UI inference is never runtime truth.
4. Add a new task renderer or panel only when a real CV model forces the requirement.
5. Keep the current Resource workflow as a repeatable platform validation path; do not force every real training experiment into that contract.
6. Delete completed or abandoned plan material instead of accumulating plan history.
