# PulseGraph Plan Index

This directory contains only the active product and implementation stage. Completed stage plans are removed rather than archived here.

## Product Boundary

PulseGraph is a local-first PyTorch training investigation workbench: a training glass box and, eventually, a training APM.

It should answer:

- where a run first changed or failed;
- which phase, step, operation, layer, sample, or system event was involved;
- which code, configuration, data, environment, and checkpoint produced the observed weights;
- what evidence supports a diagnosis and what experiment should verify it.

PulseGraph is not becoming a generic cloud trainer, cluster orchestrator, dataset platform, or drag-and-drop model builder.

## Current Stage

- [09-attach-first-training-investigation.md](09-attach-first-training-investigation.md): attach PulseGraph to the existing MiniGPT training loop and capture AMP and `torch.compile` evidence without taking ownership of training.

## Working Rules

1. Preserve the user's real training script and add low-friction instrumentation around it.
2. Separate lightweight continuous telemetry from opt-in deep capture.
3. Every diagnosis must link to recorded evidence; UI inference is never runtime truth.
4. Add a new task renderer or panel only when a real model forces the requirement.
5. Keep the current Resource workflow as Managed Lab while Attach & Observe becomes the long-term integration path.
