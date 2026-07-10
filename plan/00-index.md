# PulseGraph Plan Index

This directory is the working plan surface for PulseGraph product and implementation decisions.

## Current Focus

PulseGraph is moving from a classification-focused visualization workbench into a local-first general CV training platform.

The product direction is:

- keep Ops, telemetry, graph replay, inference output, and reports as the core differentiation;
- support mainstream CV task families through explicit task, dataset, output, metric, and renderer contracts;
- use object detection as the first non-classification slice because it forces boxes, scores, labels, IoU-style metrics, and image overlay rendering;
- stay local and trusted for now, without cloud sandbox complexity;
- add real datasets only when a specific model/task needs them.

## Documents

- [06-cv-platform-runtime.md](06-cv-platform-runtime.md): current-stage plan for general CV task runtime and object detection support.

## Working Rule

Every platform change should preserve this hierarchy:

1. Observe: show current run health without asking the user to dig.
2. Diagnose: let the user expand a stage, click a node, or choose a step to inspect evidence.
3. Act: expose replay, report, export, and next-run actions only when they are relevant.

Task support is not complete until it works across resource loading, training, replay, inference rendering, telemetry, and report surfaces.
