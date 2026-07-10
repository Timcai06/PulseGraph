# PulseGraph Plan Index

This directory is the working plan surface for PulseGraph product and implementation decisions.

## Current Focus

PulseGraph is validating its general CV runtime against a mainstream detector and real images before adding another task family.

The product direction is:

- keep Ops, telemetry, graph replay, inference output, and reports as the core differentiation;
- preserve explicit task, dataset, output, metric, and renderer contracts;
- make resource startup observable and remove duplicate model construction;
- allow resources to own optimizer, training-step, and evaluation behavior without bypassing telemetry;
- validate object detection with Faster R-CNN, Penn-Fudan images, and honestly named AP@0.50 metrics;
- stay local and trusted for now, without cloud sandbox complexity;
- add real datasets only when a specific model/task needs them.

## Documents

- [07-real-cv-compatibility.md](07-real-cv-compatibility.md): current-stage plan for mainstream real-model compatibility.

## Working Rule

Every platform change should preserve this hierarchy:

1. Observe: show current run health without asking the user to dig.
2. Diagnose: let the user expand a stage, click a node, or choose a step to inspect evidence.
3. Act: expose replay, report, export, and next-run actions only when they are relevant.

Task support is not complete until it works across resource loading, training, replay, inference rendering, telemetry, and report surfaces.
