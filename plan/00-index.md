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

- [01-ops-product-vision.md](01-ops-product-vision.md): what Ops should ultimately become and why.
- [02-ops-information-architecture.md](02-ops-information-architecture.md): screen structure, hierarchy, and interaction model.
- [03-ops-data-contract.md](03-ops-data-contract.md): existing data we can use now and future telemetry needed later.
- [04-ops-v2-implementation-plan.md](04-ops-v2-implementation-plan.md): executable task plan for the next build phase.
- [05-ops-composer.md](05-ops-composer.md): next-stage design for port semantics and Ghost Compose.
- [06-cv-platform-runtime.md](06-cv-platform-runtime.md): current-stage plan for general CV task runtime and object detection support.

## Working Rule

Every platform change should preserve this hierarchy:

1. Observe: show current run health without asking the user to dig.
2. Diagnose: let the user expand a stage, click a node, or choose a step to inspect evidence.
3. Act: expose replay, report, export, and next-run actions only when they are relevant.

Task support is not complete until it works across resource loading, training, replay, inference rendering, telemetry, and report surfaces.
