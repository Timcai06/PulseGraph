# PulseGraph Plan Index

This directory is the working plan surface for PulseGraph product and implementation decisions.

## Current Focus

PulseGraph is restructuring its observability cockpit around task workspaces and a truthful, event-driven Training Loop.

The product direction is:

- keep Ops, telemetry, graph replay, inference output, and reports as the core differentiation;
- separate Prepare, Train, Evaluate, and Runs without losing the active resource or run;
- represent lifecycle, repeated train steps, and checkpoint/evaluation milestones at distinct levels;
- drive the Training Loop from execution events instead of UI inference;
- keep Ops, telemetry, replay, inference, and reports as coordinated evidence surfaces;
- stay local and trusted for now, without cloud sandbox complexity;
- add real datasets only when a specific model/task needs them.

## Documents

- [08-workspace-training-loop-ux.md](08-workspace-training-loop-ux.md): current-stage plan for task workspaces and event-driven training navigation.

## Working Rule

Every platform change should preserve this hierarchy:

1. Observe: show current run health without asking the user to dig.
2. Diagnose: let the user expand a stage, click a node, or choose a step to inspect evidence.
3. Act: expose replay, report, export, and next-run actions only when they are relevant.

Task support is not complete until it works across resource loading, training, replay, inference rendering, telemetry, and report surfaces.
