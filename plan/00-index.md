# PulseGraph Plan Index

This directory is the working plan surface for PulseGraph product and implementation decisions.

## Current Focus

Ops v2 turns the current operator graph from a layer chain into a training runtime cockpit:

- show the training loop as the primary mental model;
- keep the operator graph as the forward execution map;
- add layer health overlays from existing telemetry;
- open a focused layer inspector when a node is selected;
- prepare the UI for future time-scrubbing and causal debugging.

## Documents

- [01-ops-product-vision.md](01-ops-product-vision.md): what Ops should ultimately become and why.
- [02-ops-information-architecture.md](02-ops-information-architecture.md): screen structure, hierarchy, and interaction model.
- [03-ops-data-contract.md](03-ops-data-contract.md): existing data we can use now and future telemetry needed later.
- [04-ops-v2-implementation-plan.md](04-ops-v2-implementation-plan.md): executable task plan for the next build phase.

## Working Rule

Every Ops change should preserve this hierarchy:

1. Observe: show current run health without asking the user to dig.
2. Diagnose: let the user click a stage, node, or step to inspect evidence.
3. Act: expose replay, report, export, and next-run actions only when they are relevant.

