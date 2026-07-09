# PulseGraph Plan Index

This directory is the working plan surface for PulseGraph product and implementation decisions.

## Current Focus

Ops v2 is now the baseline cockpit: the app shows the training loop, operator health, selected-layer evidence, telemetry replay, causal focus, and non-linear graph topology.

The next focus is Ops Composer. Composer turns operator ports from accidental connection handles into a deliberate model-design surface:

- keep Monitor mode as the trusted runtime view;
- introduce port semantics for input and output tensors;
- add a Ghost Compose mode for safe, reversible structure experiments;
- use shape and telemetry evidence to explain whether a proposed connection is plausible;
- defer real code generation until the ghost model is useful and well bounded.

## Documents

- [01-ops-product-vision.md](01-ops-product-vision.md): what Ops should ultimately become and why.
- [02-ops-information-architecture.md](02-ops-information-architecture.md): screen structure, hierarchy, and interaction model.
- [03-ops-data-contract.md](03-ops-data-contract.md): existing data we can use now and future telemetry needed later.
- [04-ops-v2-implementation-plan.md](04-ops-v2-implementation-plan.md): executable task plan for the next build phase.
- [05-ops-composer.md](05-ops-composer.md): next-stage design for port semantics and Ghost Compose.

## Working Rule

Every Ops change should preserve this hierarchy:

1. Observe: show current run health without asking the user to dig.
2. Diagnose: let the user expand a stage, click a node, or choose a step to inspect evidence.
3. Act: expose replay, report, export, and next-run actions only when they are relevant.
