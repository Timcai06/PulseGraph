# Workspace And Training Loop UX

## Decision

PulseGraph will move from one overloaded monitor page to four task workspaces while preserving the current observability cockpit. The Training Loop will become an event-driven navigator with three levels: run lifecycle, repeated train step, and periodic milestones.

## Product Model

The shared object hierarchy is:

`Resource -> Run -> Step -> Node or Sample`

The workspaces are:

- `Prepare`: import and validate a resource, inspect samples and the preflight operator graph;
- `Train`: follow lifecycle, step execution, Ops, telemetry, Live replay, and diagnostics;
- `Evaluate`: run inference and inspect task-specific output without permanently occupying Train;
- `Runs`: browse live and completed runs, reports, and replay entry points.

## Training Loop Model

The loop is not another model graph or metric panel. It answers where execution is, whether a stage completed, and which existing evidence surface should open next.

Three levels:

1. Lifecycle: queued, loading, building, preparing, initializing, training, checkpointing, completed.
2. Step loop: data, forward, loss, backward, optimizer, repeated for each step.
3. Milestones: checkpoint and evaluation, which branch from the repeated loop instead of pretending to run every step.

## Stage Tasks

### Task 1: Emit Truthful Stage Events

- [x] Add a typed `training_stage` event with scope, stage, state, step, message, and optional duration.
- [x] Instrument resource loading, model building, data preparation, optimizer initialization, step execution, checkpoint, and evaluation.
- [x] Preserve coarse `run_status` events for compatibility.
- [x] Test ordering, persistence, cancellation, and failure behavior.

### Task 2: Introduce Task Workspaces

- [x] Replace Monitor and History navigation with Prepare, Train, Evaluate, and Runs.
- [x] Keep the selected resource and run context while switching workspaces.
- [x] Reflect the workspace in the URL hash for reload and deep-link stability.
- [x] Move permanent inference content out of the Train telemetry dock.

### Task 3: Rebuild Training Loop Navigation

- [x] Render lifecycle, repeated step loop, and milestones as separate visual levels.
- [x] Prefer `training_stage` events and use legacy evidence only for recorded older runs.
- [x] Show the real active stage, current step, completion, and message in the collapsed handle.
- [x] Make stage selection request the relevant Ops, telemetry, diagnostics, checkpoint, or evaluation surface.

### Task 4: Preserve Workbench Quality

- [x] Keep Ops, Telemetry, Live, node inspection, inference renderers, and reports intact.
- [x] Keep dark and light themes aligned.
- [x] Verify constrained desktop and mobile layouts.
- [x] Avoid adding another permanent panel or nested card hierarchy.

## Acceptance Criteria

- [x] Train no longer permanently renders Inference Output beside Telemetry.
- [x] Evaluate provides a focused inference workspace using the same active run.
- [x] The Training Loop distinguishes lifecycle, per-step work, and milestones.
- [x] The active stage comes from backend execution events for new runs.
- [x] Clicking Forward, Loss, Backward, Checkpoint, or Eval moves to relevant evidence.
- [x] Existing classification and detection resources still train, replay, infer, and report.
- [x] Complete backend and frontend suites pass with dark and light visual verification.
