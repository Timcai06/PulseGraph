# Attach-First Training Investigation

## Objective

Prove that PulseGraph can investigate a real PyTorch training script without owning its training loop.

The reference workload is `training/llm/minigpt.py`. This stage adds mixed precision and optional `torch.compile`, emits truthful semantic training spans through the PulseGraph client, and turns the resulting run into a linked investigation record.

## Product Decision

PulseGraph has two integration modes:

- **Managed Lab**: the existing trusted Resource contract runs small, reproducible experiments inside PulseGraph.
- **Attach & Observe**: an existing PyTorch script keeps control of data, model, optimizer, precision, accumulation, compilation, and checkpointing while PulseGraph records evidence.

Attach & Observe is the long-term product path. This stage must not expand PulseGraph into another training framework.

## Evidence Model

The investigation hierarchy is:

`Run -> Span -> Evidence -> Artifact`

A span represents real execution such as data loading, forward, loss, backward, optimizer update, evaluation, compilation, or checkpointing. Evidence includes metrics, layer summaries, system telemetry, compiler diagnostics, samples, and anomalies. Artifacts include source, configuration, environment, datasets, and checkpoints with fingerprints.

## Task 1: Semantic Client Spans

- [ ] Extend the event schema backward-compatibly with trace, parent, rank, device, phase, and micro-step context.
- [ ] Add client APIs or context managers for lifecycle, step, and milestone spans with active, completed, failed, and cancelled outcomes.
- [ ] Make attached runs drive the existing Training Loop from emitted events instead of legacy UI inference.
- [ ] Preserve non-blocking batching, bounded memory, and fail-open behavior when the PulseGraph backend is unavailable.
- [ ] Add schema, ordering, exception, batching, and compatibility tests.

## Task 2: Instrument The Real MiniGPT Loop

- [ ] Add explicit CLI configuration for precision, compilation, capture level, and telemetry stride to `training/llm/minigpt.py`.
- [ ] Implement mixed precision with device-aware autocast and scaling behavior without changing full-precision defaults.
- [ ] Add optional `torch.compile` with a clear unsupported-platform fallback and captured compile state.
- [ ] Emit lifecycle and step spans around data, forward, loss, backward, optimizer, evaluation, and checkpoint operations.
- [ ] Record code, arguments, seed, device, PyTorch version, dataset identity, and checkpoint fingerprint.
- [ ] Keep the training script directly runnable when PulseGraph is stopped or disabled.

## Task 3: Compiler And Precision Evidence

- [ ] Distinguish compile cold start from steady-state step timing.
- [ ] Record precision mode, scaler state, skipped updates, non-finite loss, and gradient norm evidence where available.
- [ ] Capture graph-break and recompile summaries through supported PyTorch diagnostics as optional artifacts.
- [ ] Treat compiler diagnostics as evidence linked to spans, not as permanent raw-log UI.
- [ ] Measure and report telemetry overhead for `off`, `metrics`, and `standard` capture levels.

## Task 4: Investigation Experience

- [ ] Show attached runs as first-class live runs with truthful active phase and step context.
- [ ] Add anomaly markers for non-finite values, loss spikes, step-time regressions, and compiler interruptions.
- [ ] Let an anomaly open one focused evidence view linking timeline, operation or layer, metrics, runtime context, and artifacts.
- [ ] Keep deep compiler and layer evidence behind on-demand disclosure rather than adding another permanent panel.
- [ ] Preserve Prepare, Train, Evaluate, Runs, current themes, and responsive behavior.

## Task 5: Run Comparison Proof

- [ ] Produce comparable eager and compiled MiniGPT runs from the same seed and configuration.
- [ ] Show compile startup cost, steady-state step time, loss trajectory, capture overhead, and checkpoint provenance together.
- [ ] Generate an evidence-backed summary that distinguishes observations from diagnostic hypotheses.
- [ ] Verify replay and report generation for the attached run without reconstructing the training loop inside PulseGraph.

## Acceptance Criteria

- [ ] MiniGPT remains the owner of its complete training loop.
- [ ] One command creates a PulseGraph run with semantic spans, metrics, runtime evidence, provenance, and checkpoint artifacts.
- [ ] Training continues successfully when PulseGraph is unavailable.
- [ ] The Training Loop is event-driven for attached runs and represents micro-steps or custom phases without inventing internals.
- [ ] Eager and compiled runs can be compared using recorded evidence.
- [ ] Capture overhead is measured, visible, and separated by capture level.
- [ ] Existing classification and detection Resource workflows remain compatible.
- [ ] Backend and frontend suites, production build, and desktop/mobile visual checks pass.

## Explicitly Out Of Scope

- new CV task runtimes such as segmentation, OCR, or keypoints;
- cloud deployment, authentication, teams, queues, or cluster orchestration;
- automatic model editing or executable Composer graphs;
- a general AI diagnosis assistant before the evidence model is trustworthy;
- replacing JSONL storage before attached-run query volume proves the need.
