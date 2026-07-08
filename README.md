# PulseGraph

PulseGraph is a local-first PyTorch model, training, and infrastructure visualization dashboard.

Meaning:

- Pulse: live training signals such as activations, gradients, metrics, checkpoints, and runtime telemetry.
- Graph: model structure, tensor flow, layer relationships, and execution traces.

Product direction:

- Netron-like `.pt` model inspection.
- Training-process visualization for trusted runnable models.
- Infra-oriented observability for step time, throughput, memory, checkpoints, and artifacts.
- Plugin-based visual panels for MLP, CNN, Transformer, Autoencoder, and future LLM fine-tuning workflows.

Core boundary:

- Any `.pt`: safe parameter and checkpoint inspection first.
- Trusted runnable `.pt`: forward execution, hooks, activations, predictions, and dynamic graph visualization.
- Training runs: event-streamed metrics, layer snapshots, infra telemetry, and checkpoint timeline.

Initial architecture:

```text
frontend/    React, React Flow, ECharts, GSAP, shadcn/ui
backend/     FastAPI, PyTorch, event stream, model inspector, runtime hooks
docs/        design specs and architecture notes
```
