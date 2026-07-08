# PulseGraph Design Spec

## Product

PulseGraph is a local-first PyTorch model, training, and infrastructure visualization dashboard for AI infra engineers.

It combines:

- Netron-like `.pt` inspection.
- Training-process visualization for trusted runnable models.
- Infra-oriented telemetry such as step time, throughput, memory, checkpoints, and artifacts.
- Plugin-shaped visualization panels for MLP, CNN, Transformer, Autoencoder, and future LLM fine-tuning workflows.
- GSAP-driven motion for training pulses, forward/backward flow, and state emphasis.

## Boundary

PulseGraph treats model files by trust level:

- Any `.pt`: safe parameter/checkpoint inspection only.
- Trusted runnable `.pt`: model adapter may load code, run forward, collect hooks, and expose activations.
- Training runs: instrumented trainer emits metrics, layer snapshots, infra telemetry, checkpoints, and animation events.

The default `.pt` path must not execute arbitrary pickled Python model code.

## MVP Scope

The first working version covers:

1. Backend and frontend project skeleton.
2. Safe `.pt` state/checkpoint inspection.
3. Netron-like inferred graph from tensor keys and shapes.
4. React Flow graph display and layer inspector.
5. Trusted MNIST-style MLP demo forward with activations and prediction probabilities.
6. Simulated training event stream with metric, layer, infra, checkpoint, and animation events.
7. Dark Lab Console visual language with GSAP node pulse animation.

## Architecture

```text
frontend React app
  |
  | REST + SSE
  v
FastAPI backend
  |
  +-- inspector: safe torch.load and tensor stats
  +-- graph: inferred graph nodes and edges
  +-- runtime: trusted demo model and hook-style snapshots
  +-- events: training event stream
  +-- plugins: algorithm visual metadata
  |
  v
PyTorch runtime
```

## Frontend

Stack:

- React + TypeScript + Vite.
- React Flow for model graph.
- ECharts for metric and probability charts.
- GSAP + `@gsap/react` for semantic training pulses.
- CSS modules/plain CSS for MVP; Tailwind/shadcn can be layered later.
- Zustand-style store is optional for MVP; React state is acceptable until cross-panel state grows.

Layout:

- Top status bar: product, backend status, run status, device, step.
- Left rail: file inspector controls and trusted demo controls.
- Center: model graph.
- Right inspector: selected layer metadata, tensor stats, activation summary.
- Bottom dock: loss/accuracy/infra curves and event log.

GSAP rule:

- GSAP animates only DOM emphasis and flow pulses.
- React owns state.
- React Flow owns graph layout.
- ECharts owns chart transitions.
- GSAP uses scoped refs and cleans up via `useGSAP`.
- Reduced-motion support must disable pulse loops.

## Backend

Backend modules:

- `app.schemas`: Pydantic contracts shared by APIs.
- `app.inspector.pt_inspector`: safe `.pt` loading, checkpoint selection, tensor summaries.
- `app.inspector.graph_builder`: inferred graph from state dict keys.
- `app.runtime.demo_mlp`: trusted MLP adapter and sample generation.
- `app.events.training_stream`: async event generator for run visualization.
- `app.main`: FastAPI routes.

Routes:

- `GET /health`
- `GET /api/demo/model`
- `GET /api/demo/forward?index=0`
- `GET /api/runs/demo/stream`
- `POST /api/inspect/upload`

## Event Protocol

Events are JSON objects streamed over SSE:

```json
{
  "type": "metric",
  "run_id": "demo-run",
  "step": 12,
  "epoch": 1,
  "payload": {
    "loss": 0.42,
    "accuracy": 0.88
  }
}
```

Required event types:

- `metric`
- `layer_snapshot`
- `infra`
- `checkpoint`
- `animation`
- `run_complete`

## Plugin Model

Plugins expose visualization intent rather than owning the whole app:

```text
Plugin id
Supported layer/model patterns
Panel descriptors
Snapshot extractors
Recommended charts
```

MVP ships a built-in MLP visual profile. CNN/Transformer plugins are planned but not required for the first runnable app.

## Acceptance Criteria

- Backend tests pass for `.pt` inspector, graph inference, demo forward, and event stream shape.
- Frontend builds successfully.
- Running backend + frontend shows:
  - graph nodes and edges,
  - layer inspector,
  - demo prediction probabilities,
  - streaming training metrics,
  - GSAP pulse on animation events.
- README documents startup commands.

