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
frontend/    React, React Flow, ECharts, GSAP (hand-rolled CSS design tokens)
backend/     FastAPI, PyTorch, event stream, model inspector, runtime hooks
client/      zero-dependency telemetry emitter for training scripts
docs/        design specs and architecture notes
```

## Current MVP

The current implementation includes:

- FastAPI backend.
- Safe `.pt` upload inspection with `torch.load(..., weights_only=True)`.
- SHA-256 artifact identity for uploaded model files.
- Inferred graph generation from state dict tensor names and shapes.
- Trusted demo MLP runtime backed by real data: it loads the trained checkpoint from
  `exercises/02_mnist_digit_recognition/outputs/mnist_mlp.pt` and real MNIST test digits
  (`data/mnist/MNIST/raw`), falling back to random weights and synthetic digits when absent.
  Override paths with `PULSEGRAPH_MODEL_PATH` and `PULSEGRAPH_MNIST_DIR`.
- Live run ingestion: training scripts POST batched events to `/api/runs/{run_id}/events`,
  the dashboard lists runs from `/api/runs` and follows them over SSE at `/api/runs/{run_id}/stream`.
- `client/pulsegraph_client.py`: a stdlib-only emitter (`PulseGraphRun`) that batches events on a
  background thread; `exercises/02_mnist_digit_recognition/02_train_mlp.py` uses it to stream real
  training metrics, layer snapshots, infra telemetry, and checkpoint events.
- SSE demo training stream with metric, layer, infra, checkpoint, animation, and completion events.
- React dashboard with React Flow model graph, ECharts metrics/probabilities, GSAP node pulse
  animation, live-run picker, and empty/loading/error states.

## Stream a Real Training Run

With the backend running, start the MNIST exercise in another terminal:

```bash
cd /Users/tim/Documents/ai_infra
/opt/homebrew/Caskroom/miniconda/base/envs/ai_infra/bin/python exercises/02_mnist_digit_recognition/02_train_mlp.py
```

The run appears under "Live Runs" in the dashboard; click it to follow metrics live.
Set `PULSEGRAPH_URL=""` to disable telemetry, or point it at a non-default backend URL.

## Run Locally

Install backend dependencies:

```bash
cd /Users/tim/Documents/ai_infra/projects/pulsegraph
/opt/homebrew/Caskroom/miniconda/base/envs/ai_infra/bin/python -m pip install -r backend/requirements.txt
```

Start backend:

```bash
cd /Users/tim/Documents/ai_infra/projects/pulsegraph
/opt/homebrew/Caskroom/miniconda/base/envs/ai_infra/bin/python -m uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8010
```

Install frontend dependencies:

```bash
cd /Users/tim/Documents/ai_infra/projects/pulsegraph/frontend
npm install
```

Start frontend:

```bash
cd /Users/tim/Documents/ai_infra/projects/pulsegraph/frontend
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

## Verify

Backend tests:

```bash
cd /Users/tim/Documents/ai_infra/projects/pulsegraph
/opt/homebrew/Caskroom/miniconda/base/envs/ai_infra/bin/python -m pytest backend -q
```

Frontend build:

```bash
cd /Users/tim/Documents/ai_infra/projects/pulsegraph/frontend
npm run build
```

## Design Docs

- `docs/superpowers/specs/2026-07-08-pulsegraph-design.md`
- `docs/superpowers/plans/2026-07-08-pulsegraph-mvp.md`
