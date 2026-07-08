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
  `model_repo/platform_validation/mnist_digit_recognition/outputs/mnist_mlp.pt` and real MNIST test digits
  (`data/mnist/MNIST/raw`), falling back to random weights and synthetic digits when absent.
  Override paths with `PULSEGRAPH_MODEL_PATH` and `PULSEGRAPH_MNIST_DIR`.
- Live run ingestion: training scripts POST batched events to `/api/runs/{run_id}/events`,
  the dashboard lists runs from `/api/runs` and follows them over SSE at `/api/runs/{run_id}/stream`.
- `client/pulsegraph_client.py`: a stdlib-only emitter (`PulseGraphRun`) that batches events on a
  background thread; `model_repo/platform_validation/mnist_digit_recognition/02_train_mlp.py` uses it to stream real
  training metrics, layer snapshots, infra telemetry, and checkpoint events.
- SSE demo training stream with metric, layer, infra, checkpoint, animation, and completion events.
- React dashboard with React Flow model graph, ECharts metrics/probabilities, GSAP node pulse
  animation, live-run picker, and empty/loading/error states.

## Training Provenance (record everything, replay anything)

PulseGraph records the full context of a training run so any resulting `.pt` file can be
traced back and replayed later:

- `register_source` / `register_config` / `register_graph`: model source, hyperparameters,
  and the exact `torch.fx` compute graph are captured at training time.
- Probe samples (a small input batch) are stored with the run so replay works for any data domain.
- Per-epoch checkpoints are uploaded to `backend/runs/{run_id}/checkpoints/` with a
  **canonical weights fingerprint** (name + shape + dtype + raw bytes), independent of
  serialization format.
- Uploading any `.pt`/`.safetensors` matches by fingerprint and opens the run's full archive.
- `GET /api/runs/{id}/forward` rebuilds the model from recorded source + checkpoint and runs
  inference with per-layer activation capture. Only source recorded into the local runs/
  store is ever executed.
- `GET /api/runs/{id}/report` produces a diagnosis report: overfit gap, loss plateau,
  layer health (dead neurons, gradient anomalies), per-checkpoint probe accuracy,
  confusion matrix with misclassified samples, and rule-based insights.

## Stream a Real Training Run

With the backend running, train via the persistent training layer:

```bash
cd /Users/tim/Documents/ai_infra
/opt/homebrew/Caskroom/miniconda/base/envs/ai_infra/bin/python training/train.py --model mlp --epochs 3
/opt/homebrew/Caskroom/miniconda/base/envs/ai_infra/bin/python training/train.py --model cnn --epochs 2
```

The run appears under "Live Runs" in the dashboard: click it to follow metrics live, or open
its detail view (ⓘ) for source, config, checkpoint timeline, replay, and the analysis report.
Set `PULSEGRAPH_URL=""` to disable telemetry, or point it at a non-default backend URL.
The client is an installable package: `pip install -e projects/pulsegraph/client`.

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
