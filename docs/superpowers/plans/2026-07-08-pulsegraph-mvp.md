# PulseGraph MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable PulseGraph MVP that safely inspects `.pt` files, displays an inferred model graph, runs a trusted demo model, and streams training/infra events to a GSAP-enhanced dashboard.

**Architecture:** FastAPI owns model inspection, trusted runtime, and SSE events. React owns the dashboard, React Flow graph, ECharts charts, and GSAP pulse animation. The default `.pt` path is safe inspection only; dynamic execution is only through trusted adapters.

**Tech Stack:** FastAPI, PyTorch, Pydantic, pytest, React, TypeScript, Vite, React Flow, ECharts, GSAP, `@gsap/react`.

---

### Task 1: Backend Contracts and Inspector

**Files:**
- Create: `backend/app/schemas.py`
- Create: `backend/app/inspector/pt_inspector.py`
- Create: `backend/app/inspector/graph_builder.py`
- Test: `backend/tests/test_inspector.py`

- [ ] Define Pydantic contracts for tensor summaries, graph nodes, graph edges, inspection responses, predictions, and run events.
- [ ] Implement `inspect_state_dict()` for dictionaries of tensors and checkpoint bundles.
- [ ] Implement `build_inferred_graph()` for linear/conv-like tensor keys.
- [ ] Test state dict inspection and graph inference.

### Task 2: Trusted Demo Runtime

**Files:**
- Create: `backend/app/runtime/demo_mlp.py`
- Test: `backend/tests/test_demo_runtime.py`

- [ ] Implement a trusted MNIST-style MLP with deterministic sample generation.
- [ ] Expose graph metadata, forward steps, activations, logits, and probabilities.
- [ ] Test prediction output shapes and probability sum.

### Task 3: Event Stream and FastAPI Routes

**Files:**
- Create: `backend/app/events/training_stream.py`
- Create: `backend/app/main.py`
- Test: `backend/tests/test_api.py`

- [ ] Implement demo SSE event generator with metric, layer, infra, checkpoint, animation, and run_complete events.
- [ ] Add `/health`, `/api/demo/model`, `/api/demo/forward`, `/api/runs/demo/stream`, `/api/inspect/upload`.
- [ ] Test health, demo model, demo forward, and upload inspection.

### Task 4: Frontend Dashboard

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/api/client.ts`
- Create: `frontend/src/components/*`
- Create: `frontend/src/styles/app.css`

- [ ] Scaffold Vite React TypeScript app.
- [ ] Implement API client and SSE consumer.
- [ ] Implement dark console layout.
- [ ] Render React Flow graph and layer inspector.
- [ ] Render ECharts metrics and probabilities.
- [ ] Implement GSAP pulse animation for selected/pulsed graph nodes.

### Task 5: Docs, Scripts, Verification

**Files:**
- Modify: `README.md`
- Create: `backend/requirements.txt`
- Create: `backend/pytest.ini`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tsconfig.json`

- [ ] Document startup commands.
- [ ] Install dependencies.
- [ ] Run backend tests.
- [ ] Build frontend.
- [ ] Commit the MVP.

