# Ops Product Vision

## North Star

PulseGraph Ops should become a live runtime map for PyTorch training and inference.

It should not stop at drawing model layers. It should explain how a training run moves through data loading, forward execution, loss, backward gradient flow, optimizer updates, checkpointing, evaluation, and infra constraints.

## The Mental Model

Current simple view:

```text
Input -> Layer -> Layer -> Output
```

Target Ops view:

```text
Data Loader -> Batch -> Forward Graph -> Loss -> Backward -> Optimizer -> Checkpoint -> Eval
                    |                                                  |
                    +-------------------- next step -------------------+
```

The operator graph remains important, but it sits inside the Forward stage. Training is a loop, not a line.

## Product Promise

Ops should answer three questions quickly:

1. What is running?
2. Where is the signal right now?
3. Where is training unhealthy or wasteful?

## High-End Future Shape

### Training Loop Map

The first view shows the whole training system:

- Data: batch size, sample source, data wait, class distribution.
- Forward: operator graph, shapes, layer latency, activation health.
- Loss: loss value, spike detection, plateau state.
- Backward: gradient norm, vanishing or exploding risk.
- Optimizer: learning rate, update magnitude, parameter drift.
- Checkpoint: save time, size, fingerprint, best checkpoint.
- Eval: accuracy, confusion matrix, misclassified samples.
- Infra: device, step time, memory, throughput, queue or worker state.

### Hierarchical Operator Map

The graph should support three levels:

1. Module level: Backbone, Head, Loss, Optimizer.
2. Block level: Residual Block, Attention Block, MLP Block.
3. Operator level: Conv2d, MatMul, Add, LayerNorm, GELU.

Simple models can still render as a line. Complex models should show branches, skip connections, nested modules, and repeated blocks without becoming a long unreadable chain.

### Health Overlay

Every operator node should become a live health surface:

- shape and parameter count;
- safe, inferred, or trusted graph confidence;
- activation sparsity;
- gradient norm;
- weight standard deviation;
- current runtime status.

Health should be visible without opening a modal.

### Time Machine

Training should be replayable by step. Scrubbing the time axis should update:

- graph node health;
- selected layer inspector;
- metric cursor;
- nearby runtime events;
- checkpoint and report context.

### Causal Debugger

The long-term product should connect symptoms to evidence:

```text
loss spike at step 240
  -> unusual batch distribution
  -> block3.relu activation sparsity 0.97
  -> head gradient norm dropped 10x
  -> checkpoint at step 200 was healthier
```

This is the difference between a dashboard and an infra diagnostic tool.

## What Not To Do Yet

Do not add a generic chat box before the evidence model exists. An AI copilot should come after the app can cite structured telemetry, layer health, checkpoints, and report findings.

Do not make the graph decorative. Motion and visual polish must communicate runtime state.

