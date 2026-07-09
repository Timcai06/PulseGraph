# Ops Information Architecture

## Screen Model

Ops should keep a stable cockpit layout:

```text
Top Status Bar
  API | run status | device | step | theme

Main Stage
  Training Loop Strip
  Operator Graph / Neural View
  Floating Stage Stats

Right Rail
  Resource import
  Run controls
  Current run state
  Live runs

Bottom Dock
  Timeline and metrics
  Prediction result
  Runtime events
```

## Primary User Flow

1. Import a training resource.
2. Preview graph and samples.
3. Run training.
4. Watch the training loop and operator graph pulse.
5. Click an unhealthy node.
6. Inspect shape, activation, gradient, and related events.
7. Replay a checkpoint or open a report.

## Ops v2 First Screen

Ops v2 should introduce a training loop strip above the graph:

```text
Data -> Forward -> Loss -> Backward -> Optimizer -> Checkpoint -> Eval
```

Each stage has:

- label;
- short state;
- evidence count or latest metric;
- active, healthy, warning, or idle visual state.

Each stage is also an expandable trigger. The collapsed row stays compact; clicking a stage opens a stage detail panel above the graph, similar in spirit to the bottom telemetry dock but smaller and scoped to one training-loop stage.

In the first implementation, most stages can be derived from existing events:

- Data: resource loaded or source imported.
- Forward: graph exists and forward/replay can run.
- Loss: latest metric loss exists.
- Backward: layer snapshots with gradient fields exist.
- Optimizer: training config has learning rate.
- Checkpoint: checkpoint event exists.
- Eval: report or prediction exists.

## Operator Node Shape

Each node should show:

- `kind`;
- `label` or `id`;
- compact shape line;
- parameter count when known;
- health badge.

Example:

```text
Linear
net.3
[16,128] -> [16,10]
1.3k params
grad stable
```

## Layer Inspector

Clicking a node should open a focused inspector in the graph area rather than only selecting the node.

Inspector sections:

- Identity: id, kind, confidence.
- Tensor shape: input and output.
- Parameters: count and metadata.
- Latest health: activation mean, activation sparsity, gradient norm, weight std.
- Trend: small history view for the selected layer.
- Evidence: events for this layer.

## Interaction Rules

- The graph remains the primary object.
- The left control rail is only for controls and run state. It must not host layer details.
- The layer inspector opens as a graph-area drawer so the selected node and its evidence stay spatially connected.
- The bottom dock owns time-series and event detail.
- Deep reports remain in Run Detail until the Ops view has enough live evidence.

## Future IA

After Ops v2, add a real time scrubber. The scrubber should become the control that synchronizes graph state, metrics, events, checkpoints, and reports.
