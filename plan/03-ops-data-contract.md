# Ops Data Contract

## Existing Data We Can Use Now

### Graph Node

Source: `frontend/src/api/types.ts`

Fields:

- `id`
- `label`
- `kind`
- `input_shape`
- `output_shape`
- `param_count`
- `confidence`
- `metadata`

Use in Ops:

- node title and subtitle;
- compact shape display;
- parameter count display;
- graph confidence badge.

### Layer Snapshot

Fields:

- `layer_id`
- `input_shape`
- `output_shape`
- `activation_mean`
- `activation_sparsity`
- `gradient_norm`
- `weight_std`

Use in Ops:

- node health badge;
- layer inspector latest health;
- selected-layer trend.

### Run Events

Existing event types:

- `metric`
- `layer_snapshot`
- `infra`
- `checkpoint`
- `animation`
- `graph`
- `source_registered`
- `config_registered`
- `graph_registered`
- `run_complete`

Use in Ops:

- training loop stage states;
- bottom event list;
- selected-layer evidence;
- checkpoint state.

### Stream State

Source: `frontend/src/hooks/useRunStream.ts`

Existing state:

- `status`
- `runId`
- `metrics`
- `events`
- `layerSnapshots`
- `layerHistory`
- `graph`
- `pulsedNodeId`
- `device`

Use in Ops:

- current stage;
- graph overlay;
- inspector data;
- telemetry dock.

## Derived Health Rules For Ops v2

These rules are intentionally simple and local. They can be replaced by backend report analysis later.

### Activation Health

- `activation_sparsity >= 0.95`: warning, possible dead layer.
- `activation_sparsity >= 0.80`: caution.
- otherwise healthy.

### Gradient Health

- `gradient_norm == null`: unknown.
- `gradient_norm <= 1e-7`: warning, possible vanishing gradient.
- `gradient_norm >= 100`: warning, possible exploding gradient.
- otherwise healthy.

### Confidence Health

- `trusted`: strong.
- `safe`: safe but incomplete.
- `inferred`: inferred, show caution copy.

## Future Telemetry Needed

Ops v3 and later should add:

- per-layer latency;
- per-layer memory allocation;
- backward pass graph or gradient edge events;
- optimizer update magnitude;
- data loader wait time;
- batch class distribution;
- GPU utilization;
- VRAM usage over time;
- distributed rank and worker state;
- cost per step and cost per accuracy point.

## Boundary

Do not pretend safe checkpoint inspection can reconstruct a full runtime graph. Full runtime views require trusted model execution, hooks, or an explicit traced graph.

