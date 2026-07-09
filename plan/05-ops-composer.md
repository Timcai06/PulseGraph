# Ops Composer

## Goal

Ops Composer turns the visible operator ports into a deliberate model-structure workbench.

The product should feel like model training can be explored through modules and tensor ports, but it must not pretend that dragging a line is already safe code generation. The first version is a ghost design layer: users can propose structure changes, inspect compatibility, and understand likely consequences without mutating the real run.

## Product Position

PulseGraph should have two graph modes:

- **Monitor:** trusted view of the current run, stream, telemetry, checkpoints, and causal focus.
- **Compose:** speculative view for model-structure experiments using the same graph and telemetry evidence.

Monitor answers: what happened in the real run?

Compose answers: what might happen if the structure changed?

## Why This Matters

The React Flow handle points surfaced an important product idea: model training can be made tactile. Ports can make tensor flow visible and composable.

The danger is overpromising. A neural network graph is not only boxes and lines; every connection has shape rules, module semantics, parameter ownership, initialization behavior, optimizer state, and training dynamics. Composer must expose these constraints instead of hiding them.

## Phase 1: Port Semantics

Turn each node handle into a named port.

### Input Port

The left port represents the tensor entering the operator.

Hover or click should show:

- node id and kind;
- expected input shape;
- known upstream source;
- graph confidence;
- latest activation evidence when available.

### Output Port

The right port represents the tensor leaving the operator.

Hover or click should show:

- output shape;
- downstream consumers;
- parameter count context;
- latest activation sparsity;
- latest gradient evidence when available.

### Rule

Ports are visible in Monitor mode but do not create real edges. They are inspection anchors first.

## Phase 2: Ghost Compose

Add a Composer toggle beside the graph view tabs:

```text
Ops | Neurons | Composer
```

In Composer mode, dragging from an output port to an input port creates a ghost edge.

Ghost edges are local UI artifacts. They do not change:

- backend graph;
- saved run history;
- training resource source files;
- checkpoint metadata;
- report data.

## Compatibility Checks

Every ghost edge should produce a structured result.

### Shape

- Exact output/input shape match: compatible.
- Same rank with batch-compatible dimensions: caution.
- Unknown shape on either side: unknown.
- Rank or terminal dimension mismatch: incompatible.

### Direction

- Output to input: valid direction.
- Input to input or output to output: invalid.
- Edge that creates a cycle in a feed-forward graph: invalid unless a future recurrent mode exists.

### Operator Semantics

The first implementation can use simple local rules:

- Linear after convolution flatten mismatch: caution unless flatten exists.
- Add or merge node requires multiple compatible inputs.
- Loss nodes must consume prediction and target-like tensors.
- Optimizer and checkpoint are training-loop stages, not forward operator connections.

## UI Behavior

### Monitor Mode

- Ports are visible but passive.
- Hover opens a compact port tooltip.
- Clicking a node still opens the Layer Inspector.
- Runtime pulses and scrubber replay stay unchanged.

### Composer Mode

- Ports become active drag anchors.
- Dragging shows a provisional line.
- Dropping on a compatible port creates a ghost edge.
- Dropping on an invalid port shows a short reason.
- Ghost edges have a distinct style and can be removed.

### Bottom Dock

The telemetry dock should gain a Composer detail state when a ghost edge is selected:

- proposed connection;
- compatibility status;
- shape explanation;
- affected downstream nodes;
- available health evidence;
- next suggested operator if the connection is incompatible.

## Data Model

Add a frontend-only structure:

```ts
type GraphPort = {
  id: string;
  nodeId: string;
  direction: "input" | "output";
  shape?: number[] | null;
  tensorName?: string;
};

type GhostEdge = {
  id: string;
  sourcePortId: string;
  targetPortId: string;
  status: "compatible" | "caution" | "incompatible" | "unknown";
  reasons: string[];
};
```

Derive ports from existing `GraphNode` fields:

- input port shape from `input_shape`;
- output port shape from `output_shape`;
- runtime evidence from `layerSnapshots`;
- downstream consumers from `ModelGraph.edges`.

## Implementation Boundary

Do not generate Python code in this phase.

Do not mutate real model graphs in the backend.

Do not let ghost edges appear in reports or run history.

Do not imply that a compatible shape guarantees good training behavior.

## Testing Strategy

Add pure helper tests first:

- derive input/output ports from graph nodes;
- detect exact shape compatibility;
- detect caution for unknown or batch-compatible shapes;
- detect invalid direction;
- detect simple cycles;
- keep ghost edges frontend-only.

Add source-level UI regression tests:

- Composer mode exists separately from Ops and Neurons.
- React Flow handles are intentionally named as ports.
- Ghost edges use a distinct class.
- Monitor mode does not create real connections.

## First Implementation Slice

1. Create `graphPorts.ts` for port derivation and compatibility.
2. Add tests for port and ghost-edge rules.
3. Update `ModelGraphPanel` to expose passive named ports in Monitor mode.
4. Add Composer view mode.
5. Enable ghost edge creation and deletion in Composer mode only.
6. Show selected ghost-edge explanation in the bottom dock.
7. Update plan docs with actual implementation status after verification.

## Success Criteria

Composer is successful when a user can:

1. see every node as an operator with input/output tensor ports;
2. understand what each port carries;
3. drag a speculative connection without corrupting the real graph;
4. get a concrete compatibility explanation;
5. return to Monitor mode with runtime telemetry untouched.
