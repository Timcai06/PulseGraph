# Ops v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first Ops v2 cockpit: training loop strip, richer operator node health, and a selected-layer inspector driven by existing telemetry.

**Architecture:** Keep `App.tsx` as the state owner for now. Add small frontend-only helpers for layer health and loop stage derivation, then pass derived state into focused components. Avoid backend changes in this phase because existing `RunEvent`, `GraphNode`, `layerSnapshots`, and `layerHistory` already cover the first useful slice.

**Tech Stack:** React, TypeScript, GSAP, React Flow, Vitest, existing PulseGraph SSE stream state.

---

## UX Amendment

The implementation was adjusted after visual review:

- The training loop strip is no longer a passive status row. Each stage is an expandable trigger with a focused stage detail panel.
- Layer details do not live in the left control rail. The control rail remains for resource/run actions, while selected-node details open in a graph-area drawer.
- Dark and light themes both require explicit styling for stage panels and layer detail surfaces.

## File Map

- Create: `frontend/src/lib/layerHealth.ts`
  - Derive compact layer health labels and severity from `GraphNode` and `LayerSnapshot`.
- Create: `frontend/src/lib/trainingLoop.ts`
  - Derive stage states for Data, Forward, Loss, Backward, Optimizer, Checkpoint, and Eval.
- Create: `frontend/src/components/TrainingLoopStrip.tsx`
  - Render the expandable stage strip above the operator graph.
- Create: `frontend/src/components/LayerInspector.tsx`
  - Render selected node identity, shape, params, latest layer health, history, and related events in the graph-area drawer.
- Modify: `frontend/src/components/ModelGraphPanel.tsx`
  - Enrich node display with shape, params, confidence, and health badge.
- Modify: `frontend/src/components/ControlRail.tsx`
  - Keep the rail focused on controls and remove layer detail ownership.
- Modify: `frontend/src/App.tsx`
  - Pass stream layer snapshots/history/events into Ops components.
- Modify: `frontend/src/App.test.ts`
  - Add source-level regression tests for Ops v2 surfaces.
- Modify: `frontend/src/styles/modules/graph.css`
  - Style richer graph nodes and loop strip placement.
- Modify: `frontend/src/styles/modules/controls.css`
  - Keep control rail styles limited to controls and run state.

## Task 1: Add Layer Health Derivation

**Files:**

- Create: `frontend/src/lib/layerHealth.ts`
- Modify: `frontend/src/App.test.ts`

- [x] **Step 1: Write the failing test**

Add source-level assertions to `frontend/src/App.test.ts`:

```ts
import layerHealthSource from "./lib/layerHealth.ts?raw";

it("derives operator health from layer snapshots", () => {
  expect(layerHealthSource).toContain("deriveLayerHealth");
  expect(layerHealthSource).toContain("activation_sparsity");
  expect(layerHealthSource).toContain("gradient_norm");
  expect(layerHealthSource).toContain("possible dead layer");
  expect(layerHealthSource).toContain("possible vanishing gradient");
});
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
cd frontend && npm test -- src/App.test.ts --run
```

Expected: FAIL because `frontend/src/lib/layerHealth.ts` does not exist yet.

- [x] **Step 3: Implement `layerHealth.ts`**

Create `frontend/src/lib/layerHealth.ts`:

```ts
import type { GraphNode, LayerSnapshot } from "../api/client";

export type HealthSeverity = "healthy" | "caution" | "warning" | "unknown";

export type LayerHealthSummary = {
  severity: HealthSeverity;
  label: string;
  detail: string;
};

function formatShape(shape?: number[] | null) {
  return shape?.length ? `[${shape.join("x")}]` : "?";
}

export function formatNodeShape(node: GraphNode, snapshot?: LayerSnapshot) {
  const input = snapshot?.input_shape ?? node.input_shape;
  const output = snapshot?.output_shape ?? node.output_shape;
  return `${formatShape(input)} -> ${formatShape(output)}`;
}

export function formatParamCount(value: number) {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return String(value);
}

export function deriveLayerHealth(node: GraphNode, snapshot?: LayerSnapshot): LayerHealthSummary {
  if (!snapshot) {
    return {
      severity: node.confidence === "inferred" ? "caution" : "unknown",
      label: node.confidence,
      detail: `Graph confidence is ${node.confidence}; no live snapshot recorded.`
    };
  }

  if (snapshot.activation_sparsity != null && snapshot.activation_sparsity >= 0.95) {
    return {
      severity: "warning",
      label: "sparse activation",
      detail: "Activation sparsity is high; possible dead layer."
    };
  }

  if (snapshot.gradient_norm != null && snapshot.gradient_norm <= 1e-7) {
    return {
      severity: "warning",
      label: "low gradient",
      detail: "Gradient norm is near zero; possible vanishing gradient."
    };
  }

  if (snapshot.gradient_norm != null && snapshot.gradient_norm >= 100) {
    return {
      severity: "warning",
      label: "high gradient",
      detail: "Gradient norm is very high; possible exploding gradient."
    };
  }

  if (snapshot.activation_sparsity != null && snapshot.activation_sparsity >= 0.8) {
    return {
      severity: "caution",
      label: "activation caution",
      detail: "Activation sparsity is elevated."
    };
  }

  return {
    severity: "healthy",
    label: "healthy",
    detail: "Latest layer snapshot is within the local Ops thresholds."
  };
}
```

- [x] **Step 4: Run test to verify it passes**

Run:

```bash
cd frontend && npm test -- src/App.test.ts --run
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add frontend/src/App.test.ts frontend/src/lib/layerHealth.ts
git commit -m "Add layer health derivation"
```

## Task 2: Enrich Operator Nodes

**Files:**

- Modify: `frontend/src/components/ModelGraphPanel.tsx`
- Modify: `frontend/src/styles/modules/graph.css`
- Modify: `frontend/src/App.test.ts`

- [x] **Step 1: Write the failing test**

Add assertions:

```ts
it("shows richer Ops node health metadata", () => {
  expect(modelGraphSource).toContain("layerSnapshots");
  expect(modelGraphSource).toContain("deriveLayerHealth");
  expect(modelGraphSource).toContain("formatNodeShape");
  expect(modelGraphSource).toContain("node-health");
  expect(modelGraphSource).toContain("node-shape");
});
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
cd frontend && npm test -- src/App.test.ts --run
```

Expected: FAIL because `ModelGraphPanel` does not yet accept `layerSnapshots` or render health metadata.

- [x] **Step 3: Modify `ModelGraphPanel` props and node render**

Add `layerSnapshots` to props and node data:

```ts
import { deriveLayerHealth, formatNodeShape, formatParamCount } from "../lib/layerHealth";
import type { LayerSnapshot } from "../api/client";

type PulseNodeData = GraphNode & {
  snapshot?: LayerSnapshot;
};
```

Update node creation:

```ts
const nodes = useMemo<Node<PulseNodeData>[]>(() => {
  return visibleGraph.nodes.map((node) => ({
    id: node.id,
    type: "pulse",
    position: positions[node.id] ?? { x: 0, y: 0 },
    data: { ...node, snapshot: layerSnapshots[node.id] },
    selected: node.id === selectedNodeId
  }));
}, [visibleGraph.nodes, positions, selectedNodeId, layerSnapshots]);
```

Update `PulseNode`:

```tsx
const PulseNode = memo(({ data, selected }: NodeProps<Node<PulseNodeData>>) => {
  const health = deriveLayerHealth(data, data.snapshot);
  return (
    <div className={`model-node ${selected ? "selected" : ""} health-${health.severity}`} data-layer-id={data.id}>
      <Handle type="target" position={Position.Left} />
      <div className="node-kind">{data.kind}</div>
      <div className="node-label">{data.label || data.id}</div>
      <div className="node-shape">{formatNodeShape(data, data.snapshot)}</div>
      <div className="node-footer">
        <span>{formatParamCount(data.param_count)} params</span>
        <span className="node-health">{health.label}</span>
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
});
```

- [x] **Step 4: Add graph node styles**

Add to `frontend/src/styles/modules/graph.css`:

```css
.node-shape {
  margin-top: 6px;
  color: var(--text-muted);
  font-family: var(--font-mono);
  font-size: 10px;
}

.node-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-top: 8px;
  color: var(--text-muted);
  font-size: 10px;
}

.node-health {
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 2px 6px;
  color: var(--text);
}

.model-node.health-warning {
  border-color: rgba(248, 113, 113, 0.75);
}

.model-node.health-caution {
  border-color: rgba(251, 191, 36, 0.75);
}

.model-node.health-healthy .node-health {
  border-color: rgba(52, 211, 153, 0.5);
}
```

- [x] **Step 5: Pass stream snapshots from `App.tsx`**

Update the `ModelGraphPanel` call:

```tsx
<ModelGraphPanel
  graph={graph}
  selectedNodeId={selectedNode?.id}
  pulsedNodeId={stream.pulsedNodeId}
  probabilities={prediction?.probabilities}
  forwardTick={forwardTick}
  layerSnapshots={stream.layerSnapshots}
  onSelect={setSelectedNode}
/>
```

- [x] **Step 6: Run focused tests**

Run:

```bash
cd frontend && npm test -- src/App.test.ts --run
```

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add frontend/src/App.test.ts frontend/src/App.tsx frontend/src/components/ModelGraphPanel.tsx frontend/src/styles/modules/graph.css
git commit -m "Show Ops node health metadata"
```

## Task 3: Add Training Loop Strip

**Files:**

- Create: `frontend/src/lib/trainingLoop.ts`
- Create: `frontend/src/components/TrainingLoopStrip.tsx`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles/modules/graph.css`
- Modify: `frontend/src/App.test.ts`

- [x] **Step 1: Write the failing test**

Add source imports and assertions:

```ts
import trainingLoopSource from "./lib/trainingLoop.ts?raw";
import trainingLoopStripSource from "./components/TrainingLoopStrip.tsx?raw";

it("adds a training loop strip to Ops", () => {
  expect(trainingLoopSource).toContain("deriveTrainingLoopStages");
  expect(trainingLoopSource).toContain("Data");
  expect(trainingLoopSource).toContain("Forward");
  expect(trainingLoopSource).toContain("Backward");
  expect(trainingLoopStripSource).toContain("training-loop-strip");
  expect(appSource).toContain("TrainingLoopStrip");
});
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
cd frontend && npm test -- src/App.test.ts --run
```

Expected: FAIL because the files do not exist.

- [x] **Step 3: Create `trainingLoop.ts`**

```ts
import type { MetricPoint } from "../hooks/useRunStream";
import type { RunEvent } from "../api/client";

export type TrainingLoopStage = {
  id: "data" | "forward" | "loss" | "backward" | "optimizer" | "checkpoint" | "eval";
  label: string;
  state: "idle" | "active" | "healthy" | "warning";
  detail: string;
};

type Input = {
  hasResource: boolean;
  hasGraph: boolean;
  hasPrediction: boolean;
  metrics: MetricPoint[];
  events: RunEvent[];
  learningRate?: number | null;
};

export function deriveTrainingLoopStages(input: Input): TrainingLoopStage[] {
  const latestMetric = input.metrics.at(-1);
  const hasLayerSnapshot = input.events.some((event) => event.type === "layer_snapshot");
  const hasCheckpoint = input.events.some((event) => event.type === "checkpoint");
  const hasRunComplete = input.events.some((event) => event.type === "run_complete");

  return [
    {
      id: "data",
      label: "Data",
      state: input.hasResource ? "healthy" : "idle",
      detail: input.hasResource ? "resource loaded" : "waiting for resource"
    },
    {
      id: "forward",
      label: "Forward",
      state: input.hasGraph ? "active" : "idle",
      detail: input.hasGraph ? "operator graph ready" : "no graph"
    },
    {
      id: "loss",
      label: "Loss",
      state: latestMetric?.loss == null ? "idle" : "active",
      detail: latestMetric?.loss == null ? "no loss yet" : `loss ${latestMetric.loss.toFixed(4)}`
    },
    {
      id: "backward",
      label: "Backward",
      state: hasLayerSnapshot ? "active" : "idle",
      detail: hasLayerSnapshot ? "layer snapshots flowing" : "no gradient evidence"
    },
    {
      id: "optimizer",
      label: "Optimizer",
      state: input.learningRate == null ? "idle" : "healthy",
      detail: input.learningRate == null ? "learning rate unknown" : `lr ${input.learningRate}`
    },
    {
      id: "checkpoint",
      label: "Checkpoint",
      state: hasCheckpoint ? "healthy" : "idle",
      detail: hasCheckpoint ? "checkpoint recorded" : "no checkpoint"
    },
    {
      id: "eval",
      label: "Eval",
      state: input.hasPrediction || hasRunComplete ? "healthy" : "idle",
      detail: input.hasPrediction ? "prediction ready" : hasRunComplete ? "run complete" : "waiting"
    }
  ];
}
```

- [x] **Step 4: Create `TrainingLoopStrip.tsx`**

```tsx
import type { TrainingLoopStage } from "../lib/trainingLoop";

type Props = {
  stages: TrainingLoopStage[];
};

export function TrainingLoopStrip({ stages }: Props) {
  return (
    <section className="training-loop-strip" aria-label="training loop">
      {stages.map((stage) => (
        <article className={`loop-stage ${stage.state}`} key={stage.id}>
          <span>{stage.label}</span>
          <strong>{stage.detail}</strong>
        </article>
      ))}
    </section>
  );
}
```

- [x] **Step 5: Render it in `App.tsx`**

Derive stages near `predictionSummary`:

```ts
const loopStages = useMemo(
  () =>
    deriveTrainingLoopStages({
      hasResource: Boolean(sourceRecipe),
      hasGraph: graph.nodes.length > 0,
      hasPrediction: Boolean(prediction),
      metrics: stream.metrics,
      events: stream.events,
      learningRate: typeof sourceRecipe?.summary?.metadata?.lr === "number" ? sourceRecipe.summary.metadata.lr : undefined
    }),
  [sourceRecipe, graph.nodes.length, prediction, stream.metrics, stream.events]
);
```

If `LoadedResourceSummary` does not expose metadata, pass `undefined` for `learningRate` in this phase.

Render above `ModelGraphPanel`:

```tsx
<TrainingLoopStrip stages={loopStages} />
```

- [x] **Step 6: Add styles**

Add to `frontend/src/styles/modules/graph.css`:

```css
.training-loop-strip {
  position: absolute;
  top: 18px;
  right: 18px;
  left: 18px;
  z-index: 6;
  display: grid;
  grid-template-columns: repeat(7, minmax(0, 1fr));
  gap: 6px;
}

.loop-stage {
  min-width: 0;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 7px 8px;
  background: rgba(15, 23, 42, 0.72);
}

.loop-stage span {
  display: block;
  color: var(--text-muted);
  font-size: 10px;
  text-transform: uppercase;
}

.loop-stage strong {
  display: block;
  overflow: hidden;
  color: var(--text);
  font-size: 11px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.loop-stage.active {
  border-color: rgba(56, 189, 248, 0.55);
}

.loop-stage.healthy {
  border-color: rgba(52, 211, 153, 0.45);
}

.loop-stage.warning {
  border-color: rgba(248, 113, 113, 0.55);
}
```

- [x] **Step 7: Run focused tests**

Run:

```bash
cd frontend && npm test -- src/App.test.ts --run
```

Expected: PASS.

- [x] **Step 8: Commit**

```bash
git add frontend/src/App.test.ts frontend/src/App.tsx frontend/src/components/TrainingLoopStrip.tsx frontend/src/lib/trainingLoop.ts frontend/src/styles/modules/graph.css
git commit -m "Add Ops training loop strip"
```

## Task 4: Add Layer Inspector

**Files:**

- Create: `frontend/src/components/LayerInspector.tsx`
- Modify: `frontend/src/components/ControlRail.tsx`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles/modules/controls.css`
- Modify: `frontend/src/App.test.ts`

- [x] **Step 1: Write the failing test**

Add:

```ts
import layerInspectorSource from "./components/LayerInspector.tsx?raw";

it("adds a selected layer inspector to Ops", () => {
  expect(layerInspectorSource).toContain("LayerInspector");
  expect(layerInspectorSource).toContain("activation_sparsity");
  expect(layerInspectorSource).toContain("gradient_norm");
  expect(controlRailSource).toContain("selectedNode");
  expect(controlRailSource).toContain("layer-inspector");
});
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
cd frontend && npm test -- src/App.test.ts --run
```

Expected: FAIL because `LayerInspector` does not exist.

- [x] **Step 3: Create `LayerInspector.tsx`**

```tsx
import type { GraphNode, LayerSnapshot, RunEvent } from "../api/client";
import type { LayerHistoryPoint } from "../hooks/useRunStream";
import { deriveLayerHealth, formatNodeShape, formatParamCount } from "../lib/layerHealth";

type Props = {
  node?: GraphNode;
  snapshot?: LayerSnapshot;
  history: LayerHistoryPoint[];
  events: RunEvent[];
};

function value(value?: number | null) {
  return value == null ? "n/a" : Number(value).toPrecision(4);
}

export function LayerInspector({ node, snapshot, history, events }: Props) {
  if (!node) {
    return (
      <section className="layer-inspector empty">
        <h2>Layer Inspector</h2>
        <p>Select an operator node to inspect runtime health.</p>
      </section>
    );
  }

  const health = deriveLayerHealth(node, snapshot);
  const relatedEvents = events.filter((event) => event.layer === node.id).slice(0, 5);

  return (
    <section className={`layer-inspector health-${health.severity}`}>
      <header>
        <span>Layer Inspector</span>
        <h2>{node.label || node.id}</h2>
        <p>{node.kind} · {node.confidence}</p>
      </header>
      <dl>
        <div>
          <dt>shape</dt>
          <dd>{formatNodeShape(node, snapshot)}</dd>
        </div>
        <div>
          <dt>params</dt>
          <dd>{formatParamCount(node.param_count)}</dd>
        </div>
        <div>
          <dt>activation sparsity</dt>
          <dd>{value(snapshot?.activation_sparsity)}</dd>
        </div>
        <div>
          <dt>gradient norm</dt>
          <dd>{value(snapshot?.gradient_norm)}</dd>
        </div>
        <div>
          <dt>weight std</dt>
          <dd>{value(snapshot?.weight_std)}</dd>
        </div>
      </dl>
      <div className="layer-health-note">
        <strong>{health.label}</strong>
        <span>{health.detail}</span>
      </div>
      <div className="layer-history-note">
        {history.length ? `${history.length} snapshots recorded` : "No history for this layer yet"}
      </div>
      <div className="layer-events">
        {relatedEvents.length ? (
          relatedEvents.map((event) => (
            <span key={event.event_id}>{event.type} · step {event.step}</span>
          ))
        ) : (
          <span>No related events</span>
        )}
      </div>
    </section>
  );
}
```

- [x] **Step 4: Wire inspector through `ControlRail`**

Add props to `ControlRail`:

```ts
selectedNode?: GraphNode;
selectedSnapshot?: LayerSnapshot;
selectedHistory: LayerHistoryPoint[];
selectedEvents: RunEvent[];
```

Render after the session card:

```tsx
<LayerInspector
  node={selectedNode}
  snapshot={selectedSnapshot}
  history={selectedHistory}
  events={selectedEvents}
/>
```

- [x] **Step 5: Pass data from `App.tsx`**

Compute:

```ts
const selectedLayerHistory = selectedNode ? stream.layerHistory[selectedNode.id] ?? [] : [];
const selectedLayerEvents = selectedNode ? stream.events.filter((event) => event.layer === selectedNode.id) : [];
```

Pass:

```tsx
selectedNode={selectedNode}
selectedSnapshot={selectedNode ? stream.layerSnapshots[selectedNode.id] : undefined}
selectedHistory={selectedLayerHistory}
selectedEvents={selectedLayerEvents}
```

- [x] **Step 6: Add styles**

Add to `frontend/src/styles/modules/controls.css`:

```css
.layer-inspector {
  border: 1px solid var(--glass-border);
  border-radius: var(--radius-md);
  padding: 12px;
  background: rgba(15, 23, 42, 0.42);
}

.layer-inspector header span {
  color: var(--text-muted);
  font-size: 10px;
  text-transform: uppercase;
}

.layer-inspector h2 {
  margin: 4px 0;
  font-size: 14px;
}

.layer-inspector p {
  margin: 0;
  color: var(--text-muted);
  font-size: 11px;
}

.layer-inspector dl {
  display: grid;
  gap: 7px;
  margin: 12px 0;
}

.layer-inspector dl div {
  display: flex;
  justify-content: space-between;
  gap: 10px;
}

.layer-inspector dt {
  color: var(--text-muted);
  font-size: 10px;
  text-transform: uppercase;
}

.layer-inspector dd {
  margin: 0;
  font-family: var(--font-mono);
  font-size: 11px;
}

.layer-health-note,
.layer-history-note,
.layer-events {
  display: grid;
  gap: 5px;
  color: var(--text-muted);
  font-size: 11px;
}

.layer-health-note strong {
  color: var(--text);
}
```

- [x] **Step 7: Run focused tests**

Run:

```bash
cd frontend && npm test -- src/App.test.ts --run
```

Expected: PASS.

- [x] **Step 8: Commit**

```bash
git add frontend/src/App.test.ts frontend/src/App.tsx frontend/src/components/ControlRail.tsx frontend/src/components/LayerInspector.tsx frontend/src/styles/modules/controls.css
git commit -m "Add Ops layer inspector"
```

## Task 5: Verify Ops v2 In Browser

**Files:**

- Modify only if verification finds defects.

- [x] **Step 1: Run full tests**

Run:

```bash
make test
```

Expected: backend tests pass and frontend tests pass.

- [x] **Step 2: Run production build**

Run:

```bash
cd frontend && npm run build
```

Expected: build succeeds. Existing Vite chunk-size warning is acceptable.

- [x] **Step 3: Start local services**

Run backend:

```bash
env BACKEND_PORT=8011 make backend
```

Run frontend:

```bash
cd frontend && env PULSEGRAPH_API_URL=http://127.0.0.1:8011 npm run dev -- --port 5174
```

- [x] **Step 4: Verify with Playwright**

Open:

```bash
/Users/tim/.codex/skills/playwright/scripts/playwright_cli.sh open http://127.0.0.1:5174
```

Check:

- Training loop strip is visible.
- Operator nodes show shape, params, and health badge.
- Selecting a node updates Layer Inspector.
- Training still runs.
- Console has no errors.

- [x] **Step 5: Capture screenshots**

Save ignored screenshots:

```bash
mkdir -p output/playwright
/Users/tim/.codex/skills/playwright/scripts/playwright_cli.sh screenshot
cp "$(ls -t .playwright-cli/page-*.png | head -1)" output/playwright/ops-v2-cockpit.png
```

- [x] **Step 6: Commit final fixes**

If browser verification required changes:

```bash
git add frontend/src
git commit -m "Polish Ops v2 cockpit"
```

If no changes were needed, do not create an empty commit.
