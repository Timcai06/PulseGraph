import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import type { TrainingLoopStage } from "../lib/trainingLoop";

type Props = {
  stages: TrainingLoopStage[];
};

export function TrainingLoopStrip({ stages }: Props) {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [activeStageId, setActiveStageId] = useState<TrainingLoopStage["id"] | undefined>(
    stages.find((stage) => stage.id === "forward")?.id ?? stages[0]?.id
  );
  const activeStage = stages.find((stage) => stage.id === activeStageId) ?? stages[0];

  return (
    <section className={`top-training-drawer ${drawerOpen ? "open" : ""}`} aria-label="training loop drawer">
      <button
        aria-controls="training-loop-drawer-body"
        aria-expanded={drawerOpen}
        className="training-loop-handle"
        onClick={() => setDrawerOpen((current) => !current)}
        type="button"
      >
        <i className={`training-loop-status-dot ${activeStage?.state ?? "idle"}`} aria-hidden="true" />
        <span>Training Loop</span>
        <strong>{activeStage ? `${activeStage.label}: ${activeStage.detail}` : "waiting for telemetry"}</strong>
        {drawerOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </button>

      <div className="training-loop-drawer-body" id="training-loop-drawer-body" aria-hidden={!drawerOpen}>
        <div className="loop-stage-row" role="list" aria-label="training loop stages">
          {stages.map((stage) => (
            <button
              aria-pressed={activeStageId === stage.id}
              className={`loop-stage ${stage.state} ${activeStageId === stage.id ? "open" : ""}`}
              key={stage.id}
              onClick={() => {
                setActiveStageId(stage.id);
                setDrawerOpen(true);
              }}
              type="button"
            >
              <span>{stage.label}</span>
              <strong>{stage.detail}</strong>
            </button>
          ))}
        </div>

        {activeStage && (
          <section className={`stage-drawer-panel ${activeStage.state}`} aria-live="polite">
            <header>
              <span>{activeStage.label}</span>
              <strong>{activeStage.detail}</strong>
            </header>
            <p>{stageDetailCopy[activeStage.id]}</p>
          </section>
        )}
      </div>
    </section>
  );
}

const stageDetailCopy: Record<TrainingLoopStage["id"], string> = {
  data: "Shows whether a training resource is loaded and ready to supply batches.",
  forward: "Shows whether PulseGraph has an executable operator graph for the current model.",
  loss: "Tracks the latest loss evidence emitted by the training stream.",
  backward: "Uses layer snapshots as the first signal that gradient-side telemetry exists.",
  optimizer: "Shows optimizer evidence such as learning rate when it is available.",
  checkpoint: "Shows whether the run has produced checkpoint evidence for replay or evaluation.",
  eval: "Shows whether prediction, replay, or completed-run evidence is available."
};
