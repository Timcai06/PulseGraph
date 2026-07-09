import { useState } from "react";
import type { TrainingLoopStage } from "../lib/trainingLoop";

type Props = {
  stages: TrainingLoopStage[];
};

export function TrainingLoopStrip({ stages }: Props) {
  const [activeStageId, setActiveStageId] = useState<TrainingLoopStage["id"] | undefined>();
  const activeStage = stages.find((stage) => stage.id === activeStageId);

  return (
    <section className="training-loop-strip" aria-label="training loop">
      <div className="loop-stage-row">
        {stages.map((stage) => (
          <button
            aria-expanded={activeStageId === stage.id}
            className={`loop-stage ${stage.state} ${activeStageId === stage.id ? "open" : ""}`}
            key={stage.id}
            onClick={() => setActiveStageId((current) => (current === stage.id ? undefined : stage.id))}
            type="button"
          >
            <span>{stage.label}</span>
            <strong>{stage.detail}</strong>
          </button>
        ))}
      </div>
      {activeStage && (
        <section className={`stage-detail-panel ${activeStage.state}`}>
          <header>
            <span>{activeStage.label}</span>
            <button onClick={() => setActiveStageId(undefined)} type="button" aria-label={`Collapse ${activeStage.label}`}>
              Close
            </button>
          </header>
          <strong>{activeStage.detail}</strong>
          <p>{stageDetailCopy[activeStage.id]}</p>
        </section>
      )}
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
