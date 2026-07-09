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
