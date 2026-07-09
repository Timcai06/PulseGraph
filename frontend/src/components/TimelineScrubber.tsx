import { Activity, LocateFixed, Radio, SkipBack, SkipForward } from "lucide-react";
import type { CausalFocus, TimelineFrame } from "../lib/timeline";

type Props = {
  frames: TimelineFrame[];
  selectedStep?: number;
  live: boolean;
  focus: CausalFocus;
  onStepChange: (step: number) => void;
  onLive: () => void;
  onJumpToStep: (step: number) => void;
};

function stepIndex(frames: TimelineFrame[], selectedStep?: number) {
  if (!frames.length) return 0;
  if (selectedStep == null) return frames.length - 1;
  const exact = frames.findIndex((frame) => frame.step === selectedStep);
  return exact >= 0 ? exact : frames.length - 1;
}

export function TimelineScrubber({ frames, selectedStep, live, focus, onStepChange, onLive, onJumpToStep }: Props) {
  const index = stepIndex(frames, selectedStep);
  const frame = frames[index];
  const canScrub = frames.length > 1;
  const peakStep = focus.jumpStep;

  const shift = (delta: number) => {
    if (!frames.length) return;
    const next = Math.max(0, Math.min(frames.length - 1, index + delta));
    onStepChange(frames[next].step);
  };

  return (
    <section className={`timeline-scrubber focus-${focus.severity}`}>
      <div className="timeline-head">
        <div className="timeline-state">
          {live ? <Radio size={15} /> : <Activity size={15} />}
          <span>{live ? "Live" : `Step ${frame?.step ?? 0}`}</span>
        </div>
        <div className="timeline-actions">
          {peakStep != null && (
            <button type="button" onClick={() => onJumpToStep(peakStep)} aria-label="Jump to loss peak">
              <LocateFixed size={14} />
              Peak
            </button>
          )}
          <button className={live ? "active" : ""} type="button" onClick={onLive} aria-label="Return to live telemetry">
            <Radio size={14} />
            Live
          </button>
        </div>
      </div>
      <div className="timeline-control">
        <button type="button" onClick={() => shift(-1)} disabled={!canScrub || index === 0} aria-label="Previous frame">
          <SkipBack size={14} />
        </button>
        <input
          type="range"
          min={0}
          max={Math.max(0, frames.length - 1)}
          step={1}
          value={index}
          disabled={!frames.length}
          onChange={(event) => onStepChange(frames[Number(event.currentTarget.value)]?.step ?? 0)}
          aria-label="Telemetry timeline"
        />
        <button
          type="button"
          onClick={() => shift(1)}
          disabled={!canScrub || index === frames.length - 1}
          aria-label="Next frame"
        >
          <SkipForward size={14} />
        </button>
      </div>
      <div className="causal-focus">
        <strong>{focus.title}</strong>
        <span>{focus.detail}</span>
      </div>
    </section>
  );
}
