import { useId, useRef, useState } from "react";
import { Activity, ChevronDown, LocateFixed, Radio, SkipBack, SkipForward } from "lucide-react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import { useReducedMotion } from "../hooks/useReducedMotion";
import { motionDuration, motionEase } from "../lib/motion";
import type { CausalFocus, TimelineFrame } from "../lib/timeline";

gsap.registerPlugin(useGSAP);

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
  const [open, setOpen] = useState(false);
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const bodyId = useId();
  const reducedMotion = useReducedMotion();
  const index = stepIndex(frames, selectedStep);
  const frame = frames[index];
  const canScrub = frames.length > 1;
  const peakStep = focus.jumpStep;

  const shift = (delta: number) => {
    if (!frames.length) return;
    const next = Math.max(0, Math.min(frames.length - 1, index + delta));
    onStepChange(frames[next].step);
  };

  useGSAP(
    () => {
      const body = bodyRef.current;
      if (!body) return;
      body.inert = !open;
      if (reducedMotion) {
        gsap.set(body, { height: open ? "auto" : 0, opacity: open ? 1 : 0 });
        return;
      }
      if (open) {
        gsap.set(body, { height: "auto", opacity: 1 });
        const height = body.offsetHeight;
        gsap.fromTo(
          body,
          { height: 0, opacity: 0 },
          {
            height,
            opacity: 1,
            duration: motionDuration("drawer", reducedMotion),
            ease: motionEase.panel,
            onComplete: () => gsap.set(body, { height: "auto" })
          }
        );
      } else {
        gsap.to(body, {
          height: 0,
          opacity: 0,
          duration: motionDuration("drawer", reducedMotion),
          ease: motionEase.panel
        });
      }
    },
    { dependencies: [open, reducedMotion], scope: bodyRef }
  );

  return (
    <section className={`timeline-scrubber focus-${focus.severity}`}>
      <button
        className="timeline-summary"
        type="button"
        aria-controls={bodyId}
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <div className="timeline-state">
          {live ? <Radio size={15} /> : <Activity size={15} />}
          <span>{live ? "Live" : `Step ${frame?.step ?? 0}`}</span>
        </div>
        <span className="timeline-summary-focus">{focus.title}</span>
        <span className="timeline-frame-count">{frames.length ? `${frames.length} frames` : "No frames"}</span>
        <ChevronDown className={`timeline-caret ${open ? "open" : ""}`} size={14} />
      </button>
      <div className="timeline-body-shell" id={bodyId} ref={bodyRef} aria-hidden={!open}>
        <div className="timeline-body">
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
        </div>
      </div>
    </section>
  );
}
