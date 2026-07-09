import { useRef, useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import type { TrainingLoopStage } from "../lib/trainingLoop";
import { useReducedMotion } from "../hooks/useReducedMotion";
import { motionDuration, motionDurations, motionEase, motionStagger } from "../lib/motion";

gsap.registerPlugin(useGSAP);

const TOP_DRAWER_HANDLE_PX = 42;

type Props = {
  stages: TrainingLoopStage[];
};

export function TrainingLoopStrip({ stages }: Props) {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [activeStageId, setActiveStageId] = useState<TrainingLoopStage["id"] | undefined>(
    stages.find((stage) => stage.id === "forward")?.id ?? stages[0]?.id
  );
  const activeStage = stages.find((stage) => stage.id === activeStageId) ?? stages[0];
  const drawerRef = useRef<HTMLElement | null>(null);
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const reducedMotion = useReducedMotion();

  useGSAP(
    () => {
      const drawer = drawerRef.current;
      const body = bodyRef.current;
      if (!drawer || !body) return;

      const drawerTop = drawer.getBoundingClientRect().top;
      const contentHeight = TOP_DRAWER_HANDLE_PX + body.scrollHeight;
      const maxExpandedHeight = Math.max(TOP_DRAWER_HANDLE_PX, window.innerHeight - drawerTop - 56);
      const expandedHeight = Math.min(contentHeight, maxExpandedHeight);
      const bodyMaxHeight = Math.max(0, expandedHeight - TOP_DRAWER_HANDLE_PX);
      const drawerHeight = drawerOpen ? expandedHeight : TOP_DRAWER_HANDLE_PX;
      const stageTargets = body.querySelectorAll(".loop-stage, .stage-drawer-panel");
      gsap.killTweensOf([drawer, body, stageTargets]);

      if (reducedMotion) {
        gsap.set(drawer, { height: drawerHeight });
        gsap.set(body, {
          autoAlpha: drawerOpen ? 1 : 0,
          maxHeight: drawerOpen ? bodyMaxHeight : 0,
          overflowY: drawerOpen && contentHeight > maxExpandedHeight ? "auto" : "visible",
          y: 0,
          pointerEvents: drawerOpen ? "auto" : "none"
        });
        gsap.set(stageTargets, { autoAlpha: 1, y: 0 });
        return;
      }

      const timeline = gsap.timeline({ defaults: { ease: motionEase.panel, overwrite: "auto" } });

      if (drawerOpen) {
        timeline
          .to(drawer, { height: expandedHeight, duration: motionDuration("drawer", reducedMotion) }, 0)
          .fromTo(
            body,
            { autoAlpha: 0, maxHeight: bodyMaxHeight, overflowY: "hidden", y: -8, pointerEvents: "none" },
            {
              autoAlpha: 1,
              maxHeight: bodyMaxHeight,
              overflowY: contentHeight > maxExpandedHeight ? "auto" : "visible",
              y: 0,
              pointerEvents: "auto",
              duration: motionDurations.panel,
              ease: motionEase.standard
            },
            0.1
          )
          .fromTo(
            stageTargets,
            { autoAlpha: 0, y: -6 },
            { autoAlpha: 1, y: 0, duration: motionDurations.quick, ease: motionEase.standard, stagger: motionStagger.compact },
            0.18
          );
      } else {
        timeline
          .to(
            body,
            {
              autoAlpha: 0,
              maxHeight: 0,
              overflowY: "hidden",
              y: -8,
              pointerEvents: "none",
              duration: motionDurations.quick,
              ease: motionEase.standard
            },
            0
          )
          .to(drawer, { height: TOP_DRAWER_HANDLE_PX, duration: motionDuration("drawer", reducedMotion) }, 0);
      }
    },
    { dependencies: [drawerOpen, activeStageId, activeStage?.detail, reducedMotion], scope: drawerRef }
  );

  return (
    <section
      className={`top-training-drawer ${drawerOpen ? "open" : ""}`}
      aria-label="training loop drawer"
      ref={drawerRef}
    >
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

      <div className="training-loop-drawer-body" id="training-loop-drawer-body" aria-hidden={!drawerOpen} ref={bodyRef}>
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
