import { useEffect, useRef, useState } from "react";
import { ArrowRight, ChevronDown, ChevronUp, GitBranch, RefreshCw } from "lucide-react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import type { TrainingLoopModel, TrainingLoopStage } from "../lib/trainingLoop";
import { useReducedMotion } from "../hooks/useReducedMotion";
import { motionDuration, motionDurations, motionEase, motionStagger } from "../lib/motion";

gsap.registerPlugin(useGSAP);

const TOP_DRAWER_HANDLE_PX = 42;

type Props = {
  model: TrainingLoopModel;
  onStageSelect?: (stage: TrainingLoopStage) => void;
};

function stageProgress(model: TrainingLoopModel) {
  if (model.totalSteps) return `Step ${model.currentStep}/${model.totalSteps}`;
  return model.currentStep ? `Step ${model.currentStep}` : "No active step";
}

function evidenceLabel(stage: TrainingLoopStage) {
  const labels = {
    prepare: "Open Prepare",
    ops: "Open Ops",
    telemetry: "Open Telemetry",
    diagnostics: "Open Diagnostics",
    checkpoint: "Open Checkpoints",
    evaluate: "Open Evaluate"
  };
  return stage.evidence ? labels[stage.evidence] : undefined;
}

function StageButton({
  stage,
  selected,
  onSelect
}: {
  stage: TrainingLoopStage;
  selected: boolean;
  onSelect: (stage: TrainingLoopStage) => void;
}) {
  return (
    <button
      aria-pressed={selected}
      className={`loop-stage ${stage.state} ${selected ? "open" : ""}`}
      onClick={() => onSelect(stage)}
      type="button"
    >
      <i aria-hidden="true" />
      <span>{stage.label}</span>
      {stage.durationMs != null ? <em>{stage.durationMs < 1000 ? `${stage.durationMs.toFixed(0)}ms` : `${(stage.durationMs / 1000).toFixed(1)}s`}</em> : null}
    </button>
  );
}

export function TrainingLoopStrip({ model, onStageSelect }: Props) {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [selectedStageKey, setSelectedStageKey] = useState<string | undefined>();
  const activeStage = model.activeStage;
  const selectedStage = [...model.lifecycle, ...model.stepLoop, ...model.milestones].find(
    (stage) => `${stage.scope}:${stage.id}` === selectedStageKey
  ) ?? activeStage;
  const drawerRef = useRef<HTMLElement | null>(null);
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const reducedMotion = useReducedMotion();

  useEffect(() => {
    if (drawerOpen || !activeStage) return;
    setSelectedStageKey(`${activeStage.scope}:${activeStage.id}`);
  }, [activeStage, drawerOpen]);

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
      const stageTargets = body.querySelectorAll(".loop-stage, .loop-level, .stage-evidence-bar");
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
            0.08
          )
          .fromTo(
            stageTargets,
            { autoAlpha: 0, y: -5 },
            { autoAlpha: 1, y: 0, duration: motionDurations.quick, stagger: motionStagger.compact, ease: motionEase.standard },
            0.16
          );
      } else {
        timeline
          .to(body, { autoAlpha: 0, maxHeight: 0, overflowY: "hidden", y: -8, pointerEvents: "none", duration: motionDurations.quick }, 0)
          .to(drawer, { height: TOP_DRAWER_HANDLE_PX, duration: motionDuration("drawer", reducedMotion) }, 0);
      }
    },
    { dependencies: [drawerOpen, selectedStageKey, model.message, reducedMotion], scope: drawerRef }
  );

  const selectStage = (stage: TrainingLoopStage) => {
    setSelectedStageKey(`${stage.scope}:${stage.id}`);
    setDrawerOpen(true);
    onStageSelect?.(stage);
  };

  return (
    <section className={`top-training-drawer ${drawerOpen ? "open" : ""}`} aria-label="training loop drawer" ref={drawerRef}>
      <button
        aria-controls="training-loop-drawer-body"
        aria-expanded={drawerOpen}
        className="training-loop-handle"
        onClick={() => setDrawerOpen((current) => !current)}
        type="button"
      >
        <i className={`training-loop-status-dot ${activeStage?.state ?? "pending"}`} aria-hidden="true" />
        <span>Training Loop</span>
        <strong>{activeStage ? `${stageProgress(model)} · ${activeStage.label}` : stageProgress(model)}</strong>
        <em>{model.message}</em>
        {drawerOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </button>

      <div className="training-loop-drawer-body" id="training-loop-drawer-body" aria-hidden={!drawerOpen} ref={bodyRef}>
        <section className="loop-level lifecycle-level" aria-label="run lifecycle">
          <header><span>Run Lifecycle</span><em>{model.eventDriven ? "runtime events" : "legacy evidence"}</em></header>
          <div className="lifecycle-stage-row" role="list">
            {model.lifecycle.map((stage, index) => (
              <div className="loop-stage-with-link" key={stage.id}>
                <StageButton stage={stage} selected={selectedStage === stage} onSelect={selectStage} />
                {index < model.lifecycle.length - 1 ? <ArrowRight size={12} aria-hidden="true" /> : null}
              </div>
            ))}
          </div>
        </section>

        <div className="loop-execution-grid">
          <section className="loop-level step-loop-level" aria-label="repeated training step">
            <header><span><RefreshCw size={13} /> Step Loop</span><em>{stageProgress(model)}</em></header>
            <div className="step-stage-row" role="list">
              {model.stepLoop.map((stage, index) => (
                <div className="loop-stage-with-link" key={stage.id}>
                  <StageButton stage={stage} selected={selectedStage === stage} onSelect={selectStage} />
                  {index < model.stepLoop.length - 1 ? <ArrowRight size={12} aria-hidden="true" /> : null}
                </div>
              ))}
            </div>
            <div className="loop-return"><RefreshCw size={12} /> next batch</div>
          </section>

          <section className="loop-level milestone-level" aria-label="training milestones">
            <header><span><GitBranch size={13} /> Milestones</span><em>periodic</em></header>
            <div className="milestone-stage-row" role="list">
              {model.milestones.map((stage) => (
                <StageButton key={stage.id} stage={stage} selected={selectedStage === stage} onSelect={selectStage} />
              ))}
            </div>
          </section>
        </div>

        {selectedStage ? (
          <section className={`stage-evidence-bar ${selectedStage.state}`} aria-live="polite">
            <div><span>{selectedStage.label}</span><strong>{selectedStage.detail}</strong></div>
            <em>{selectedStage.state}</em>
            {selectedStage.evidence ? (
              <button type="button" onClick={() => onStageSelect?.(selectedStage)}>{evidenceLabel(selectedStage)}</button>
            ) : null}
          </section>
        ) : null}
      </div>
    </section>
  );
}
