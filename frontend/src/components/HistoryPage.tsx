import { useEffect, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, FileText, Play, Radio, Trash2 } from "lucide-react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import type { RunSummary } from "../api/client";
import { getRunDetail } from "../api/client";
import { useReducedMotion } from "../hooks/useReducedMotion";
import { motionDuration, motionEase, motionStagger } from "../lib/motion";
import { Sparkline } from "./Sparkline";

gsap.registerPlugin(useGSAP);

type Props = {
  runs: RunSummary[];
  title?: string;
  initialStatusFilter?: StatusFilter;
  statusFilters?: StatusFilter[];
  watchedRunId?: string;
  onWatchRun: (runId: string) => void;
  onOpenDetail: (runId: string, origin?: DOMRect) => void;
  onDeleteRun: (runId: string) => void;
};

type PreviewState = {
  runId: string;
  losses?: number[];
};

type StatusFilter = "all" | "completed" | "live" | "watched";

const defaultStatusFilters: StatusFilter[] = ["all", "completed", "live", "watched"];

function formatTime(seconds: number) {
  if (!seconds) return "unknown";
  return new Date(seconds * 1000).toLocaleString();
}

function extractLosses(rows: Record<string, unknown>[]): number[] {
  return rows
    .map((row) => row.loss)
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
}

export function HistoryPage({
  runs,
  title = "Run Library",
  initialStatusFilter = "all",
  statusFilters = defaultStatusFilters,
  watchedRunId,
  onWatchRun,
  onOpenDetail,
  onDeleteRun
}: Props) {
  const stripRef = useRef<HTMLDivElement | null>(null);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const trackRef = useRef<HTMLDivElement | null>(null);
  const previewRef = useRef<HTMLDivElement | null>(null);
  const quickMove = useRef<{ x: (value: number) => void; y: (value: number) => void } | null>(null);
  const lossCache = useRef(new Map<string, number[]>());
  const [preview, setPreview] = useState<PreviewState | null>(null);
  const [query, setQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>(initialStatusFilter);
  const [canScrollBack, setCanScrollBack] = useState(false);
  const [canScrollForward, setCanScrollForward] = useState(false);
  const reducedMotion = useReducedMotion();

  const normalizedQuery = query.trim().toLowerCase();
  const orderedRuns = [...runs]
    .filter((run) => {
      const matchesQuery = !normalizedQuery || run.run_id.toLowerCase().includes(normalizedQuery);
      const matchesStatus =
        statusFilter === "all" ||
        (statusFilter === "completed" && run.completed) ||
        (statusFilter === "live" && !run.completed) ||
        (statusFilter === "watched" && run.run_id === watchedRunId);
      return matchesQuery && matchesStatus;
    })
    .sort((a, b) => b.created_at - a.created_at);

  const groupedRuns = orderedRuns.reduce<Record<string, RunSummary[]>>((groups, run) => {
    const day = run.created_at ? new Date(run.created_at * 1000).toLocaleDateString() : "Unknown date";
    groups[day] = groups[day] ?? [];
    groups[day].push(run);
    return groups;
  }, {});

  useGSAP(() => {
    if (!stripRef.current || !orderedRuns.length) return;
    if (!reducedMotion) {
      gsap.from(".run-card", {
        opacity: 0,
        y: 18,
        duration: motionDuration("enter", reducedMotion),
        stagger: motionStagger.compact,
        ease: motionEase.standard,
        clearProps: "all"
      });
    }
  }, { dependencies: [orderedRuns.length, reducedMotion], scope: stripRef });

  useEffect(() => {
    const viewport = viewportRef.current;
    const track = trackRef.current;
    if (!viewport || !track) return;
    const update = () => {
      setCanScrollBack(viewport.scrollLeft > 1);
      setCanScrollForward(viewport.scrollLeft + viewport.clientWidth < viewport.scrollWidth - 1);
    };
    const observer = new ResizeObserver(update);
    observer.observe(viewport);
    observer.observe(track);
    viewport.addEventListener("scroll", update, { passive: true });
    update();
    return () => {
      observer.disconnect();
      viewport.removeEventListener("scroll", update);
    };
  }, [normalizedQuery, orderedRuns.length, statusFilter]);

  const scrollTimeline = (direction: -1 | 1) => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    viewport.scrollBy({
      left: direction * Math.max(280, viewport.clientWidth * 0.72),
      behavior: reducedMotion ? "auto" : "smooth"
    });
  };

  const handleTimelineKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      scrollTimeline(event.key === "ArrowLeft" ? -1 : 1);
      return;
    }
    if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      viewport.scrollTo({
        left: event.key === "Home" ? 0 : viewport.scrollWidth,
        behavior: reducedMotion ? "auto" : "smooth"
      });
    }
  };

  useGSAP(() => {
    const el = previewRef.current;
    if (!el) return;
    quickMove.current = {
      x: gsap.quickTo(el, "x", { duration: motionDuration("quick", reducedMotion), ease: motionEase.panel }),
      y: gsap.quickTo(el, "y", { duration: motionDuration("quick", reducedMotion), ease: motionEase.panel })
    };
  }, { dependencies: [reducedMotion] });

  const handleCardEnter = (run: RunSummary, event: React.MouseEvent) => {
    const cached = lossCache.current.get(run.run_id);
    setPreview({ runId: run.run_id, losses: cached });
    const el = previewRef.current;
    if (el) gsap.set(el, { x: event.clientX + 20, y: event.clientY + 20 });
    if (!cached) {
      getRunDetail(run.run_id)
        .then((detail) => {
          const losses = extractLosses(detail.metrics);
          lossCache.current.set(run.run_id, losses);
          setPreview((current) => (current?.runId === run.run_id ? { runId: run.run_id, losses } : current));
        })
        .catch(() => {
          lossCache.current.set(run.run_id, []);
        });
    }
  };

  const handleStripMove = (event: React.MouseEvent) => {
    quickMove.current?.x(event.clientX + 20);
    quickMove.current?.y(event.clientY + 20);
  };

  return (
    <section className="history-page">
      <header className="page-heading">
        <div>
          <h2>{title}</h2>
          <p>Filter runs by state, inspect trends, replay checkpoints, and open reports from the same library.</p>
        </div>
        <span>{orderedRuns.length} / {runs.length} runs</span>
      </header>

      <div className="library-controls">
        <label>
          <span>Search runs</span>
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="run id" />
        </label>
        {statusFilters.length > 1 ? (
          <div className="library-filters" aria-label="run status filters">
            {statusFilters.map((filter) => (
              <button
                className={statusFilter === filter ? "active" : ""}
                key={filter}
                onClick={() => setStatusFilter(filter)}
                type="button"
              >
                {filter}
              </button>
            ))}
          </div>
        ) : null}
      </div>

      {orderedRuns.length === 0 ? (
        <div className="history-empty">
          <p>No runs match this library view.</p>
        </div>
      ) : (
        <div
          className="timeline-strip"
          ref={stripRef}
          onMouseMove={handleStripMove}
          onMouseLeave={() => setPreview(null)}
        >
          <header className="timeline-navigation">
            <span>Run timeline</span>
            <div>
              <button aria-label="Scroll to earlier runs" disabled={!canScrollBack} onClick={() => scrollTimeline(-1)} type="button">
                <ChevronLeft size={15} />
              </button>
              <button aria-label="Scroll to later runs" disabled={!canScrollForward} onClick={() => scrollTimeline(1)} type="button">
                <ChevronRight size={15} />
              </button>
            </div>
          </header>
          <div
            className="timeline-viewport"
            ref={viewportRef}
            tabIndex={0}
            aria-label="Scrollable run timeline"
            onKeyDown={handleTimelineKeyDown}
          >
            <div className="timeline-track" ref={trackRef}>
              {Object.entries(groupedRuns).map(([group, groupRuns]) => (
                <section className="library-section" key={group}>
                  <h3>{group}</h3>
                  <div className="library-run-row">
                    {groupRuns.map((run) => (
                      <article
                        className={`run-card ${watchedRunId === run.run_id ? "watching" : ""}`}
                        key={run.run_id}
                        onMouseEnter={(event) => handleCardEnter(run, event)}
                      >
                        <header>
                          <strong>{run.run_id}</strong>
                          {watchedRunId === run.run_id && <Radio size={14} className="history-watching" />}
                        </header>
                        <span className="run-date">{formatTime(run.created_at)}</span>
                        <div className="run-card-meta">
                          <span>
                            steps <b>{run.last_step}</b>
                          </span>
                          <span>
                            events <b>{run.event_count}</b>
                          </span>
                        </div>
                        <div className="run-card-actions">
                          <button
                            className={watchedRunId === run.run_id ? "watching" : ""}
                            onClick={() => onWatchRun(run.run_id)}
                            type="button"
                          >
                            <Play size={14} /> Replay
                          </button>
                          <button
                            className="secondary"
                            onClick={(event) => {
                              const card = (event.currentTarget as HTMLElement).closest(".run-card");
                              onOpenDetail(run.run_id, card?.getBoundingClientRect());
                            }}
                            type="button"
                          >
                            <FileText size={14} /> Detail
                          </button>
                          <button className="secondary" onClick={() => onDeleteRun(run.run_id)} type="button" aria-label={`Delete ${run.run_id}`}>
                            <Trash2 size={14} />
                          </button>
                        </div>
                      </article>
                    ))}
                  </div>
                </section>
              ))}
            </div>
          </div>

          <div className={`run-preview ${preview ? "visible" : ""}`} ref={previewRef} aria-hidden="true">
            {preview && (
              <>
                <span className="preview-title">{preview.runId}</span>
                {preview.losses === undefined ? (
                  <span className="preview-loading">loading loss trend…</span>
                ) : preview.losses.length < 2 ? (
                  <span className="preview-loading">no loss telemetry</span>
                ) : (
                  <Sparkline values={preview.losses} width={190} height={46} />
                )}
                <span className="preview-meta">
                  {preview.losses && preview.losses.length >= 2
                    ? `loss ${preview.losses[0].toFixed(3)} → ${preview.losses[preview.losses.length - 1].toFixed(3)}`
                    : ""}
                </span>
              </>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
