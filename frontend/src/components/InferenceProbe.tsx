import type { PredictionResponse } from "../api/client";
import {
  describeDetectionSummary,
  displayClassName,
  formatConfidencePercent,
  formatDetectionCoordinates,
  inferenceOutputKind,
  resolveInferenceRenderer,
  topProbabilityRows
} from "../lib/inferenceView";
import { ProbabilityChart } from "./Charts";
import { ImagePreview } from "./ImagePreview";

type Props = {
  prediction?: PredictionResponse;
  theme?: "dark" | "light";
};

function sourceBadge(source?: PredictionResponse["sample_source"]) {
  if (source === "mnist") return "Real dataset";
  if (source === "synthetic") return "Synthetic probe";
  if (source === "probe") return "Resource sample";
  return "No sample";
}

export function InferenceProbe({ prediction, theme = "dark" }: Props) {
  const renderer = resolveInferenceRenderer(prediction, prediction?.output_schema?.renderer);
  const classification = renderer?.renderer === "classification" ? renderer.classification : undefined;
  const detection = renderer?.renderer === "detection" ? renderer.detection : undefined;
  const probabilities = classification?.probabilities ?? [];
  const classNames = classification?.classNames;
  const top = topProbabilityRows(probabilities, classNames);
  const confidence = classification?.confidence ?? top[0]?.value ?? 0;
  const predictionLabel = classification ? displayClassName(classification.prediction, classNames) : "";
  const outputKind = renderer?.kind ?? (prediction ? inferenceOutputKind(prediction) : "classification");
  const structuredRows = renderer?.renderer === "structured" ? renderer.rows : [];
  const legacyLabel = typeof prediction?.label === "number" ? prediction.label : undefined;
  const legacyPrediction = typeof prediction?.prediction === "number" ? prediction.prediction : undefined;

  return (
    <div className="inference-body">
      <div className="inference-image recognition-image">
        <span>Input</span>
        <ImagePreview
          pixels={prediction?.image_pixels}
          imageShape={prediction?.image_shape}
          detection={detection}
          overlayLabel={detection ? "Inference detections" : undefined}
        />
      </div>
      <div className="inference-result">
        {prediction && classification ? (
          <>
            <div className="classification-output recognition-callout">
              <span>Top Prediction</span>
              <strong>{predictionLabel}</strong>
              <em>{(confidence * 100).toFixed(1)}% confidence</em>
            </div>
            <span className={`source-badge source-${prediction.sample_source}`}>
              {sourceBadge(prediction.sample_source)}
            </span>
            <div className="top-predictions">
              {top.map((item) => (
                <span key={item.index}>
                  {item.label}: {(item.value * 100).toFixed(1)}%
                </span>
              ))}
            </div>
          </>
        ) : prediction && detection ? (
          <>
            <div className="detection-output recognition-callout">
              <span>{outputKind}</span>
              <strong>{detection.totalCount}</strong>
              <em>{describeDetectionSummary(detection)}</em>
            </div>
            <span className={`source-badge source-${prediction.sample_source}`}>
              {sourceBadge(prediction.sample_source)}
            </span>
            <div className="detection-results" role="list" aria-label="Detection results">
              {detection.boxes.length ? (
                detection.boxes.map((box) => (
                  <div className="detection-row" key={`${box.index}-${box.label}-${box.coordinates.join("-")}`} role="listitem">
                    <div className="detection-copy">
                      <strong>{box.labelName}</strong>
                      <span>{formatDetectionCoordinates(box.coordinates)}</span>
                    </div>
                    <span className="detection-score">{box.score === undefined ? "score n/a" : formatConfidencePercent(box.score)}</span>
                  </div>
                ))
              ) : (
                <p className="empty-hint">No detections</p>
              )}
              {detection.truncated ? (
                <p className="detection-truncated">Showing {detection.boxes.length} of {detection.totalCount} detections.</p>
              ) : null}
            </div>
          </>
        ) : prediction ? (
          <div className="structured-output">
            <span className="output-kind">{outputKind}</span>
            <strong>Structured Output</strong>
            {structuredRows.length ? (
              <dl>
                {structuredRows.map((row) => (
                  <div key={row.key}>
                    <dt>{row.key}</dt>
                    <dd>{row.value}</dd>
                  </div>
                ))}
              </dl>
            ) : (
              <p className="empty-hint">No fields</p>
            )}
          </div>
        ) : (
          <p className="empty-hint">No output</p>
        )}
        {(classification || !prediction) && (
          <ProbabilityChart
            probabilities={probabilities.length ? probabilities : Array(10).fill(0)}
            label={classification?.label ?? legacyLabel}
            prediction={classification?.prediction ?? legacyPrediction}
            classNames={classNames}
            theme={theme}
          />
        )}
      </div>
    </div>
  );
}
