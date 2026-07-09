import type { PredictionResponse } from "../api/client";
import {
  classificationOutputFromPrediction,
  displayClassName,
  inferenceOutputKind,
  structuredOutputRows,
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
  const classification = classificationOutputFromPrediction(prediction);
  const probabilities = classification?.probabilities ?? [];
  const classNames = classification?.classNames;
  const top = topProbabilityRows(probabilities, classNames);
  const confidence = classification?.confidence ?? top[0]?.value ?? 0;
  const predictionLabel = classification ? displayClassName(classification.prediction, classNames) : "";
  const outputKind = prediction ? inferenceOutputKind(prediction) : "classification";
  const structuredRows = prediction && !classification ? structuredOutputRows(prediction.output) : [];

  return (
    <div className="inference-body">
      <div className="inference-image recognition-image">
        <span>Input</span>
        <ImagePreview pixels={prediction?.image_pixels} imageShape={prediction?.image_shape} />
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
            label={classification?.label ?? prediction?.label}
            prediction={classification?.prediction ?? prediction?.prediction}
            classNames={classNames}
            theme={theme}
          />
        )}
      </div>
    </div>
  );
}
