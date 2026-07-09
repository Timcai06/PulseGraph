import type { PredictionResponse } from "../api/client";
import { displayClassName, topProbabilityRows } from "../lib/inferenceView";
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
  const probabilities = prediction?.probabilities ?? [];
  const classNames = prediction?.class_names;
  const top = topProbabilityRows(probabilities, classNames);
  const confidence = top[0]?.value ?? 0;
  const predictionLabel = prediction ? displayClassName(prediction.prediction, classNames) : "";

  return (
    <div className="inference-body">
      <div className="recognition-image">
        <span>Image</span>
        <ImagePreview pixels={prediction?.image_pixels} imageShape={prediction?.image_shape} />
      </div>
      <div className="inference-result">
        {prediction ? (
          <>
            <div className="recognition-callout">
              <span>Recognized</span>
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
        ) : (
          <p className="empty-hint">No result</p>
        )}
        <ProbabilityChart
          probabilities={probabilities.length ? probabilities : Array(10).fill(0)}
          label={prediction?.label}
          prediction={prediction?.prediction}
          classNames={classNames}
          theme={theme}
        />
      </div>
    </div>
  );
}
