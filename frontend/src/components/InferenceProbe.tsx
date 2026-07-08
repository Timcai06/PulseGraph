import type { PredictionResponse } from "../api/client";
import { ProbabilityChart } from "./Charts";
import { DigitPreview } from "./DigitPreview";

type Props = {
  prediction?: PredictionResponse;
  theme?: "dark" | "light";
};

function topPredictions(probabilities: number[], limit = 3) {
  return probabilities
    .map((value, index) => ({ index, value }))
    .sort((left, right) => right.value - left.value)
    .slice(0, limit);
}

function sourceBadge(source?: PredictionResponse["sample_source"]) {
  if (source === "mnist") return "Real dataset";
  if (source === "synthetic") return "Synthetic probe";
  if (source === "probe") return "Resource sample";
  return "No sample";
}

export function InferenceProbe({ prediction, theme = "dark" }: Props) {
  const probabilities = prediction?.probabilities ?? [];
  const top = topPredictions(probabilities);
  const confidence = top[0]?.value ?? 0;
  const isRealDataset = prediction?.sample_source === "mnist";

  return (
    <div className="inference-body">
      <div className="recognition-image">
        <span>Image</span>
        <DigitPreview pixels={prediction?.image_pixels} />
      </div>
      <div className="inference-result">
        {prediction ? (
          <>
            <div className="recognition-callout">
              <span>{isRealDataset ? "Recognized" : "Probe output"}</span>
              <strong>{prediction.prediction}</strong>
              <em>{(confidence * 100).toFixed(1)}% confidence</em>
            </div>
            <span className={`source-badge source-${prediction.sample_source}`}>
              {sourceBadge(prediction.sample_source)}
            </span>
            <div className="top-predictions">
              {top.map((item) => (
                <span key={item.index}>
                  {item.index}: {(item.value * 100).toFixed(1)}%
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
          theme={theme}
        />
      </div>
    </div>
  );
}
