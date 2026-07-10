import type { CSSProperties } from "react";
import { formatConfidencePercent, type DetectionBoxView, type DetectionView } from "../lib/inferenceView";

type Props = {
  detection: DetectionView;
  imageWidth: number;
  imageHeight: number;
  size?: "normal" | "mini";
  label?: string;
};

const overlayColors = ["var(--cyan)", "var(--teal)", "var(--amber)", "var(--violet)", "var(--green)", "var(--accent-text)"];

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function boxLabel(box: DetectionBoxView) {
  return box.score === undefined ? box.labelName : `${box.labelName} ${formatConfidencePercent(box.score)}`;
}

function boxStyle(box: DetectionBoxView, imageWidth: number, imageHeight: number): CSSProperties {
  const [x1, y1, x2, y2] = box.coordinates;
  const left = clamp((Math.min(x1, x2) / imageWidth) * 100, 0, 100);
  const top = clamp((Math.min(y1, y2) / imageHeight) * 100, 0, 100);
  const right = clamp((Math.max(x1, x2) / imageWidth) * 100, 0, 100);
  const bottom = clamp((Math.max(y1, y2) / imageHeight) * 100, 0, 100);

  return {
    left: `${left}%`,
    top: `${top}%`,
    width: `${Math.max(right - left, 2)}%`,
    height: `${Math.max(bottom - top, 2)}%`,
    "--detection-color": overlayColors[(box.label + box.index) % overlayColors.length]
  } as CSSProperties;
}

export function DetectionOverlay({ detection, imageWidth, imageHeight, size = "normal", label = "Detected objects" }: Props) {
  if (!detection.boxes.length) return null;

  return (
    <div className={`detection-overlay detection-overlay-${size}`} role="list" aria-label={label}>
      {detection.boxes.map((box) => {
        const chip = boxLabel(box);
        const [x1, y1, x2, y2] = box.coordinates;

        return (
          <div
            className="detection-box"
            key={`${box.index}-${box.label}-${box.coordinates.join("-")}`}
            role="listitem"
            tabIndex={0}
            aria-label={`${chip}, box from ${Math.round(x1)}, ${Math.round(y1)} to ${Math.round(x2)}, ${Math.round(y2)}`}
            style={boxStyle(box, imageWidth, imageHeight)}
          >
            <span className="detection-chip">{chip}</span>
          </div>
        );
      })}
    </div>
  );
}
