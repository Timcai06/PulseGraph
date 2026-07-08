type Props = {
  pixels?: number[];
  label?: number;
  prediction?: number;
};

export function DigitPreview({ pixels, label, prediction }: Props) {
  const isDigitImage = pixels?.length === 28 * 28;
  const cells = isDigitImage ? pixels : Array(28 * 28).fill(0);
  const sampleSize = pixels?.length ?? 0;

  return (
    <div className="digit-preview">
      <div className={`digit-grid ${isDigitImage ? "" : "generic"}`} aria-label="forward sample preview">
        {cells.map((value, index) => (
          <span
            className="digit-pixel"
            key={index}
            style={{ "--pixel": Math.max(0, Math.min(1, value)) } as React.CSSProperties}
          />
        ))}
        {!isDigitImage && (
          <span className="generic-sample">
            tensor
            <strong>{sampleSize}</strong>
          </span>
        )}
      </div>
      <div className="digit-meta">
        <span>label {label ?? "-"}</span>
        <span>pred {prediction ?? "-"}</span>
      </div>
    </div>
  );
}
