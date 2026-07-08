type Props = {
  pixels?: number[];
};

export function DigitPreview({ pixels }: Props) {
  const isDigitImage = pixels?.length === 28 * 28;
  const cells = isDigitImage ? pixels : Array(28 * 28).fill(0);

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
            -
          </span>
        )}
      </div>
    </div>
  );
}
