type Props = {
  pixels?: number[];
  label?: number;
  prediction?: number;
};

export function DigitPreview({ pixels, label, prediction }: Props) {
  const cells = pixels?.length === 28 * 28 ? pixels : Array(28 * 28).fill(0);

  return (
    <div className="digit-preview">
      <div className="digit-grid" aria-label="28 by 28 digit preview">
        {cells.map((value, index) => (
          <span
            className="digit-pixel"
            key={index}
            style={{ "--pixel": Math.max(0, Math.min(1, value)) } as React.CSSProperties}
          />
        ))}
      </div>
      <div className="digit-meta">
        <span>label {label ?? "-"}</span>
        <span>pred {prediction ?? "-"}</span>
      </div>
    </div>
  );
}
