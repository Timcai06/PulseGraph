type Props = {
  values: (number | null | undefined)[];
  stroke?: string;
  width?: number;
  height?: number;
};

export function Sparkline({ values, stroke = "var(--cyan)", width = 110, height = 26 }: Props) {
  const points = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (points.length < 2) return <span className="spark-empty">–</span>;

  const min = Math.min(...points);
  const max = Math.max(...points);
  const range = max - min || 1;
  const coords = points
    .map((value, index) => {
      const x = (index / (points.length - 1)) * (width - 2) + 1;
      const y = height - 2 - ((value - min) / range) * (height - 4);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <svg className="sparkline" width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
      <polyline points={coords} fill="none" stroke={stroke} strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  );
}
