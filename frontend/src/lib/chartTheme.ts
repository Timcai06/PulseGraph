export type Theme = "dark" | "light";

export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function withAlpha(color: string, alpha: number): string {
  if (color.startsWith("#") && (color.length === 7 || color.length === 4)) {
    const hex = color.length === 4 ? color.replace(/([0-9a-f])/gi, "$1$1") : color;
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
  return color;
}

export type ChartPalette = {
  text: string;
  grid: string;
  tooltipBg: string;
  green: string;
  amber: string;
  cyan: string;
  violet: string;
  red: string;
};

/** Read the chart palette from CSS design tokens so charts follow the active theme. */
export function chartPalette(): ChartPalette {
  return {
    text: cssVar("--text-muted") || "#94a3b8",
    grid: cssVar("--border") || "rgba(100, 116, 139, 0.24)",
    tooltipBg: cssVar("--surface-raised") || "rgba(30, 41, 59, 0.94)",
    green: cssVar("--green") || "#22c55e",
    amber: cssVar("--amber") || "#f59e0b",
    cyan: cssVar("--cyan") || "#38bdf8",
    violet: cssVar("--violet") || "#8b5cf6",
    red: cssVar("--red") || "#f87171"
  };
}
