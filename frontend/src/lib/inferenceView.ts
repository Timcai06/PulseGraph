export type ProbabilityRow = {
  index: number;
  label: string;
  value: number;
};

function shapeProduct(shape: number[]) {
  return shape.reduce((total, dim) => total * dim, 1);
}

export function displayClassName(index: number, classNames?: string[] | null): string {
  const value = classNames?.[index];
  return value ? value : String(index);
}

export function topProbabilityRows(probabilities: number[], classNames?: string[] | null): ProbabilityRow[] {
  return probabilities
    .map((value, index) => ({ index, label: displayClassName(index, classNames), value }))
    .sort((left, right) => right.value - left.value)
    .slice(0, 3);
}

export function chartProbabilityRows(probabilities: number[], classNames?: string[] | null): ProbabilityRow[] {
  const rows = probabilities.map((value, index) => ({ index, label: displayClassName(index, classNames), value }));
  if (rows.length <= 20) return rows;
  return [...rows].sort((left, right) => right.value - left.value).slice(0, 10);
}

export function normalizeImageShape(_shape: number[] | undefined | null, pixelCount: number): [number, number, number] | undefined {
  const shape = _shape?.map((dim) => Number(dim)).filter((dim) => Number.isFinite(dim) && dim > 0);
  if (shape?.length === 3 && (shape[0] === 1 || shape[0] === 3) && shapeProduct(shape) === pixelCount) {
    return [shape[0], shape[1], shape[2]];
  }
  if (shape?.length === 2 && shapeProduct(shape) === pixelCount) {
    return [1, shape[0], shape[1]];
  }
  const side = Math.sqrt(pixelCount);
  return Number.isInteger(side) ? [1, side, side] : undefined;
}
