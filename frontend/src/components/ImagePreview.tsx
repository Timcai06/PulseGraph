import { useEffect, useMemo, useRef } from "react";
import { normalizeImageShape } from "../lib/inferenceView";

type Props = {
  pixels?: number[];
  imageShape?: number[] | null;
  size?: "normal" | "mini";
};

function clampByte(value: number | undefined) {
  return Math.round(Math.max(0, Math.min(1, value ?? 0)) * 255);
}

export function ImagePreview({ pixels, imageShape, size = "normal" }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const shape = useMemo(() => normalizeImageShape(imageShape, pixels?.length ?? 0), [imageShape, pixels]);
  const isRenderable = Boolean(pixels?.length && shape);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !pixels || !shape) return;
    const [channels, height, width] = shape;
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) return;
    const image = context.createImageData(width, height);
    const plane = width * height;

    for (let index = 0; index < plane; index += 1) {
      const target = index * 4;
      if (channels === 3) {
        image.data[target] = clampByte(pixels[index]);
        image.data[target + 1] = clampByte(pixels[plane + index]);
        image.data[target + 2] = clampByte(pixels[plane * 2 + index]);
      } else {
        const value = clampByte(pixels[index]);
        image.data[target] = value;
        image.data[target + 1] = value;
        image.data[target + 2] = value;
      }
      image.data[target + 3] = 255;
    }

    context.putImageData(image, 0, 0);
  }, [pixels, shape]);

  return (
    <div className={`image-preview image-preview-${size}`}>
      {isRenderable ? (
        <canvas ref={canvasRef} aria-label="forward sample preview" />
      ) : (
        <span className="image-preview-empty">-</span>
      )}
    </div>
  );
}
