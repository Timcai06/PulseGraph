import { describe, expect, it } from "vitest";
import appSource from "./App.tsx?raw";
import controlRailSource from "./components/ControlRail.tsx?raw";
import digitPreviewSource from "./components/DigitPreview.tsx?raw";
import inferenceProbeSource from "./components/InferenceProbe.tsx?raw";
import modelGraphSource from "./components/ModelGraphPanel.tsx?raw";

describe("App workspace layout", () => {
  it("does not render the right-side layer inspector panel", () => {
    expect(appSource).not.toContain("<Layer" + "Inspector");
  });

  it("keeps the monitor free of explanatory demo copy", () => {
    const combined = [appSource, controlRailSource, inferenceProbeSource, digitPreviewSource, modelGraphSource].join("\n");

    expect(combined).not.toContain("Netron-like structure with training pulses");
    expect(combined).not.toContain("Upload source, then run forward or train a short local recipe.");
    expect(combined).not.toContain("Secondary path: inspect weights");
    expect(combined).not.toContain("No active training runs. Completed runs are in History.");
    expect(combined).not.toContain("recorded probe");
    expect(combined).not.toContain("tensor");
  });

  it("waits for source training completion before running forward", () => {
    expect(appSource).toContain("pendingForwardRun");
    expect(appSource).not.toContain("applyPredictionResult(await runForward(result.run_id, result.checkpoint.step));");
  });
});
