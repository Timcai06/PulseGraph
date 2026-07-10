import { describe, expect, it } from "vitest";
import controlRailSource from "./ControlRail.tsx?raw";

describe("ControlRail", () => {
  it("organizes the drawer into resource, train, and run contexts", () => {
    expect(controlRailSource).toContain("control-tabs");
    expect(controlRailSource).toContain('role="tablist"');
    expect(controlRailSource).toContain('role="tab"');
    expect(controlRailSource).toContain('aria-controls={`control-panel-${value}`}');
    expect(controlRailSource).toContain("tabIndex={active ? 0 : -1}");
    expect(controlRailSource).toContain('event.key === "ArrowRight"');
    expect(controlRailSource).toContain('event.key === "ArrowLeft"');
    expect(controlRailSource).toContain('event.key === "Home"');
    expect(controlRailSource).toContain('event.key === "End"');
    expect(controlRailSource).toContain("Resource");
    expect(controlRailSource).toContain("Train");
    expect(controlRailSource).toContain("Run");
  });

  it("repositions the floating drawer when responsive width changes", () => {
    expect(controlRailSource).toContain("new ResizeObserver");
    expect(controlRailSource).toContain("drawerWidth, railDrawerOpen, reducedMotion");
  });

  it("keeps packaged image and manifest assets when importing a resource folder", () => {
    expect(controlRailSource).toContain('const resourceAssetSuffixes = [".py", ".json", ".png", ".jpg", ".jpeg"]');
    expect(controlRailSource).toContain("resourceAssetSuffixes.some");
    expect(controlRailSource).toContain("webkitRelativePath");
  });

  it("keeps only live runs inside a collapsed surface", () => {
    expect(controlRailSource).toContain("control-disclosure");
    expect(controlRailSource).toContain("Live runs");
    expect(controlRailSource).toContain("run-item");
    expect(controlRailSource).toContain("control-summary-grid");
    expect(controlRailSource).not.toContain("Preview samples");
    expect(controlRailSource).not.toContain("Resource contract");
    expect(controlRailSource).not.toContain("resourceContractRows");
    expect(controlRailSource).not.toContain("resource-samples");
    expect(controlRailSource).not.toContain("Three compact contexts");
    expect(controlRailSource).not.toContain("Preview stays");
    expect(controlRailSource).not.toContain("Training uses");
    expect(controlRailSource).not.toContain("Prediction output");
  });
});
