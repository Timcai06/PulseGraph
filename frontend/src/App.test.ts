import { describe, expect, it } from "vitest";
import appSource from "./App.tsx?raw";

describe("App workspace layout", () => {
  it("does not render the right-side layer inspector panel", () => {
    expect(appSource).not.toContain("<Layer" + "Inspector");
  });
});
