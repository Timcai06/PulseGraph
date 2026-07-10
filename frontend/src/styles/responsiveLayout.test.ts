import { describe, expect, it } from "vitest";
import { readFile } from "node:fs/promises";


describe("responsive observability layout", () => {
  it("keeps all three dock panels available on compact desktop widths", async () => {
    const css = await readFile(new URL("./modules/responsive.css", import.meta.url), "utf8");

    expect(css).toContain("grid-template-columns: minmax(270px, 1fr) minmax(280px, 1fr) minmax(180px, 0.64fr)");
    expect(css).not.toContain(".event-panel {\n    display: none;");
  });

  it("anchors the graph toolbar inside the mobile graph stage", async () => {
    const css = await readFile(new URL("./modules/responsive.css", import.meta.url), "utf8");

    expect(css).toContain(".graph-stage {\n    position: relative;");
    expect(css).toContain(".topology-summary {\n    display: none;");
  });
});
