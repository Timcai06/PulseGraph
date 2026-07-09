import { describe, expect, it } from "vitest";
import { readFile } from "node:fs/promises";

async function cssModule(name: string) {
  return readFile(new URL(`./modules/${name}`, import.meta.url), "utf8");
}

function rule(css: string, selector: string) {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = css.match(new RegExp(`${escaped}\\s*\\{(?<body>[^}]*)\\}`, "m"));
  return match?.groups?.body ?? "";
}

describe("theme token usage", () => {
  it("keeps the timeline scrubber aligned with global theme surfaces", async () => {
    const panels = await cssModule("panels.css");
    const timeline = rule(panels, ".timeline-scrubber");

    expect(timeline).toContain("var(--surface");
    expect(timeline).toContain("var(--border");
    expect(timeline).not.toContain("rgba(2, 6, 23");
    expect(timeline).not.toContain("rgba(34, 211, 238");
  });
});
