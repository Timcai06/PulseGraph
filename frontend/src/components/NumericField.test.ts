import { describe, expect, it } from "vitest";
import numericFieldSource from "./NumericField.tsx?raw";
import { assessNumericDraft, commitNumericDraft } from "./NumericField";

describe("NumericField", () => {
  it("keeps draft editing logic local until blur or enter commit", () => {
    expect(numericFieldSource).toContain("const [draft, setDraft] = useState(() => String(value))");
    expect(numericFieldSource).toContain('if (event.key === "Enter")');
    expect(numericFieldSource).toContain("event.currentTarget.blur()");
    expect(numericFieldSource).toContain('if (event.key === "Escape")');
    expect(numericFieldSource).toContain("restoreDraft()");
    expect(numericFieldSource).toContain("onBlur={commitDraft}");
    expect(numericFieldSource).toContain('type="text"');
  });

  it("accepts empty drafts during editing but rejects them on commit", () => {
    expect(assessNumericDraft("", { min: 1, max: 500 })).toEqual({
      kind: "empty",
      nextValue: null,
      message: "Required. Enter a whole number from 1 to 500."
    });

    expect(commitNumericDraft("", 120, { min: 1, max: 500 })).toEqual({
      accepted: false,
      nextValue: 120,
      nextDraft: "120",
      message: "Required. Enter a whole number from 1 to 500.",
      tone: "error"
    });
  });

  it("supports large pasted integers and clamps them at commit time", () => {
    expect(assessNumericDraft("9".repeat(100), { min: 1, max: 500 })).toEqual({
      kind: "range",
      nextValue: 500,
      message: "Will clamp to 500 on apply. Allowed range 1 to 500."
    });

    expect(commitNumericDraft("9".repeat(100), 100, { min: 1, max: 500 })).toEqual({
      accepted: true,
      nextValue: 500,
      nextDraft: "500",
      message: "Clamped to 500. Allowed range 1 to 500.",
      tone: "note"
    });
  });

  it("keeps valid in-range integers untouched", () => {
    expect(assessNumericDraft("240", { min: 1, max: 500 })).toEqual({
      kind: "valid",
      nextValue: 240,
      message: null
    });

    expect(commitNumericDraft("240", 120, { min: 1, max: 500 })).toEqual({
      accepted: true,
      nextValue: 240,
      nextDraft: "240",
      message: null,
      tone: "note"
    });
  });

  it("reports non-digit drafts accessibly and restores the last committed value", () => {
    expect(assessNumericDraft("12x", { min: 1, max: 500 })).toEqual({
      kind: "invalid",
      nextValue: null,
      message: "Enter digits only. Allowed range 1 to 500."
    });

    expect(commitNumericDraft("12x", 80, { min: 1, max: 500 })).toEqual({
      accepted: false,
      nextValue: 80,
      nextDraft: "80",
      message: "Enter digits only. Allowed range 1 to 500.",
      tone: "error"
    });
  });
});
