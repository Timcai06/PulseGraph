import { useEffect, useId, useMemo, useState, type KeyboardEvent } from "react";

export type NumericRange = {
  min: number;
  max: number;
};

export type NumericDraftAssessment =
  | { kind: "valid"; nextValue: number; message: null }
  | { kind: "range"; nextValue: number; message: string }
  | { kind: "empty" | "invalid"; nextValue: null; message: string };

type NumericFieldProps = {
  label: string;
  value: number;
  range: NumericRange;
  onCommit: (value: number) => void;
};

function normalizeRange(range: NumericRange) {
  const min = Math.trunc(range.min);
  const max = Math.trunc(range.max);
  return min <= max ? { min, max } : { min: max, max: min };
}

function rangeLabel(range: NumericRange) {
  return `${range.min} to ${range.max}`;
}

function parseIntegerDraft(draft: string) {
  const trimmed = draft.trim();
  if (!trimmed) return { kind: "empty" as const };
  if (!/^-?\d+$/.test(trimmed)) return { kind: "invalid" as const };

  try {
    return { kind: "parsed" as const, value: BigInt(trimmed) };
  } catch {
    return { kind: "invalid" as const };
  }
}

export function assessNumericDraft(draft: string, inputRange: NumericRange): NumericDraftAssessment {
  const range = normalizeRange(inputRange);
  const parsed = parseIntegerDraft(draft);

  if (parsed.kind === "empty") {
    return {
      kind: "empty",
      nextValue: null,
      message: `Required. Enter a whole number from ${rangeLabel(range)}.`
    };
  }

  if (parsed.kind === "invalid") {
    return {
      kind: "invalid",
      nextValue: null,
      message: `Enter digits only. Allowed range ${rangeLabel(range)}.`
    };
  }

  const min = BigInt(range.min);
  const max = BigInt(range.max);

  if (parsed.value < min) {
    return {
      kind: "range",
      nextValue: range.min,
      message: `Will clamp to ${range.min} on apply. Allowed range ${rangeLabel(range)}.`
    };
  }

  if (parsed.value > max) {
    return {
      kind: "range",
      nextValue: range.max,
      message: `Will clamp to ${range.max} on apply. Allowed range ${rangeLabel(range)}.`
    };
  }

  return { kind: "valid", nextValue: Number(parsed.value), message: null };
}

export function commitNumericDraft(draft: string, currentValue: number, inputRange: NumericRange) {
  const range = normalizeRange(inputRange);
  const assessment = assessNumericDraft(draft, range);

  if (assessment.kind === "valid") {
    return {
      accepted: true,
      nextValue: assessment.nextValue,
      nextDraft: String(assessment.nextValue),
      message: null,
      tone: "note" as const
    };
  }

  if (assessment.kind === "range") {
    return {
      accepted: true,
      nextValue: assessment.nextValue,
      nextDraft: String(assessment.nextValue),
      message: `Clamped to ${assessment.nextValue}. Allowed range ${rangeLabel(range)}.`,
      tone: "note" as const
    };
  }

  return {
    accepted: false,
    nextValue: currentValue,
    nextDraft: String(currentValue),
    message: assessment.message,
    tone: "error" as const
  };
}

export function NumericField({ label, value, range, onCommit }: NumericFieldProps) {
  const statusId = useId();
  const [draft, setDraft] = useState(() => String(value));
  const [editing, setEditing] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [statusTone, setStatusTone] = useState<"note" | "error">("error");

  useEffect(() => {
    if (editing) return;
    setDraft(String(value));
  }, [editing, value]);

  const assessment = useMemo(() => assessNumericDraft(draft, range), [draft, range]);
  const liveMessage = editing && assessment.kind !== "valid" ? assessment.message : null;
  const message = liveMessage ?? statusMessage;
  const messageClassName =
    liveMessage ? (assessment.kind === "range" ? "control-empty-note" : "error-hint") : statusTone === "note" ? "control-empty-note" : "error-hint";

  const commitDraft = () => {
    const result = commitNumericDraft(draft, value, range);
    setDraft(result.nextDraft);
    setEditing(false);
    setStatusMessage(result.message);
    setStatusTone(result.tone);
    if (result.accepted && result.nextValue !== value) onCommit(result.nextValue);
  };

  const restoreDraft = () => {
    setDraft(String(value));
    setEditing(false);
    setStatusMessage(null);
    setStatusTone("error");
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") {
      event.preventDefault();
      event.currentTarget.blur();
      return;
    }

    if (event.key === "Escape") {
      event.preventDefault();
      restoreDraft();
    }
  };

  return (
    <label className="number-field">
      <span>{label}</span>
      <input
        type="text"
        inputMode="numeric"
        pattern="-?[0-9]*"
        enterKeyHint="done"
        autoComplete="off"
        spellCheck={false}
        aria-describedby={message ? statusId : undefined}
        aria-invalid={editing && assessment.kind !== "valid"}
        value={draft}
        onBlur={commitDraft}
        onFocus={() => {
          setEditing(true);
          setStatusMessage(null);
          setStatusTone("error");
        }}
        onChange={(event) => {
          setEditing(true);
          setDraft(event.target.value);
          setStatusMessage(null);
          setStatusTone("error");
        }}
        onKeyDown={handleKeyDown}
      />
      {message ? (
        <p aria-live="polite" className={messageClassName} id={statusId} role="status">
          {message}
        </p>
      ) : null}
    </label>
  );
}
