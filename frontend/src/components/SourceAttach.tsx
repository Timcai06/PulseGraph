import { useState } from "react";
import { CheckCircle2, FileCode2, FolderOpen, Loader2, XCircle } from "lucide-react";
import type { SourceCandidate, SourceValidationResult } from "../api/client";
import { analyzeSourceCandidates, attachRunSource, type NamedSourceFile } from "../api/client";

type Props = {
  runId: string;
  onAttached: () => void;
};

function toNamedFiles(list: FileList | null): NamedSourceFile[] {
  if (!list) return [];
  return Array.from(list)
    .filter((file) => file.name.endsWith(".py") || file.name.endsWith(".zip"))
    .map((file) => {
      const relative = (file as File & { webkitRelativePath?: string }).webkitRelativePath;
      // folder uploads prefix every path with the selected folder's name — drop it
      const path = relative ? relative.split("/").slice(1).join("/") || file.name : file.name;
      return { file, path };
    });
}

export function SourceAttach({ runId, onAttached }: Props) {
  const [files, setFiles] = useState<NamedSourceFile[]>([]);
  const [candidates, setCandidates] = useState<SourceCandidate[]>([]);
  const [selected, setSelected] = useState<SourceCandidate | undefined>();
  const [phase, setPhase] = useState<"pick" | "analyzing" | "choose" | "attaching">("pick");
  const [validation, setValidation] = useState<SourceValidationResult | null | undefined>();
  const [error, setError] = useState<string | undefined>();

  const handleFiles = async (list: FileList | null) => {
    const named = toNamedFiles(list);
    if (!named.length) return;
    setFiles(named);
    setError(undefined);
    setValidation(undefined);
    setPhase("analyzing");
    try {
      const result = await analyzeSourceCandidates(named);
      setCandidates(result.candidates);
      setSelected(result.candidates[0]);
      setPhase("choose");
      if (!result.candidates.length) setError("No nn.Module subclass was found in the uploaded files.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed.");
      setPhase("pick");
    }
  };

  const handleAttach = async () => {
    if (!selected) return;
    setPhase("attaching");
    setError(undefined);
    try {
      const result = await attachRunSource(runId, files, selected.file, selected.class_name);
      setValidation(result.validation);
      if (!result.validation || result.validation.ok) onAttached();
      setPhase("choose");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Attaching source failed.");
      setPhase("choose");
    }
  };

  return (
    <div className="source-attach">
      <p className="hint">
        Attach the model definition (entry class + any custom modules it imports) to enable forward replay and
        checkpoint evaluation. Attached source is executed on replay — only attach code you trust.
      </p>

      <div className="attach-pickers">
        <label className="attach-picker">
          <FileCode2 size={15} /> Select .py / .zip
          <input type="file" multiple accept=".py,.zip" onChange={(event) => void handleFiles(event.target.files)} />
        </label>
        <label className="attach-picker">
          <FolderOpen size={15} /> Select folder
          <input
            type="file"
            multiple
            // @ts-expect-error non-standard folder upload attribute
            webkitdirectory=""
            onChange={(event) => void handleFiles(event.target.files)}
          />
        </label>
      </div>

      {files.length > 0 && (
        <p className="hint">
          {files.length} file{files.length === 1 ? "" : "s"}: {files.map((item) => item.path).join(", ")}
        </p>
      )}

      {phase === "analyzing" && (
        <p className="hint">
          <Loader2 size={13} className="spin" /> Scanning for nn.Module classes…
        </p>
      )}

      {(phase === "choose" || phase === "attaching") && candidates.length > 0 && (
        <div className="attach-entry">
          <label>
            Entry class
            <select
              value={selected ? `${selected.file}::${selected.class_name}` : ""}
              onChange={(event) => {
                const [file, className] = event.target.value.split("::");
                setSelected(candidates.find((item) => item.file === file && item.class_name === className));
              }}
            >
              {candidates.map((candidate) => (
                <option key={`${candidate.file}::${candidate.class_name}`} value={`${candidate.file}::${candidate.class_name}`}>
                  {candidate.class_name} — {candidate.file}
                </option>
              ))}
            </select>
          </label>
          <button disabled={!selected || phase === "attaching"} onClick={() => void handleAttach()} type="button">
            {phase === "attaching" ? <Loader2 size={14} className="spin" /> : <FileCode2 size={14} />} Attach & validate
          </button>
        </div>
      )}

      {validation && (
        <div className={`attach-validation ${validation.ok ? "ok" : "fail"}`}>
          {validation.ok ? (
            <>
              <CheckCircle2 size={15} /> Source matches the checkpoint weights — replay is ready.
            </>
          ) : (
            <>
              <XCircle size={15} />
              <span>
                {validation.error}
                {validation.missing_keys.length > 0 && ` Missing: ${validation.missing_keys.join(", ")}.`}
                {validation.unexpected_keys.length > 0 && ` Unexpected: ${validation.unexpected_keys.join(", ")}.`}
              </span>
            </>
          )}
        </div>
      )}

      {error && <p className="error-hint">{error}</p>}
    </div>
  );
}
