import { Upload, Play, RotateCcw, Zap } from "lucide-react";

type Props = {
  onInspect: (file: File) => void;
  onDemoForward: () => void;
  onStartStream: () => void;
  onReset: () => void;
};

export function ControlRail({ onInspect, onDemoForward, onStartStream, onReset }: Props) {
  return (
    <aside className="control-rail">
      <section>
        <h2>Model File</h2>
        <label className="file-drop">
          <Upload size={18} />
          <span>Inspect .pt</span>
          <input
            type="file"
            accept=".pt,.pth"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) onInspect(file);
            }}
          />
        </label>
        <p className="hint">Default path uses safe weights-only inspection.</p>
      </section>

      <section>
        <h2>Trusted Demo</h2>
        <button onClick={onDemoForward}><Zap size={16} /> Run forward</button>
        <button onClick={onStartStream}><Play size={16} /> Start stream</button>
        <button className="secondary" onClick={onReset}><RotateCcw size={16} /> Reset</button>
      </section>
    </aside>
  );
}

