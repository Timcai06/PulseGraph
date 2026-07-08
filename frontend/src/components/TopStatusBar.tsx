import { Activity, Cpu, GitBranch } from "lucide-react";

type Props = {
  backendStatus: string;
  runStatus: string;
  step: number;
  device: string;
};

export function TopStatusBar({ backendStatus, runStatus, step, device }: Props) {
  return (
    <header className="top-bar">
      <div className="brand">
        <div className="brand-mark">PG</div>
        <div>
          <h1>PulseGraph</h1>
          <p>PyTorch model, training, and infra observability</p>
        </div>
      </div>
      <div className="status-strip">
        <span><Activity size={16} /> API {backendStatus}</span>
        <span><GitBranch size={16} /> {runStatus}</span>
        <span><Cpu size={16} /> {device}</span>
        <span>step {step}</span>
      </div>
    </header>
  );
}

