import { Activity, Cpu, GitBranch } from "lucide-react";

type Props = {
  backendStatus: string;
  runStatus: string;
  step: number;
  device: string;
};

function backendDotClass(status: string) {
  if (status === "ok") return "ok";
  if (status === "checking") return "idle";
  return "offline";
}

function runDotClass(status: string) {
  if (status === "streaming") return "streaming";
  if (status === "complete") return "complete";
  if (status === "error") return "offline";
  return "idle";
}

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
        <span>
          <i className={`status-dot ${backendDotClass(backendStatus)}`} />
          <Activity size={16} /> API {backendStatus}
        </span>
        <span>
          <i className={`status-dot ${runDotClass(runStatus)}`} />
          <GitBranch size={16} /> {runStatus}
        </span>
        <span>
          <Cpu size={16} /> {device}
        </span>
        <span className="numeric">step {step}</span>
      </div>
    </header>
  );
}
