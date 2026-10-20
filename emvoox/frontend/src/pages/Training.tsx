import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ApiError, DatasetOut, ModelVersion, TrainingRun, api } from "../api";

interface Preflight { ok: boolean; checks: { name: string; ok: boolean; detail: string }[]; resolved_config: Record<string, unknown> }
const ACTIVE = new Set(["queued", "preparing", "running", "validating"]);

export default function Training() {
  const qc = useQueryClient();
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: () => api.get<DatasetOut[]>("/datasets") });
  const models = useQuery({ queryKey: ["models"], queryFn: () => api.get<ModelVersion[]>("/models") });
  const runs = useQuery({ queryKey: ["runs"], queryFn: () => api.get<TrainingRun[]>("/training-runs"), refetchInterval: 4000 });
  const [versionId, setVersionId] = useState("");
  const [baseId, setBaseId] = useState("");
  const [trigger, setTrigger] = useState("");
  const [preset, setPreset] = useState("sdxl-8gb-explore");
  const [steps, setSteps] = useState<number | "">("");
  const [pf, setPf] = useState<Preflight | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);

  const bases = (models.data ?? []).filter((m) => m.kind === "base");
  const versions = (datasets.data ?? []).flatMap((d) => d.versions.map((v) => ({ ...v, name: d.name, trigger: d.trigger_token })));
  const body = () => ({
    dataset_version_id: versionId, base_model_id: baseId || bases[0]?.id, trigger_token: trigger, preset,
    overrides: steps === "" ? {} : { max_train_steps: steps },
  });
  const preflight = useMutation({
    mutationFn: () => api.post<Preflight>("/training-runs/preflight", body()),
    onSuccess: (r) => { setPf(r); setErr(null); },
    onError: (e: unknown) => setErr(e instanceof ApiError ? e.message : String(e)),
  });
  const start = useMutation({
    mutationFn: () => api.post<TrainingRun>("/training-runs", body()),
    onSuccess: (r) => { setOpen(r.id); setErr(null); qc.invalidateQueries({ queryKey: ["runs"] }); },
    onError: (e: unknown) => setErr(e instanceof ApiError ? e.message : String(e)),
  });

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Huấn luyện Character LoRA</h1>
      <div className="card grid gap-3 sm:grid-cols-2">
        <div>
          <label className="label">Dataset version</label>
          <select className="input" value={versionId} onChange={(e) => {
            setVersionId(e.target.value);
            const v = versions.find((x) => x.id === e.target.value);
            if (v?.trigger && !trigger) setTrigger(v.trigger);
          }}>
            <option value="">—</option>
            {versions.map((v) => <option key={v.id} value={v.id}>{v.name} v{v.version} ({v.item_count} ảnh)</option>)}
          </select>
        </div>
        <div>
          <label className="label">Model nền</label>
          <select className="input" value={baseId} onChange={(e) => setBaseId(e.target.value)}>
            {bases.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
          </select>
        </div>
        <div><label className="label">Trigger token</label><input className="input" value={trigger} onChange={(e) => setTrigger(e.target.value)} /></div>
        <div>
          <label className="label">Preset (thăm dò, chưa benchmark)</label>
          <select className="input" value={preset} onChange={(e) => setPreset(e.target.value)}>
            <option value="sdxl-8gb-explore">8 GB — 512px, batch 1, checkpointing</option>
            <option value="sdxl-24gb-explore">≥20 GB — 1024px</option>
          </select>
        </div>
        <div><label className="label">Ghi đè số bước (tùy chọn)</label>
          <input className="input" type="number" value={steps} onChange={(e) => setSteps(e.target.value === "" ? "" : Number(e.target.value))} /></div>
        <div className="flex items-end gap-2">
          <button className="btn-ghost" disabled={!versionId || !trigger} onClick={() => preflight.mutate()}>Preflight</button>
          <button className="btn" disabled={!versionId || !trigger || start.isPending} onClick={() => start.mutate()}>Bắt đầu huấn luyện</button>
        </div>
      </div>
      {err && <p className="text-sm text-red-600">{err}</p>}
      {pf && (
        <div className="card text-sm">
          <b>{pf.ok ? "Preflight đạt" : "Preflight có vấn đề"}</b>
          <ul className="mt-2 space-y-1">
            {pf.checks.map((c) => <li key={c.name} className={c.ok ? "text-emerald-700" : "text-red-700"}>{c.ok ? "✓" : "✗"} {c.name} — {c.detail}</li>)}
          </ul>
          <details className="mt-2"><summary>Cấu hình đã resolve</summary><pre className="text-xs">{JSON.stringify(pf.resolved_config, null, 2)}</pre></details>
        </div>
      )}
      <div className="space-y-3">
        <h2 className="font-medium">Các lần huấn luyện</h2>
        {runs.data?.length === 0 && <p className="text-sm text-slate-500">Chưa có run nào.</p>}
        {runs.data?.map((r) => <RunCard key={r.id} run={r} open={open === r.id} toggle={() => setOpen(open === r.id ? null : r.id)} />)}
      </div>
    </div>
  );
}

function RunCard({ run, open, toggle }: { run: TrainingRun; open: boolean; toggle: () => void }) {
  const qc = useQueryClient();
  const metrics = useQuery({ queryKey: ["metrics", run.id], enabled: open, refetchInterval: ACTIVE.has(run.state) ? 4000 : false,
    queryFn: () => api.get<{ step: number; loss: number | null }[]>(`/training-runs/${run.id}/metrics`) });
  const log = useQuery({ queryKey: ["log", run.id], enabled: open, refetchInterval: ACTIVE.has(run.state) ? 4000 : false,
    queryFn: () => api.get<{ lines: string[] }>(`/training-runs/${run.id}/log?lines=60`) });
  const review = (decision: string) => api.post(`/model-versions/${run.output_model_version_id}/review`, { decision })
    .then(() => { qc.invalidateQueries({ queryKey: ["runs"] }); qc.invalidateQueries({ queryKey: ["models"] }); });
  const pct = run.total_steps ? Math.round((run.current_step / run.total_steps) * 100) : 0;
  const pts = metrics.data?.filter((m) => m.loss !== null) ?? [];
  return (
    <div className="card space-y-2">
      <div className="flex items-center justify-between">
        <button className="text-left text-sm font-medium" onClick={toggle}>{run.trigger_token} · {run.state} · bước {run.current_step}/{run.total_steps ?? "?"}</button>
        <div className="flex gap-2">
          {ACTIVE.has(run.state) && <button className="btn-ghost" onClick={() => api.post(`/training-runs/${run.id}/cancel`)}>Hủy</button>}
          {run.state === "awaiting_review" && run.output_model_version_id && (<>
            <button className="btn" onClick={() => review("approved")}>Duyệt LoRA</button>
            <button className="btn-ghost" onClick={() => review("rejected")}>Loại</button>
          </>)}
        </div>
      </div>
      <div className="h-1.5 rounded bg-slate-100"><div className="h-1.5 rounded bg-indigo-600" style={{ width: `${pct}%` }} /></div>
      {run.error_message && <pre className="max-h-32 overflow-auto rounded bg-red-50 p-2 text-xs text-red-700">{run.error_code}: {run.error_message}</pre>}
      {open && (
        <div className="space-y-2">
          {pts.length > 1 ? <LossChart pts={pts as { step: number; loss: number }[]} /> : <p className="text-xs text-slate-500">Chưa có số liệu loss.</p>}
          <pre className="max-h-48 overflow-auto rounded bg-slate-900 p-2 text-xs text-slate-100">{log.data?.lines.join("\n") || "(chưa có log)"}</pre>
        </div>
      )}
    </div>
  );
}

function LossChart({ pts }: { pts: { step: number; loss: number }[] }) {
  const w = 480, h = 120;
  const xs = pts.map((p) => p.step), ys = pts.map((p) => p.loss);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const px = (x: number) => ((x - x0) / Math.max(1e-9, x1 - x0)) * (w - 10) + 5;
  const py = (y: number) => h - 5 - ((y - y0) / Math.max(1e-9, y1 - y0)) * (h - 10);
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full max-w-xl rounded border bg-white">
      <polyline fill="none" stroke="#4f46e5" strokeWidth="1.5" points={pts.map((p) => `${px(p.step)},${py(p.loss)}`).join(" ")} />
      <text x="6" y="12" fontSize="10" fill="#64748b">loss {y1.toFixed(3)} → {ys[ys.length - 1].toFixed(3)}</text>
    </svg>
  );
}
