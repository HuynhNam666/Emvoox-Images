import { useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ApiError, DatasetItem, DatasetOut, api, assetUrl } from "../api";

export default function Datasets() {
  const qc = useQueryClient();
  const list = useQuery({ queryKey: ["datasets"], queryFn: () => api.get<DatasetOut[]>("/datasets") });
  const [sel, setSel] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [trigger, setTrigger] = useState("");
  const create = useMutation({
    mutationFn: () => api.post<DatasetOut>("/datasets", { name, trigger_token: trigger || null }),
    onSuccess: (d) => { setName(""); setSel(d.id); qc.invalidateQueries({ queryKey: ["datasets"] }); },
  });
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Dataset Manager</h1>
      <div className="card flex flex-wrap items-end gap-3">
        <div><label className="label">Tên dataset</label><input className="input" value={name} onChange={(e) => setName(e.target.value)} /></div>
        <div><label className="label">Trigger token (vd a01char)</label><input className="input" value={trigger} onChange={(e) => setTrigger(e.target.value)} /></div>
        <button className="btn" disabled={!name} onClick={() => create.mutate()}>Tạo dataset</button>
      </div>
      <div className="flex flex-wrap gap-2">
        {list.data?.map((d) => (
          <button key={d.id} onClick={() => setSel(d.id)} className={`rounded border px-3 py-1 text-sm ${sel === d.id ? "border-indigo-600 bg-indigo-50" : "bg-white"}`}>
            {d.name} · {d.approved_count}/{d.item_count} duyệt · {d.versions.length} version
          </button>
        ))}
      </div>
      {list.data?.length === 0 && <p className="text-sm text-slate-500">Chưa có dataset. Tạo một dataset rồi tải ảnh lên.</p>}
      {sel && list.data && <DatasetPanel key={sel} ds={list.data.find((d) => d.id === sel)!} />}
    </div>
  );
}

function DatasetPanel({ ds }: { ds: DatasetOut }) {
  const qc = useQueryClient();
  const file = useRef<HTMLInputElement>(null);
  const [license, setLicense] = useState("");
  const [source, setSource] = useState("");
  const [report, setReport] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const items = useQuery({ queryKey: ["items", ds.id], queryFn: () => api.get<{ total: number; items: DatasetItem[] }>(`/datasets/${ds.id}/items?limit=200`) });
  const refresh = () => { qc.invalidateQueries({ queryKey: ["items", ds.id] }); qc.invalidateQueries({ queryKey: ["datasets"] }); };

  function upload() {
    const files = file.current?.files;
    if (!files?.length) return;
    const form = new FormData();
    Array.from(files).forEach((f) => form.append("files", f));
    form.append("license", license);
    if (source) form.append("source", source);
    // XHR gives real upload progress (fetch has none).
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/v1/datasets/${ds.id}/uploads`);
    xhr.upload.onprogress = (e) => e.lengthComputable && setProgress(Math.round((e.loaded / e.total) * 100));
    xhr.onload = () => {
      setProgress(null);
      const body = JSON.parse(xhr.responseText);
      if (xhr.status >= 400) { setErr(body.detail?.message ?? `Lỗi ${xhr.status}`); return; }
      setErr(null);
      setReport(`Thêm ${body.added}, trùng ${body.duplicates}, từ chối ${body.rejected.length}` +
        (body.rejected.length ? ": " + body.rejected.slice(0, 5).map((r: { file: string; code: string; message: string }) => `${r.file} (${r.code}: ${r.message})`).join("; ") +
        (body.rejected.length > 5 ? ` … và ${body.rejected.length - 5} file khác` : "") : ""));
      refresh();
    };
    xhr.onerror = () => { setProgress(null); setErr("Mất kết nối khi upload."); };
    xhr.send(form);
  }

  const patch = useMutation({
    mutationFn: ({ id, body }: { id: string; body: Partial<DatasetItem> }) => api.patch(`/dataset-items/${id}`, body),
    onSuccess: refresh,
  });
  const approveAll = useMutation({
    mutationFn: () => api.post(`/datasets/${ds.id}/items/bulk`, {
      item_ids: items.data!.items.filter((i) => i.status === "pending").map((i) => i.id), patch: { status: "approved" },
      caption_prefix: ds.trigger_token,
    }),
    onSuccess: refresh,
  });
  const version = useMutation({
    mutationFn: () => api.post(`/datasets/${ds.id}/versions`),
    onSuccess: refresh,
    onError: (e: unknown) => setErr(e instanceof ApiError ? e.message : String(e)),
  });

  return (
    <div className="space-y-4">
      <div className="card flex flex-wrap items-end gap-3">
        <input ref={file} type="file" multiple accept=".jpg,.jpeg,.png,.webp,.txt,.zip" className="text-sm" />
        <div><label className="label">Quyền sử dụng / license (bắt buộc)</label><input className="input" value={license} onChange={(e) => setLicense(e.target.value)} placeholder="Ảnh tự tạo / có giấy phép…" /></div>
        <div><label className="label">Nguồn</label><input className="input" value={source} onChange={(e) => setSource(e.target.value)} /></div>
        <button className="btn" disabled={!license || progress !== null} onClick={upload}>{progress !== null ? `Đang tải ${progress}%` : "Upload"}</button>
      </div>
      {report && <p className="text-sm text-emerald-700">{report}</p>}
      {err && <p className="text-sm text-red-600">{err}</p>}
      <div className="flex flex-wrap gap-2">
        <button className="btn-ghost" onClick={() => approveAll.mutate()} disabled={!items.data?.items.some((i) => i.status === "pending")}>Duyệt tất cả ảnh chờ (gắn trigger vào caption)</button>
        <button className="btn" onClick={() => version.mutate()} disabled={ds.approved_count === 0}>Tạo dataset version (bất biến)</button>
      </div>
      {ds.versions.map((v) => (
        <div key={v.id} className="rounded border bg-white p-2 text-sm">
          <b>v{v.version}</b> — {v.item_count} ảnh <span className="font-mono text-xs text-slate-500">{v.id}</span>
          {v.warnings.map((w) => <div key={w} className="text-xs text-amber-700">⚠ {w}</div>)}
        </div>
      ))}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {items.data?.items.map((it) => (
          <div key={it.id} className={`card space-y-2 ${it.status === "rejected" ? "opacity-50" : ""}`}>
            <img src={assetUrl(it.asset_id)} className="h-40 w-full rounded object-cover" loading="lazy" />
            <textarea className="input h-16" defaultValue={it.caption} onBlur={(e) => e.target.value !== it.caption && patch.mutate({ id: it.id, body: { caption: e.target.value } })} />
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="rounded bg-slate-100 px-1">{it.status}</span>
              {it.quality_flags.map((f) => <span key={f} className="rounded bg-amber-100 px-1 text-amber-800">{f}</span>)}
              <button className="text-emerald-700 underline" onClick={() => patch.mutate({ id: it.id, body: { status: "approved" } })}>Duyệt</button>
              <button className="text-red-700 underline" onClick={() => patch.mutate({ id: it.id, body: { status: "rejected" } })}>Loại</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
