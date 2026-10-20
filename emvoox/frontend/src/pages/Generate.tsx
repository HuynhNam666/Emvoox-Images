import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ApiError, Character, Job, ModelVersion, api, assetUrl } from "../api";

const PRESETS: Record<string, [string, string][]> = {
  hair: [["toc_den_ngang_vai", "Tóc đen ngang vai"], ["toc_den_dai", "Tóc đen dài"], ["toc_ngan", "Tóc ngắn"], ["toc_xoan", "Tóc xoăn"]],
  emotion: [["vui", "Vui"], ["buon", "Buồn"], ["tu_tin", "Tự tin"], ["ngac_nhien", "Ngạc nhiên"], ["binh_than", "Bình thản"]],
  setting: [["thu_vien", "Thư viện"], ["quan_ca_phe", "Quán cà phê"], ["pho_phuong", "Phố"], ["van_phong", "Văn phòng"], ["cong_vien", "Công viên"]],
  camera: [["can_canh", "Cận cảnh"], ["ban_than", "Bán thân"], ["toan_than", "Toàn thân"], ["goc_thap", "Góc thấp"]],
  lighting: [["anh_sang_cua_so", "Ánh sáng cửa sổ"], ["hoang_hon", "Hoàng hôn"], ["studio", "Studio"], ["tu_nhien", "Tự nhiên"]],
  style: [["dien_anh", "Điện ảnh chân thực"], ["anime", "Anime"], ["hoat_hinh_3d", "Hoạt hình 3D"]],
};
const LABELS: Record<string, string> = { hair: "Tóc", emotion: "Cảm xúc", setting: "Bối cảnh", camera: "Góc máy", lighting: "Ánh sáng", style: "Phong cách" };
const ACTIVE = new Set(["queued", "running"]);

export default function Generate() {
  const qc = useQueryClient();
  const models = useQuery({ queryKey: ["models"], queryFn: () => api.get<ModelVersion[]>("/models") });
  const chars = useQuery({ queryKey: ["characters"], queryFn: () => api.get<Character[]>("/characters") });
  const bases = (models.data ?? []).filter((m) => m.kind === "base" && m.status !== "unavailable");
  const [original, setOriginal] = useState("");
  const [fields, setFields] = useState<Record<string, string>>({});
  const [descEn, setDescEn] = useState("");
  const [compiled, setCompiled] = useState("");
  const [negative, setNegative] = useState("");
  const [warnings, setWarnings] = useState<string[]>([]);
  const [modelId, setModelId] = useState("");
  const [loraId, setLoraId] = useState("");
  const [charId, setCharId] = useState("");
  const [aspect, setAspect] = useState("9:16");
  const [seed, setSeed] = useState(42);
  const [steps, setSteps] = useState(28);
  const [guidance, setGuidance] = useState(6);
  const [loraScale, setLoraScale] = useState(1);
  const [n, setN] = useState(1);
  const [jobId, setJobId] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);

  const character = chars.data?.find((c) => c.id === charId);
  const trigger = character?.current_version?.trigger_token ?? null;
  const activeModel = modelId || bases[0]?.id || "";
  const loras = (models.data ?? []).filter((m) => m.kind.endsWith("_lora") && m.review_status === "approved" && m.parent_base_id === activeModel);

  const compile = useMutation({
    mutationFn: () => api.post<{ prompt: string; negative_prompt: string; warnings: string[] }>("/prompt/compile",
      { fields, description_en: descEn || null, trigger_token: trigger }),
    onSuccess: (r) => { setCompiled(r.prompt); setNegative(r.negative_prompt); setWarnings(r.warnings); },
  });

  const submit = useMutation({
    // One key per click; the button is disabled while pending so a double-click cannot duplicate the job.
    mutationFn: () => api.post<Job>("/generation-jobs", {
      character_version_id: character?.current_version?.id ?? null,
      prompt_original: original || compiled, prompt_compiled: compiled || null, negative_prompt: negative || null,
      model_version_id: activeModel, lora_version_id: loraId || null, aspect_ratio: aspect, num_images: n, seed,
      parameters: { steps, guidance_scale: guidance, lora_scale: loraScale },
    }, { "Idempotency-Key": crypto.randomUUID() }),
    onSuccess: (j) => { setJobId(j.id); setMsg(null); },
    onError: (e: unknown) => setMsg(e instanceof ApiError ? e.message : String(e)),
  });

  const job = useQuery({
    queryKey: ["job", jobId], enabled: !!jobId,
    queryFn: () => api.get<Job>(`/generation-jobs/${jobId}`),
    refetchInterval: (q) => (q.state.data && !ACTIVE.has(q.state.data.state) ? false : 2000),
  });
  useEffect(() => { if (job.data && !ACTIVE.has(job.data.state)) qc.invalidateQueries({ queryKey: ["dashboard"] }); }, [job.data?.state]);

  const [saveCode, setSaveCode] = useState("");
  const [saveName, setSaveName] = useState("");
  const save = useMutation({
    mutationFn: (assetId: string) => api.post<Character>("/characters", {
      code: saveCode, name: saveName || saveCode, identity_spec: { prompt: compiled || original, fields, seed },
      trigger_token: null, canonical_asset_id: assetId, reference_asset_ids: [assetId],
    }),
    onSuccess: () => { setMsg("Đã lưu nhân vật và ảnh chuẩn."); qc.invalidateQueries({ queryKey: ["characters"] }); },
    onError: (e: unknown) => setMsg(e instanceof ApiError ? e.message : String(e)),
  });

  const j = job.data;
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <section className="space-y-4">
        <h1 className="text-xl font-semibold">Tạo ảnh nhân vật</h1>
        <div className="card space-y-3">
          <div>
            <label className="label">Mô tả nhân vật (tiếng Việt, được lưu nguyên văn)</label>
            <textarea className="input h-24" value={original} onChange={(e) => setOriginal(e.target.value)}
              placeholder="Nhân vật nữ Việt Nam 25 tuổi, tóc đen ngang vai, áo trắng, đứng trong thư viện…" />
          </div>
          <div className="grid grid-cols-2 gap-3">
            {Object.entries(PRESETS).map(([k, opts]) => (
              <div key={k}>
                <label className="label">{LABELS[k]}</label>
                <select className="input" value={fields[k] ?? ""} onChange={(e) => setFields({ ...fields, [k]: e.target.value })}>
                  <option value="">—</option>
                  {opts.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                </select>
              </div>
            ))}
            <div>
              <label className="label">Tuổi biểu kiến</label>
              <input className="input" type="number" min={1} max={120} value={fields.age ?? ""}
                onChange={(e) => setFields({ ...fields, age: e.target.value })} />
            </div>
            <div>
              <label className="label">Trang phục (tiếng Anh)</label>
              <input className="input" value={fields.outfit ?? ""} onChange={(e) => setFields({ ...fields, outfit: e.target.value })} placeholder="white shirt" />
            </div>
          </div>
          <div>
            <label className="label">Ghi chú bổ sung bằng tiếng Anh (tùy chọn)</label>
            <input className="input" value={descEn} onChange={(e) => setDescEn(e.target.value)} />
          </div>
          <button className="btn-ghost" onClick={() => compile.mutate()} disabled={compile.isPending}>Biên dịch prompt (template)</button>
          <p className="text-xs text-slate-500">Biên dịch dùng preset song ngữ, không tự dịch tiếng Việt tự do. Hãy sửa prompt tiếng Anh bên dưới nếu cần.</p>
          {warnings.map((w) => <p key={w} className="text-xs text-amber-700">⚠ {w}</p>)}
          <div>
            <label className="label">Prompt gửi model (có thể sửa)</label>
            <textarea className="input h-20" value={compiled} onChange={(e) => setCompiled(e.target.value)} />
          </div>
          <div>
            <label className="label">Negative prompt</label>
            <input className="input" value={negative} onChange={(e) => setNegative(e.target.value)} />
          </div>
        </div>

        <div className="card grid grid-cols-2 gap-3">
          <div>
            <label className="label">Model nền</label>
            <select className="input" value={activeModel} onChange={(e) => { setModelId(e.target.value); setLoraId(""); }}>
              {bases.map((m) => <option key={m.id} value={m.id}>{m.name} ({m.status})</option>)}
            </select>
          </div>
          <div>
            <label className="label">LoRA đã duyệt (tương thích)</label>
            <select className="input" value={loraId} onChange={(e) => setLoraId(e.target.value)}>
              <option value="">Không dùng</option>
              {loras.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
            </select>
          </div>
          <div>
            <label className="label">Dùng nhân vật đã lưu</label>
            <select className="input" value={charId} onChange={(e) => setCharId(e.target.value)}>
              <option value="">Nhân vật mới</option>
              {(chars.data ?? []).map((c) => <option key={c.id} value={c.id}>{c.code} — {c.name} (v{c.current_version?.version})</option>)}
            </select>
          </div>
          <div>
            <label className="label">Tỉ lệ</label>
            <select className="input" value={aspect} onChange={(e) => setAspect(e.target.value)}>
              <option>9:16</option><option>1:1</option><option>16:9</option>
            </select>
          </div>
          <Num label="Seed" v={seed} set={setSeed} min={0} />
          <Num label="Số ảnh" v={n} set={setN} min={1} max={4} />
          <Num label="Steps" v={steps} set={setSteps} min={1} max={100} />
          <Num label="Guidance" v={guidance} set={setGuidance} min={0} max={20} step={0.5} />
          {loraId && <Num label="LoRA scale" v={loraScale} set={setLoraScale} min={0} max={2} step={0.1} />}
        </div>
        <button className="btn" disabled={submit.isPending || !(original || compiled) || !activeModel} onClick={() => submit.mutate()}>
          {submit.isPending ? "Đang gửi…" : "Tạo ảnh"}
        </button>
        {msg && <p className="text-sm text-red-600">{msg}</p>}
      </section>

      <section className="space-y-4">
        <h2 className="text-lg font-medium">Kết quả</h2>
        {!j && <p className="text-sm text-slate-500">Chưa có job. Ảnh sẽ hiện ở đây khi worker hoàn tất.</p>}
        {j && (
          <div className="card space-y-3">
            <div className="text-sm">
              Trạng thái: <b>{j.state}</b>
              {j.state === "running" && j.progress_total ? ` — bước ${j.progress_step}/${j.progress_total}` : ""}
              {j.state === "queued" && " — đang chờ worker"}
            </div>
            {j.error_message && <p className="text-sm text-red-600">{j.error_code}: {j.error_message}</p>}
            {ACTIVE.has(j.state) && <button className="btn-ghost" onClick={() => api.post(`/generation-jobs/${j.id}/cancel`)}>Hủy</button>}
            <div className="grid grid-cols-2 gap-3">
              {j.outputs.map((o) => (
                <figure key={o.asset_id} className="space-y-1">
                  <img src={assetUrl(o.asset_id, "content")} className="w-full rounded border" />
                  <figcaption className="flex flex-wrap gap-2 text-xs text-slate-600">
                    seed {o.seed}
                    <a className="text-indigo-600 underline" href={assetUrl(o.asset_id, "content")} download>Tải gốc</a>
                    <a className="text-indigo-600 underline" href={`${assetUrl(o.asset_id, "export")}?width=1080&height=1920`}>1080×1920</a>
                    <button className="text-indigo-600 underline" onClick={() => { if (saveCode) save.mutate(o.asset_id); else setMsg("Nhập mã nhân vật trước khi lưu."); }}>
                      Lưu làm ảnh chuẩn
                    </button>
                  </figcaption>
                </figure>
              ))}
            </div>
            {j.state === "review_required" && (
              <div className="flex gap-2">
                <button className="btn" onClick={() => api.post(`/generation-jobs/${j.id}/review`, { decision: "approved" }).then(() => job.refetch())}>Duyệt</button>
                <button className="btn-ghost" onClick={() => api.post(`/generation-jobs/${j.id}/review`, { decision: "rejected" }).then(() => job.refetch())}>Loại</button>
              </div>
            )}
            <div className="grid grid-cols-2 gap-2">
              <input className="input" placeholder="Mã nhân vật (A01)" value={saveCode} onChange={(e) => setSaveCode(e.target.value)} />
              <input className="input" placeholder="Tên nhân vật" value={saveName} onChange={(e) => setSaveName(e.target.value)} />
            </div>
            <details className="text-xs text-slate-600"><summary>Thông số thực tế</summary><pre>{JSON.stringify(j.config, null, 2)}</pre></details>
          </div>
        )}
      </section>
    </div>
  );
}

function Num({ label, v, set, min, max, step }: { label: string; v: number; set: (n: number) => void; min?: number; max?: number; step?: number }) {
  return (
    <div>
      <label className="label">{label}</label>
      <input className="input" type="number" value={v} min={min} max={max} step={step} onChange={(e) => set(Number(e.target.value))} />
    </div>
  );
}
