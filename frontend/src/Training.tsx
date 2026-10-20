import { useState } from "react";
import { useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { thumbUrl, assetUrl } from "./api";
import { listDatasets, listVersions } from "./datasetApi";
import {
  getMetrics, getRun, listModels, listRuns, preflight, reviewModel, runAction, startRun,
  type Preflight, type RunRequest, type TrainConfig,
} from "./trainingApi";
import { Alert, Badge, Card, Empty, Field, Icon, PageHead, Segmented, Spinner, type Tone } from "./ui";

const err = (e: unknown) => {
  const m = (e as Error).message;
  try { const j = JSON.parse(m); return j.message ?? j.code ?? m; } catch { return m; }
};
const ACTIVE = ["queued", "preparing", "running", "validating"];
const STATE: Record<string, { text: string; tone: Tone }> = {
  queued: { text: "Đang chờ", tone: "info" }, preparing: { text: "Chuẩn bị dữ liệu", tone: "primary" },
  running: { text: "Đang train", tone: "primary" }, validating: { text: "Validation", tone: "primary" },
  awaiting_review: { text: "Chờ duyệt", tone: "warn" }, approved: { text: "Đã duyệt", tone: "ok" },
  rejected: { text: "Bị loại", tone: "bad" }, failed: { text: "Thất bại", tone: "bad" }, cancelled: { text: "Đã hủy", tone: "" },
};
const MODEL_TONE: Record<string, Tone> = { approved: "ok", awaiting_review: "warn", rejected: "bad", verified: "ok", experimental: "info", unavailable: "bad" };
const PF_ICON = { ok: "check", warn: "alert", fail: "x" } as const;

// Preset thăm dò cho GPU ~8GB: KHÔNG phải cấu hình tối ưu đã chứng minh.
const PRESET_8GB: TrainConfig = {
  resolution: 512, rank: 16, learning_rate: 1e-4, max_train_steps: 30, train_batch_size: 1,
  gradient_accumulation_steps: 4, checkpointing_steps: 10, mixed_precision: "fp16",
  gradient_checkpointing: true, use_vae_fp16_fix: true, seed: 42,
};

export function Training() {
  return (
    <>
      <PageHead title="Training" sub="Train Character LoRA cho SDXL từ dataset version đã khóa, theo dõi loss thật, duyệt LoRA rồi dùng ở trang Tạo ảnh." />
      <div className="split">
        <NewRun />
        <div className="stack">
          <Runs />
          <Registry />
        </div>
      </div>
    </>
  );
}

function NewRun() {
  const qc = useQueryClient();
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: listDatasets });
  const bases = useQuery({ queryKey: ["models", "base"], queryFn: () => listModels("base") });
  const [dsId, setDsId] = useState("");
  const versions = useQuery({ queryKey: ["versions", dsId], queryFn: () => listVersions(dsId), enabled: !!dsId });
  const [versionId, setVersionId] = useState("");
  // Gộp thêm nhóm dataset: datasetId -> versionId đã chọn (chỉ các dataset được tick)
  const [extra, setExtra] = useState<Record<string, string>>({});
  const withVersions = (datasets.data ?? []).filter((d) => d.latest_version > 0);
  const allVersions = useQueries({ queries: withVersions.map((d) => ({ queryKey: ["versions", d.id], queryFn: () => listVersions(d.id) })) });
  const versionsOf = (id: string) => allVersions[withVersions.findIndex((d) => d.id === id)]?.data ?? [];
  const primary = withVersions.find((d) => d.id === dsId);
  const others = withVersions.filter((d) => d.id !== dsId);
  const setExtraFor = (id: string, on: boolean, v?: string) => {
    setPf(null);
    setExtra((cur) => {
      const n = { ...cur };
      if (on) n[id] = v ?? versionsOf(id)[0]?.id ?? ""; else delete n[id];
      return n;
    });
  };
  const sameCharacter = primary?.character_id ? others.filter((d) => d.character_id === primary.character_id) : [];
  const [baseId, setBaseId] = useState("");
  const [trigger, setTrigger] = useState("emv_a02");
  const [objective, setObjective] = useState<"character" | "style">("character");
  const [cfg, setCfg] = useState<TrainConfig>(PRESET_8GB);
  const [pf, setPf] = useState<Preflight | null>(null);
  const [mode, setMode] = useState<"steps" | "epochs">("steps");
  const num = (k: keyof TrainConfig) => (e: React.ChangeEvent<HTMLInputElement>) => { setPf(null); setCfg({ ...cfg, [k]: +e.target.value }); };

  const req = (): RunRequest => ({
    dataset_version_id: versionId,
    extra_dataset_version_ids: Object.entries(extra).filter(([d, v]) => d !== dsId && v).map(([, v]) => v),
    base_model_version_id: baseId || bases.data?.[0]?.id || "",
    trigger_token: trigger, objective,
    config: mode === "steps" ? { ...cfg, num_train_epochs: undefined, max_train_steps: cfg.max_train_steps ?? 30 }
      : { ...cfg, max_train_steps: undefined, num_train_epochs: cfg.num_train_epochs ?? 10 },
  });
  const check = useMutation({ mutationFn: () => preflight(req()), onSuccess: setPf });
  const start = useMutation({
    mutationFn: () => startRun(req()),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["runs"] }); setPf(null); },
    onError: (e) => { try { const j = JSON.parse((e as Error).message); if (j.preflight) setPf(j.preflight); } catch { /* lỗi thường */ } },
  });
  const ready = !!versionId && trigger.length >= 3;

  return (
    <div className="stack">
      <Card title={`Train ${objective === "style" ? "Style" : "Character"} LoRA (SDXL)`} sub="Chọn dữ liệu, kiểm tra GPU, rồi bắt đầu.">
        <div className="stack">
          <Field label="Mục tiêu" hint={objective === "style" ? "Học phong cách/khái niệm chung từ nhiều đối tượng (vd FFHQ, Unsplash, COCO) — không học một nhân vật cụ thể." : "Học một nhân vật cụ thể từ ảnh của đúng nhân vật đó."}>
            <Segmented label="Mục tiêu" value={objective} options={[{ value: "character", label: "Character (một nhân vật)" }, { value: "style", label: "Style / khái niệm" }]}
              onChange={(v) => { setPf(null); setObjective(v); setTrigger(v === "style" ? "emv_style" : "emv_a02"); }} />
          </Field>
          <Field label="Dataset">
            <select value={dsId} onChange={(e) => { setDsId(e.target.value); setVersionId(""); setExtra({}); setPf(null); }}>
              <option value="">Chọn dataset…</option>
              {datasets.data?.filter((d) => d.latest_version > 0).map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
          </Field>
          <div className="cols-2">
            <Field label="Version bất biến">
              <select value={versionId} onChange={(e) => { setVersionId(e.target.value); setPf(null); }} disabled={!dsId}>
                <option value="">Chọn version…</option>
                {versions.data?.map((v) => <option key={v.id} value={v.id}>v{v.version} · {v.item_count} ảnh · {v.manifest_hash.slice(0, 8)}</option>)}
              </select>
            </Field>
            <Field label="Trigger token" hint="Chữ, số, gạch dưới">
              <input value={trigger} onChange={(e) => { setPf(null); setTrigger(e.target.value.replace(/[^A-Za-z0-9_]/g, "")); }} />
            </Field>
          </div>
          <Field label="Base model">
            <select value={baseId} onChange={(e) => setBaseId(e.target.value)}>
              {bases.data?.map((b) => <option key={b.id} value={b.id}>{b.name} [{b.status}]</option>)}
            </select>
          </Field>
        </div>
      </Card>

      <Card title="Gộp thêm nhóm dataset" sub="Train một LoRA từ nhiều nhóm (khuôn mặt, toàn thân, biểu cảm…). Mỗi nhóm dùng một version bất biến."
        actions={sameCharacter.length > 0 && (
          <button type="button" className="btn sm" onClick={() => { setPf(null); setExtra(Object.fromEntries(sameCharacter.map((d) => [d.id, versionsOf(d.id)[0]?.id ?? ""]))); }}>
            Chọn {sameCharacter.length} nhóm cùng nhân vật
          </button>
        )}>
        {!primary && <small>Chọn dataset chính ở trên trước.</small>}
        {primary && others.length === 0 && <small>Chưa có dataset nào khác có version.</small>}
        <div className="stack-sm">
          {primary && others.map((d) => {
            const on = d.id in extra;
            const vs = versionsOf(d.id);
            return (
              <div key={d.id} className="row" style={{ gap: 10 }}>
                <label className="check" style={{ flex: 1, minWidth: 0 }}>
                  <input type="checkbox" checked={on} onChange={(e) => setExtraFor(d.id, e.target.checked)} />
                  <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{d.name}</span>
                  <Badge tone="ok">{d.counts.approved ?? 0}</Badge>
                </label>
                <select style={{ width: 170 }} disabled={!on} aria-label={`Version của ${d.name}`} value={extra[d.id] ?? ""}
                  onChange={(e) => setExtraFor(d.id, true, e.target.value)}>
                  {on ? vs.map((v) => <option key={v.id} value={v.id}>v{v.version} · {v.item_count} ảnh</option>) : <option value="">—</option>}
                </select>
              </div>
            );
          })}
        </div>
        {Object.keys(extra).length > 0 && <small style={{ display: "block", marginTop: 10 }}>Sẽ gộp {1 + Object.keys(extra).length} dataset version (cả ảnh trùng giữa các nhóm sẽ bị cảnh báo ở preflight).</small>}
      </Card>

      <Card title="Cấu hình" actions={<button type="button" className="btn sm" onClick={() => { setCfg(PRESET_8GB); setPf(null); }}>Preset thăm dò 8GB</button>}>
        <div className="stack">
          <Field label="Độ phân giải">
            <Segmented label="Độ phân giải" value={String(cfg.resolution)} options={[{ value: "512", label: "512" }, { value: "768", label: "768" }, { value: "1024", label: "1024" }]}
              onChange={(v) => { setPf(null); setCfg({ ...cfg, resolution: +v as TrainConfig["resolution"] }); }} />
          </Field>
          <div className="cols-3">
            <Field label="Rank"><input type="number" min={4} max={64} value={cfg.rank} onChange={num("rank")} /></Field>
            <Field label="Learning rate"><input type="number" step="0.00001" value={cfg.learning_rate} onChange={num("learning_rate")} /></Field>
            <Field label="Seed"><input type="number" value={cfg.seed} onChange={num("seed")} /></Field>
          </div>
          <div className="cols-3">
            <Field label="Độ dài">
              <select value={mode} onChange={(e) => { setPf(null); setMode(e.target.value as "steps" | "epochs"); }}><option value="steps">Số bước</option><option value="epochs">Số epoch</option></select>
            </Field>
            {mode === "steps"
              ? <Field label="Max steps"><input type="number" min={1} value={cfg.max_train_steps ?? 30} onChange={num("max_train_steps")} /></Field>
              : <Field label="Epochs"><input type="number" min={1} value={cfg.num_train_epochs ?? 10} onChange={num("num_train_epochs")} /></Field>}
            <Field label="Checkpoint mỗi"><input type="number" min={5} value={cfg.checkpointing_steps} onChange={num("checkpointing_steps")} /></Field>
          </div>
          <div className="cols-3">
            <Field label="Batch"><input type="number" min={1} max={4} value={cfg.train_batch_size} onChange={num("train_batch_size")} /></Field>
            <Field label="Grad accum"><input type="number" min={1} max={16} value={cfg.gradient_accumulation_steps} onChange={num("gradient_accumulation_steps")} /></Field>
            <Field label="Precision">
              <select value={cfg.mixed_precision} onChange={(e) => { setPf(null); setCfg({ ...cfg, mixed_precision: e.target.value as TrainConfig["mixed_precision"] }); }}><option>fp16</option><option>bf16</option><option>no</option></select>
            </Field>
          </div>
          <div className="row" style={{ gap: 18 }}>
            <label className="check"><input type="checkbox" checked={cfg.gradient_checkpointing} onChange={(e) => { setPf(null); setCfg({ ...cfg, gradient_checkpointing: e.target.checked }); }} />Gradient checkpointing</label>
            <label className="check"><input type="checkbox" checked={cfg.use_vae_fp16_fix} onChange={(e) => { setPf(null); setCfg({ ...cfg, use_vae_fp16_fix: e.target.checked }); }} />VAE fp16-fix</label>
          </div>
          <Alert tone="warn">Preset chỉ là cấu hình thăm dò, chưa tối ưu. Với 8GB VRAM, train SDXL ở 1024px không được cam kết; nếu hết VRAM hãy giảm resolution.</Alert>
        </div>
      </Card>

      <Card title="Kiểm tra & bắt đầu" actions={pf && <Badge tone={pf.overall === "ok" ? "ok" : pf.overall === "warn" ? "warn" : "bad"}>Preflight {pf.overall.toUpperCase()} · {pf.resolved_steps} bước</Badge>}>
        <div className="stack">
          <div className="row">
            <button className="btn" disabled={!ready || check.isPending} onClick={() => check.mutate()}>{check.isPending ? <><Spinner /> Đang hỏi worker GPU…</> : <><Icon name="cpu" className="sm" />Chạy preflight</>}</button>
            <button className="btn primary" disabled={!ready || !pf || pf.overall === "fail" || start.isPending} onClick={() => start.mutate()}><Icon name="play" className="sm" />Bắt đầu train</button>
          </div>
          {(check.error || start.error) && !pf && <Alert tone="bad">{err(check.error ?? start.error)}</Alert>}
          {pf && (
            <div className="check-list">
              {pf.checks.map((c) => (
                <div key={c.name} className={`check-item ${c.status}`}>
                  <Icon name={PF_ICON[c.status]} className="sm" />
                  <div><b>{c.name.replace(/_/g, " ")}</b> — {c.message}</div>
                </div>
              ))}
            </div>
          )}
          {!pf && <small>Preflight hỏi worker GPU thật để lấy VRAM, kiểm tra dữ liệu, trainer và dung lượng đĩa.</small>}
        </div>
      </Card>
    </div>
  );
}

function Runs() {
  const runs = useQuery({ queryKey: ["runs"], queryFn: listRuns, refetchInterval: (q) => (q.state.data?.some((r) => ACTIVE.includes(r.state)) ? 3000 : false) });
  const [open, setOpen] = useState<string | null>(null);
  return (
    <Card title="Lần train" sub={runs.data ? `${runs.data.length} lần gần nhất` : undefined}>
      {runs.data?.length === 0 && <Empty icon="cpu" title="Chưa có lần train nào">Chạy preflight rồi bấm “Bắt đầu train”.</Empty>}
      <div className="stack">
        {runs.data?.map((r) => {
          const st = STATE[r.state] ?? { text: r.state, tone: "" as Tone };
          const pct = r.max_steps ? Math.round((r.step / r.max_steps) * 100) : 0;
          return (
            <div key={r.id} className="run">
              <button className="run-head" onClick={() => setOpen(open === r.id ? null : r.id)} aria-expanded={open === r.id}>
                <span className="stack-sm" style={{ minWidth: 0 }}>
                  <span className="row" style={{ gap: 8 }}><strong>{r.trigger_token}</strong><code>{r.id.slice(0, 8)}</code></span>
                  {ACTIVE.includes(r.state) && (
                    <span className="stack-sm"><span className="progress" aria-label="Tiến độ"><i style={{ width: `${pct}%` }} /></span><small>{r.step}/{r.max_steps} bước · {pct}%{r.phase ? ` · ${r.phase}` : ""}</small></span>
                  )}
                </span>
                <Badge tone={st.tone}>{ACTIVE.includes(r.state) && <Spinner />}{st.text}</Badge>
              </button>
              {open === r.id && <RunDetail id={r.id} />}
            </div>
          );
        })}
      </div>
    </Card>
  );
}

function RunDetail({ id }: { id: string }) {
  const qc = useQueryClient();
  const run = useQuery({ queryKey: ["run", id], queryFn: () => getRun(id), refetchInterval: (q) => (q.state.data && ACTIVE.includes(q.state.data.state) ? 3000 : false) });
  const metrics = useQuery({ queryKey: ["metrics", id], queryFn: () => getMetrics(id), refetchInterval: run.data && ACTIVE.includes(run.data.state) ? 3000 : false });
  const act = useMutation({ mutationFn: (a: "cancel" | "resume" | "validate") => runAction(id, a), onSuccess: () => { qc.invalidateQueries({ queryKey: ["runs"] }); qc.invalidateQueries({ queryKey: ["run", id] }); } });
  const r = run.data;
  if (!r) return <div className="run-body"><div className="row"><Spinner /> Đang tải…</div></div>;
  const m = metrics.data ?? [];
  return (
    <div className="run-body">
      <dl className="kv" style={{ marginTop: 12 }}>
        <dt>Trainer</dt><dd>{r.trainer_revision ?? "—"}</dd>
        <dt>Base model</dt><dd><code>{r.base_revision?.slice(0, 10) ?? "—"}</code></dd>
        <dt>Dataset hash</dt><dd><code>{r.manifest_hash?.slice(0, 12) ?? "—"}</code></dd>
        <dt>Cấu hình</dt><dd>{r.config.resolution}px · rank {r.config.rank} · lr {r.config.learning_rate} · batch {r.config.train_batch_size}×{r.config.gradient_accumulation_steps} · {r.config.mixed_precision}</dd>
      </dl>
      {r.error && <Alert tone="bad">{r.error}</Alert>}
      <LossChart points={m} />
      <div className="row">
        {ACTIVE.includes(r.state) && <button className="btn sm danger" onClick={() => act.mutate("cancel")}><Icon name="x" className="sm" />Hủy</button>}
        {["failed", "cancelled"].includes(r.state) && <button className="btn sm" onClick={() => act.mutate("resume")}><Icon name="refresh" className="sm" />Resume từ checkpoint</button>}
        {r.state === "awaiting_review" && <button className="btn sm" onClick={() => act.mutate("validate")}><Icon name="refresh" className="sm" />Chạy lại ảnh validation</button>}
      </div>
      {act.error && <Alert tone="bad">{err(act.error)}</Alert>}
      {r.validation_asset_ids.length > 0 && (
        <div className="stack-sm">
          <span className="section-label">Ảnh validation (LoRA vừa train)</span>
          <div className="row">{r.validation_asset_ids.map((a) => <a key={a} href={assetUrl(a)} target="_blank" rel="noreferrer"><img src={thumbUrl(a)} alt="validation" width={150} style={{ borderRadius: 10, display: "block" }} /></a>)}</div>
        </div>
      )}
      <details><summary>Log (đã che secret)</summary><pre>{r.log_tail || "(trống)"}</pre></details>
    </div>
  );
}

function LossChart({ points }: { points: { step: number; loss: number }[] }) {
  if (points.length < 2) return <small>Chưa đủ điểm loss ({points.length}).</small>;
  const W = 520, H = 120, P = 6;
  const maxS = Math.max(...points.map((p) => p.step)), maxL = Math.max(...points.map((p) => p.loss), 1e-6);
  const xy = points.map((p) => [P + (p.step / maxS) * (W - 2 * P), H - P - (p.loss / maxL) * (H - 2 * P)] as const);
  const line = xy.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const area = `${line} L${xy[xy.length - 1][0].toFixed(1)},${H - P} L${xy[0][0].toFixed(1)},${H - P} Z`;
  return (
    <figure style={{ margin: 0 }} className="stack-sm">
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Đường loss theo bước">
        <defs><linearGradient id="lossg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="var(--primary)" stopOpacity=".28" /><stop offset="1" stopColor="var(--primary)" stopOpacity="0" /></linearGradient></defs>
        <path d={area} fill="url(#lossg)" />
        <path d={line} fill="none" stroke="var(--primary)" strokeWidth="1.8" strokeLinejoin="round" />
      </svg>
      <figcaption><small>Loss thật theo bước · {points.length} điểm · cao nhất {maxL.toFixed(3)} · cuối {points[points.length - 1].loss.toFixed(3)}. Loss nhiễu theo bước, không phản ánh chất lượng nhận dạng.</small></figcaption>
    </figure>
  );
}

function Registry() {
  const qc = useQueryClient();
  const models = useQuery({ queryKey: ["models", "all"], queryFn: () => listModels() });
  const [note, setNote] = useState("");
  const review = useMutation({
    mutationFn: (v: { id: string; d: "approved" | "rejected" }) => reviewModel(v.id, v.d, note),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["models"] }); qc.invalidateQueries({ queryKey: ["runs"] }); },
  });
  return (
    <Card title="Model registry" sub="Base model, LoRA và adapter tham chiếu. Chỉ LoRA đã duyệt dùng cho sản xuất.">
      <div className="stack">
        <Field label="Ghi chú khi duyệt (tùy chọn)"><input value={note} onChange={(e) => setNote(e.target.value)} placeholder="vd: đạt nhận dạng trên 6 cảnh chưa học" /></Field>
        {models.data?.map((m) => (
          <div key={m.id} className="model-card">
            <div className="row">
              <Icon name="layers" className="sm" />
              <strong>{m.name}</strong>
              <span className="spacer" />
              <Badge>{m.kind.replace("_", " ")}</Badge>
              <Badge tone={MODEL_TONE[m.status] ?? ""}>{m.status.replace("_", " ")}</Badge>
              {m.trainable && <Badge tone="primary">trainable</Badge>}
            </div>
            <dl className="kv">
              {m.model_id && <><dt>Repo</dt><dd>{m.model_id}</dd></>}
              <dt>Revision</dt><dd><code>{m.revision?.slice(0, 10) ?? "—"}</code></dd>
              {m.sha256 && <><dt>sha256</dt><dd><code>{m.sha256.slice(0, 16)}</code></dd></>}
              {m.trigger_token && <><dt>Trigger</dt><dd><code>{m.trigger_token}</code></dd></>}
              {m.license && <><dt>License</dt><dd>{m.license}</dd></>}
              {m.review_note && <><dt>Ghi chú</dt><dd>{m.review_note}</dd></>}
            </dl>
            {m.kind !== "base" && m.kind !== "ip_adapter" && m.status === "awaiting_review" && (
              <div className="row">
                <button className="btn sm primary" onClick={() => review.mutate({ id: m.id, d: "approved" })}><Icon name="check" className="sm" />Duyệt</button>
                <button className="btn sm danger" onClick={() => review.mutate({ id: m.id, d: "rejected" })}><Icon name="x" className="sm" />Loại</button>
              </div>
            )}
          </div>
        ))}
        {review.error && <Alert tone="bad">{err(review.error)}</Alert>}
      </div>
    </Card>
  );
}
