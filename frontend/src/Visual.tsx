import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { assetUrl, listCharacters, thumbUrl } from "./api";
import {
  createVisual, exportFileUrl, exportVisual, getVisual, listModels, listScenes, listVisual, reviewVisual,
  type Candidate, type VisualJob,
} from "./visualApi";
import { confirmDelete, deleteVisualJob } from "./manageApi";
import { Alert, Badge, Card, Empty, Field, Icon, PageHead, Segmented, Spinner, type Tone } from "./ui";

const STATE: Record<string, { text: string; tone: Tone }> = {
  queued: { text: "Đang chờ", tone: "info" }, running: { text: "Đang tạo", tone: "primary" }, review_required: { text: "Chờ duyệt", tone: "warn" },
  approved: { text: "Đã duyệt", tone: "ok" }, rejected: { text: "Bị loại", tone: "bad" }, failed: { text: "Thất bại", tone: "bad" },
  needs_human: { text: "Cần người xử lý", tone: "bad" },
};
const VERDICT: Record<string, { text: string; tone: Tone }> = {
  pass: { text: "QC đạt", tone: "ok" }, review: { text: "QC: cần xem", tone: "warn" }, fail: { text: "QC lỗi", tone: "bad" }, pending: { text: "chưa QC", tone: "" },
};
const ACTIVE = ["queued", "running"];
const err = (e: unknown) => (e as Error).message;
const n = (x: number | null | undefined, d = 1) => (x == null ? "—" : x.toFixed(d));
const usd = (x: number | null | undefined) => (x == null ? "—" : x.toFixed(4));
const REJECT_REASONS = ["Sai nhân vật", "Sai hành động cốt lõi", "Lỗi tay/hình thể", "Sai trang phục/bối cảnh", "Chất lượng kém", "Khác"];

export function Visual() {
  return (
    <>
      <PageHead title="Ảnh theo shot" sub="Pipeline theo báo cáo: Character/Scene Library → Prompt Compiler → Model Router → Render Worker → Quality Gate → người duyệt → xuất bản (nhãn AI + C2PA)." />
      <div className="split">
        <NewShot />
        <Jobs />
      </div>
    </>
  );
}

function NewShot() {
  const qc = useQueryClient();
  const chars = useQuery({ queryKey: ["characters"], queryFn: listCharacters });
  const scenes = useQuery({ queryKey: ["scenes"], queryFn: listScenes });
  const models = useQuery({ queryKey: ["vmodels"], queryFn: listModels });
  const [project, setProject] = useState("PILOT");
  const [episode, setEpisode] = useState("E01");
  const [shot, setShot] = useState("S01");
  const [c1, setC1] = useState("");
  const [c2, setC2] = useState("");
  const [scene, setScene] = useState("");
  const [action, setAction] = useState("looks back over the shoulder");
  const [emotion, setEmotion] = useState("calm");
  const [size, setSize] = useState("medium shot");
  const [angle, setAngle] = useState("eye level");
  const [light, setLight] = useState("soft window light from camera left");
  const [style, setStyle] = useState("cinematic realistic");
  const [aspect, setAspect] = useState<"9:16" | "16:9" | "1:1">("9:16");
  const [variants, setVariants] = useState(2);
  const [budget, setBudget] = useState(0);
  const [pin, setPin] = useState("");
  const [seed, setSeed] = useState(42);
  const [key, setKey] = useState(() => crypto.randomUUID());

  const sel = [c1, c2].filter(Boolean);
  const ver = (id: string) => chars.data?.find((c) => c.id === id)?.current_version?.id;
  const m = useMutation({
    mutationFn: () => createVisual({
      project_id: project, episode_id: episode || null, shot_id: shot,
      characters: sel.map((id, i) => ({ character_version_id: ver(id)!, position: sel.length > 1 ? (i === 0 ? "left" : "right") : null })),
      scene_code: scene || null,
      scene_spec: { action, emotion, camera: { size, angle }, lighting: light, style },
      aspect_ratio: aspect, num_variants: variants, seed, budget_limit: budget, model_key: pin || null,
    }, key),
    onSuccess: () => { setKey(crypto.randomUUID()); qc.invalidateQueries({ queryKey: ["visual"] }); },
  });
  const ok = shot.trim() && project.trim() && action.trim() && sel.every((id) => ver(id));

  return (
    <Card title="Shot mới" sub="Mã nhân vật/bối cảnh được thay bằng mô tả + ảnh chuẩn có phiên bản; phiên bản được ghim vào job.">
      <div className="stack">
        <div className="cols-3">
          <Field label="Project"><input value={project} onChange={(e) => setProject(e.target.value)} /></Field>
          <Field label="Tập"><input value={episode} onChange={(e) => setEpisode(e.target.value)} /></Field>
          <Field label="Shot"><input value={shot} onChange={(e) => setShot(e.target.value)} /></Field>
        </div>
        <div className="cols-2">
          <Field label="Nhân vật 1"><select value={c1} onChange={(e) => setC1(e.target.value)}><option value="">— không —</option>
            {chars.data?.map((c) => <option key={c.id} value={c.id}>{c.code} · {c.name} (v{c.current_version?.version})</option>)}</select></Field>
          <Field label="Nhân vật 2"><select value={c2} onChange={(e) => setC2(e.target.value)}><option value="">— không —</option>
            {chars.data?.filter((c) => c.id !== c1).map((c) => <option key={c.id} value={c.id}>{c.code} · {c.name} (v{c.current_version?.version})</option>)}</select></Field>
        </div>
        <Field label="Bối cảnh (Scene Library)"><select value={scene} onChange={(e) => setScene(e.target.value)}><option value="">— không —</option>
          {scenes.data?.map((s) => <option key={s.code} value={s.code}>{s.code} · {s.name} (v{s.current_version})</option>)}</select></Field>
        <Field label="Hành động"><textarea rows={2} value={action} onChange={(e) => setAction(e.target.value)} /></Field>
        <div className="cols-2">
          <Field label="Cảm xúc"><input value={emotion} onChange={(e) => setEmotion(e.target.value)} /></Field>
          <Field label="Ánh sáng"><input value={light} onChange={(e) => setLight(e.target.value)} /></Field>
          <Field label="Cỡ cảnh"><input value={size} onChange={(e) => setSize(e.target.value)} /></Field>
          <Field label="Góc máy"><input value={angle} onChange={(e) => setAngle(e.target.value)} /></Field>
        </div>
        <Field label="Phong cách"><input value={style} onChange={(e) => setStyle(e.target.value)} /></Field>
        <div className="cols-2">
          <Field label="Khung hình"><Segmented label="Khung hình" value={aspect} onChange={setAspect}
            options={[{ value: "9:16", label: "9:16" }, { value: "16:9", label: "16:9" }, { value: "1:1", label: "1:1" }]} /></Field>
          <Field label="Số phương án (2–4 theo báo cáo)"><input type="number" min={1} max={4} value={variants} onChange={(e) => setVariants(Number(e.target.value))} /></Field>
          <Field label="Trần chi phí (USD)" hint="0 = không giới hạn"><input type="number" min={0} step={0.01} value={budget} onChange={(e) => setBudget(Number(e.target.value))} /></Field>
          <Field label="Seed"><input type="number" min={0} value={seed} onChange={(e) => setSeed(Number(e.target.value))} /></Field>
        </div>
        <Field label="Ghim model" hint="Để trống: Model Router chọn tuyến rẻ nhất đủ năng lực, kèm dự phòng">
          <select value={pin} onChange={(e) => setPin(e.target.value)}><option value="">Tự chọn (Model Router)</option>
            {models.data?.models.filter((x) => x.route !== "manual").map((x) => (
              <option key={x.key} value={x.key} disabled={!x.available}>{x.name}{x.available ? "" : ` — ${x.availability}`}</option>))}</select>
        </Field>
        {m.error && <Alert tone="bad">{err(m.error)}</Alert>}
        {m.data && (
          <Alert tone="ok">Đã xếp hàng <code>{m.data.job_id.slice(0, 8)}</code>. Tuyến: {m.data.route_plan?.map((p) => p.model_key).join(" → ")}
            {m.data.excluded?.length ? ` · loại ${m.data.excluded.length} model (xem tab Model)` : ""}</Alert>
        )}
        <button className="btn primary lg block" disabled={!ok || m.isPending} onClick={() => m.mutate()}>
          {m.isPending ? <Spinner /> : <Icon name="sparkles" />}Tạo ảnh
        </button>
      </div>
    </Card>
  );
}

function Jobs() {
  const [open, setOpen] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const q = useQuery({
    queryKey: ["visual", filter], queryFn: () => listVisual(filter || undefined),
    refetchInterval: (query) => (query.state.data?.items.some((j) => ACTIVE.includes(j.state)) ? 3000 : 15000),
  });
  const items = q.data?.items ?? [];
  return (
    <Card title="Job ảnh" sub={`${q.data?.total ?? 0} job`}>
      <div className="stack">
        <div className="seg" role="group" aria-label="Lọc">
          {[["", "Tất cả"], ["review_required", "Chờ duyệt"], ["approved", "Đã duyệt"], ["needs_human", "Cần xử lý"], ["failed", "Lỗi"]].map(([v, l]) => (
            <button key={v} type="button" aria-pressed={filter === v} onClick={() => setFilter(v)}>{l}</button>))}
        </div>
        {q.error && <Alert tone="bad">{err(q.error)}</Alert>}
        {q.data && items.length === 0 && <Empty icon="image" title="Chưa có job">Tạo shot đầu tiên ở form bên cạnh.</Empty>}
        {items.map((j) => {
          const st = STATE[j.state] ?? { text: j.state, tone: "" as Tone };
          return (
            <div key={j.id} className="stack-sm">
              <button className="hcard" style={{ textAlign: "left" }} aria-expanded={open === j.id} onClick={() => setOpen(open === j.id ? null : j.id)}>
                {j.thumb_asset_id ? <img src={thumbUrl(j.thumb_asset_id)} alt="" loading="lazy" /> : <div className="ph">{ACTIVE.includes(j.state) ? <Spinner /> : <Icon name="image" />}</div>}
                <div className="hc-body">
                  <div className="row" style={{ gap: 6 }}>
                    <Badge tone={st.tone}>{st.text}</Badge><Badge>{j.project_id}/{j.shot_id}</Badge>
                    {j.model_key && <Badge tone="info">{j.model_key}</Badge>}
                    {j.repair_count > 0 && <Badge tone="warn">sửa {j.repair_count}/{j.max_repairs}</Badge>}
                  </div>
                  <div>chi phí {usd(j.cost_spent)}{j.budget_limit ? ` / ${j.budget_limit}` : ""}</div>
                </div>
              </button>
              {open === j.id && <Detail id={j.id} />}
            </div>
          );
        })}
      </div>
    </Card>
  );
}

function Detail({ id }: { id: string }) {
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ["visual-job", id], queryFn: () => getVisual(id), refetchInterval: (query) => (ACTIVE.includes(query.state.data?.state ?? "") ? 3000 : false) });
  const [reviewer, setReviewer] = useState(() => { try { return localStorage.getItem("emvoox.reviewer") ?? ""; } catch { return ""; } });
  const [note, setNote] = useState("");
  const [reason, setReason] = useState(REJECT_REASONS[0]);
  const [instruction, setInstruction] = useState("");
  const [pick, setPick] = useState<string | null>(null);
  const refresh = () => { qc.invalidateQueries({ queryKey: ["visual-job", id] }); qc.invalidateQueries({ queryKey: ["visual"] }); qc.invalidateQueries({ queryKey: ["approved-visual"] }); };
  const rv = useMutation({
    mutationFn: (b: Record<string, unknown>) => { try { localStorage.setItem("emvoox.reviewer", reviewer); } catch { /* bỏ qua */ } return reviewVisual(id, { reviewer, note, ...b }); },
    onSuccess: refresh,
  });
  const ex = useMutation({ mutationFn: () => exportVisual(id, { target: "auto", label: true }) });
  const del = useMutation({ mutationFn: () => deleteVisualJob(id), onSuccess: refresh });
  const j: VisualJob | undefined = q.data;
  if (!j) return <div className="row"><Spinner /> Đang tải…</div>;
  const cands = j.candidates ?? [];
  const chosen = pick ?? j.selected_candidate_id;
  const cur = cands.find((c) => c.id === chosen);
  const canReview = j.state === "review_required" || j.state === "needs_human";
  return (
    <div className="card flat"><div className="card-b stack">
      {j.error && <Alert tone={j.state === "failed" ? "bad" : "warn"}>{j.error}</Alert>}
      {!!j.continuity_warnings?.length && <Alert tone="warn"><strong>Liên tục:</strong> {j.continuity_warnings.join(" · ")}</Alert>}
      <div className="gallery" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))" }}>
        {cands.map((c) => <CandidateCard key={c.id} c={c} active={c.id === chosen} onPick={() => setPick(c.id)} />)}
      </div>
      {cur && <QcTable c={cur} />}
      {canReview && (
        <div className="stack-sm">
          <span className="section-label">Người phụ trách duyệt (bắt buộc ghi tên)</span>
          <div className="row">
            <input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="Tên người duyệt" aria-label="Tên người duyệt" style={{ width: 160 }} />
            <input value={note} onChange={(e) => setNote(e.target.value)} placeholder="Ghi chú" aria-label="Ghi chú" style={{ flex: 1, minWidth: 120 }} />
          </div>
          <div className="row">
            <button className="btn primary" disabled={!reviewer || !chosen || rv.isPending}
              onClick={() => rv.mutate({ decision: "approve", candidate_id: chosen, override_qc: cur?.qc_verdict === "fail" })}>
              <Icon name="check" />Duyệt ảnh này{cur?.qc_verdict === "fail" ? " (bỏ qua QC lỗi)" : ""}</button>
            <select value={reason} onChange={(e) => setReason(e.target.value)} aria-label="Lý do loại">{REJECT_REASONS.map((r) => <option key={r}>{r}</option>)}</select>
            <button className="btn danger" disabled={!reviewer || rv.isPending} onClick={() => rv.mutate({ decision: "reject", reason })}><Icon name="x" />Loại</button>
          </div>
          <div className="row">
            <input value={instruction} onChange={(e) => setInstruction(e.target.value)} placeholder="Chỉ dẫn sửa (tiếng Anh), vd: keep the same hairstyle, open the eyes" aria-label="Chỉ dẫn sửa" style={{ flex: 1, minWidth: 200 }} />
            <button className="btn" disabled={!reviewer || !instruction || rv.isPending} onClick={() => rv.mutate({ decision: "repair", instruction })}><Icon name="refresh" />Sửa</button>
          </div>
        </div>
      )}
      {j.state === "approved" && (
        <div className="stack-sm">
          <button className="btn" disabled={ex.isPending} onClick={() => ex.mutate()}><Icon name="download" />Xuất bản (1080×1920 + nhãn AI + C2PA)</button>
          {ex.error && <Alert tone="bad">{err(ex.error)}</Alert>}
          {ex.data && (
            <Alert tone={ex.data.status === "released" ? "ok" : "bad"}>
              {ex.data.status === "released" ? <>Đã xuất bản <a href={exportFileUrl(ex.data.id)} download>tải ảnh</a></> : "Bị chặn phát hành"} ·
              C2PA: {ex.data.c2pa.signed ? (ex.data.c2pa.verified ? `đã ký + xác minh (${ex.data.c2pa.validation_state}${ex.data.c2pa.trusted ? "" : ", chứng thư chưa được tin cậy công cộng"})` : "ký nhưng xác minh lỗi") : `chưa ký (${ex.data.c2pa.error ?? ""})`}
              · nhãn “{ex.data.label?.text}” {ex.data.label?.font_px}px
            </Alert>
          )}
        </div>
      )}
      {(rv.error || del.error) && <Alert tone="bad">{err(rv.error ?? del.error)}</Alert>}
      {!["queued", "running"].includes(j.state) && (
        <div><button className="btn sm danger" disabled={del.isPending} onClick={() => confirmDelete(`job ảnh ${j.shot_id} cùng mọi ảnh ứng viên`) && del.mutate()}><Icon name="x" className="sm" />Xóa job này</button></div>
      )}
      <details><summary>Prompt, tuyến, lượt chạy</summary>
        <div className="stack-sm">
          <div className="promptline">{j.prompt}</div>
          {j.negative_prompt && <div className="promptline">Negative: {j.negative_prompt}</div>}
          <dl className="kv">
            <dt>Tác vụ</dt><dd>{j.task}</dd>
            <dt>Tuyến</dt><dd>{j.route_plan?.map((p) => `${p.model_key}${p.est_cost ? ` (~${usd(p.est_cost)})` : ""}`).join(" → ")}</dd>
            <dt>Ghim</dt><dd>{j.characters?.map((c) => `${c.code} v${c.version}${c.costume ? ` · ${c.costume}` : ""}`).join(", ") || "—"}{j.scene ? ` · ${j.scene.code} v${j.scene.version}` : ""}</dd>
            <dt>Lượt</dt>
            <dd>{j.attempts?.map((a) => {
              const m = a.metrics;
              return `#${a.no} ${a.kind} · ${a.model_key} · ${a.state}${a.error ? ` (${a.error.slice(0, 60)})` : ""} · ${n(m.wall_s)}s · GPU ${n(m.gpu_peak_mb, 0)}MB · RAM ${n(m.ram_peak_mb, 0)}MB · ${usd(a.cost_api + a.cost_gpu)}`;
            }).join("  |  ")}</dd>
            {j.review_history?.length ? <><dt>Lịch sử duyệt</dt><dd>{j.review_history.map((h) => `${h.reviewer}: ${h.decision}${h.reason ? ` (${h.reason})` : ""}`).join(" · ")}</dd></> : null}
          </dl>
        </div>
      </details>
    </div></div>
  );
}

function CandidateCard({ c, active, onPick }: { c: Candidate; active: boolean; onPick: () => void }) {
  const v = VERDICT[c.qc_verdict] ?? VERDICT.pending;
  return (
    <figure className="shot" style={{ margin: 0, outline: active ? "3px solid var(--primary)" : "none" }}>
      <button type="button" onClick={onPick} style={{ padding: 0, border: 0, background: "none", cursor: "pointer", width: "100%" }} aria-pressed={active} aria-label="Chọn ảnh này">
        <img src={assetUrl(c.asset_id)} alt={`Ứng viên seed ${c.seed}`} loading="lazy" style={{ aspectRatio: "9/16", objectFit: "cover" }} />
      </button>
      <figcaption className="shot-bar"><Badge tone={v.tone}>{v.text}</Badge>{c.qc_score != null && <Badge>{c.qc_score.toFixed(2)}</Badge>}</figcaption>
    </figure>
  );
}

function QcTable({ c }: { c: Candidate }) {
  const checks = c.qc.checks ?? [];
  const tone = { pass: "ok", warn: "warn", fail: "bad", skipped: "" } as const;
  return (
    <details open={c.qc_verdict !== "pass"}><summary>Quality Gate ({checks.length} kiểm tra)</summary>
      <table className="tbl"><tbody>
        {checks.map((k) => (
          <tr key={k.name}><td><Badge tone={tone[k.status]}>{k.status}</Badge></td><td><code>{k.name}</code></td>
            <td>{k.value ?? "—"}{k.threshold != null ? ` / ngưỡng ${k.threshold}` : ""}</td><td>{k.message}</td></tr>))}
      </tbody></table>
    </details>
  );
}
