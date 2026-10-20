import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { assetUrl, listJobs, thumbUrl } from "./api";
import {
  createVideo, getVideo, listVideos, retryVideo, reviewVideo, videoHealth, videoMetrics,
  type RejectReason, type Segment, type VideoJob,
} from "./videoApi";
import { exportFileUrl, exportVideo, listVisual } from "./visualApi";
import { confirmDelete, deleteVideoJob } from "./manageApi";
import { Alert, Badge, Card, Empty, Field, Icon, PageHead, Segmented, Spinner, type Tone } from "./ui";

const STATE: Record<string, { text: string; tone: Tone }> = {
  queued: { text: "Đang chờ", tone: "info" }, running: { text: "Đang render", tone: "primary" },
  review_required: { text: "Chờ duyệt", tone: "warn" }, approved: { text: "Đã duyệt", tone: "ok" },
  rejected: { text: "Bị loại", tone: "bad" }, failed: { text: "Thất bại", tone: "bad" },
  returned_to_image: { text: "Quay về ảnh", tone: "warn" }, needs_human: { text: "Cần người xử lý", tone: "bad" },
};
const REASONS: [RejectReason, string][] = [
  ["motion_identity", "Mặt/áo đổi khi chuyển động"], ["keyframe_identity", "Sai nhân vật ở ảnh nguồn (→ làm lại ảnh)"],
  ["wrong_action", "Sai hành động/camera"], ["interaction_error", "Sai tương tác giữa các nhân vật"],
  ["visual_artifact", "Biến dạng, rung/nhấp nháy"], ["audio_issue", "Lỗi âm thanh/khớp môi"], ["other", "Lý do khác"],
];
const ACTIVE = ["queued", "running"];
const err = (e: unknown) => (e as Error).message;
const fmt = (n: number | null | undefined, d = 1) => (n == null ? "—" : n.toFixed(d));

// Ví dụ minh họa trong hướng dẫn (chưa phải nội dung phim đã chốt)
const EXAMPLE_PROMPT = "Cinematic medium shot of the same young man shown in the input image, wearing a white shirt in a quiet corridor. Soft window light from camera left. Preserve his facial identity, hairstyle, clothing, and background layout. A single continuous shot.";
const EXAMPLE_SEGS: Segment[] = [
  { start: 0, end: 25, prompt: "He looks forward and breathes naturally. Subtle shoulder motion." },
  { start: 25, end: 65, prompt: "He slowly turns his head toward the doorway on his right. The camera gently moves closer." },
  { start: 65, end: 100, prompt: "He pauses and his expression becomes concerned. Quiet indoor ambience, soft fabric movement." },
];

export function Video() {
  return (
    <>
      <PageHead title="Video" sub="Ảnh đã duyệt → clip qua ComfyUI + LTX-2.3 + PromptRelay. Render xong chỉ là “chờ duyệt”; người duyệt quyết định." />
      <div className="stack">
        <Health />
        <div className="split">
          <NewVideo />
          <div className="stack"><Jobs /><Metrics /></div>
        </div>
      </div>
    </>
  );
}

function Health() {
  const q = useQuery({ queryKey: ["video-health"], queryFn: videoHealth, retry: false, refetchInterval: 30000 });
  if (q.isLoading) return null;
  const h = q.data;
  if (q.error) return <Alert tone="bad">Không đọc được trạng thái video: {err(q.error)}</Alert>;
  if (!h) return null;
  if (h.error) return <Alert tone="bad"><strong>ComfyUI chưa sẵn sàng</strong> ({h.comfyui_url}): {h.error}</Alert>;
  if (h.missing_nodes?.length) return <Alert tone="bad">ComfyUI thiếu node: <code>{h.missing_nodes.join(", ")}</code>. Cài gói tương ứng rồi khởi động lại ComfyUI.</Alert>;
  if (h.warning) return <Alert tone="warn">{h.warning} · workflow <code>{h.workflow_version}</code></Alert>;
  return <Alert tone="ok">ComfyUI sẵn sàng · workflow <code>{h.workflow_version}</code></Alert>;
}

function NewVideo() {
  const qc = useQueryClient();
  const approved = useQuery({ queryKey: ["approved-images"], queryFn: () => listJobs({ limit: 40, state: "approved" }) });
  const vis = useQuery({ queryKey: ["approved-visual"], queryFn: () => listVisual("approved") });
  const [src, setSrc] = useState("");
  const [prompt, setPrompt] = useState(EXAMPLE_PROMPT);
  const [segs, setSegs] = useState<Segment[]>(EXAMPLE_SEGS);
  const [size, setSize] = useState<"512x896" | "448x768" | "768x512">("512x896");
  const [frames, setFrames] = useState(97);
  const [fps, setFps] = useState(25);
  const [seed, setSeed] = useState(42);
  const [audio, setAudio] = useState<"generated" | "none">("generated");
  const [budget, setBudget] = useState(0);
  const [key, setKey] = useState(() => crypto.randomUUID());

  const setEnd = (i: number, end: number) => setSegs((s) => s.map((x, j) => {
    if (j === i) return { ...x, end };
    if (j === i + 1) return { ...x, start: end };
    return x;
  }));
  const setText = (i: number, prompt: string) => setSegs((s) => s.map((x, j) => (j === i ? { ...x, prompt } : x)));
  const add = () => setSegs((s) => {
    const last = s[s.length - 1];
    const mid = Math.round((last.start + last.end) / 2);
    return [...s.slice(0, -1), { ...last, end: mid }, { start: mid, end: 100, prompt: "" }];
  });
  const remove = (i: number) => setSegs((s) => {
    const n = s.filter((_, j) => j !== i).map((x, j, a) => ({ ...x, start: j === 0 ? 0 : a[j - 1].end }));
    n[n.length - 1].end = 100;
    return n;
  });

  const [w, h] = size.split("x").map(Number);
  const frameOk = (frames - 1) % 8 === 0;
  const segOk = segs.every((s) => s.prompt.trim() && s.end > s.start);
  const m = useMutation({
    mutationFn: () => createVideo({
      source_keyframe_asset_id: src, global_prompt: prompt, motion_segments: segs, width: w, height: h,
      frame_count: frames, fps, seed, audio_mode: audio, budget_limit: budget,
    }, key),
    onSuccess: () => { setKey(crypto.randomUUID()); qc.invalidateQueries({ queryKey: ["videos"] }); },
  });

  const items = [
    ...(vis.data?.items.filter((j) => j.thumb_asset_id).map((j) => ({ id: j.id, character: null as { code: string } | null, label: `${j.project_id}/${j.shot_id}`,
      outputs: [{ asset_id: j.thumb_asset_id!, seed: j.seed }] })) ?? []),
    ...(approved.data?.items.filter((j) => j.outputs.length).map((j) => ({ id: j.id, character: j.character, label: j.character?.code ?? "", outputs: j.outputs })) ?? []),
  ];
  return (
    <Card title="Tạo clip" sub={`≈ ${(frames / fps).toFixed(2)} giây · ${w}×${h} · ${frames} frame @ ${fps} fps`}>
      <div className="stack">
        <Field label="Ảnh khung đầu (chỉ ảnh đã duyệt)" hint="Duyệt ở tab “Ảnh theo shot” (hoặc Lịch sử → “Duyệt ảnh” với job SDXL cũ).">
          {approved.isLoading ? <Spinner /> : items.length === 0 ? (
            <Empty icon="image" title="Chưa có ảnh đã duyệt">Duyệt một job ảnh trước khi tạo video.</Empty>
          ) : (
            <div className="gallery" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(84px, 1fr))", gap: 8 }}>
              {items.flatMap((j) => j.outputs.map((o) => (
                <button key={o.asset_id} type="button" className="shot" aria-pressed={src === o.asset_id}
                  onClick={() => setSrc(o.asset_id)} style={{ padding: 0, cursor: "pointer", outline: src === o.asset_id ? "3px solid var(--primary)" : "none" }}
                  title={`${j.label} seed ${o.seed}`}>
                  <img src={thumbUrl(o.asset_id)} alt={`Ảnh ${j.label} seed ${o.seed}`} loading="lazy" />
                </button>
              )))}
            </div>
          )}
        </Field>
        <Field label="Global prompt (mô tả cố định)" hint="Giữ nhận dạng, trang phục, bối cảnh; tiếng Anh.">
          <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={4} maxLength={1500} />
        </Field>
        <div className="stack-sm">
          <span className="section-label">Diễn biến theo đoạn (tỷ lệ % thời lượng, không phải giây/frame)</span>
          {segs.map((s, i) => (
            <div key={i} className="row" style={{ alignItems: "flex-start", flexWrap: "nowrap" }}>
              <Badge>{s.start}–</Badge>
              <input type="number" min={s.start + 1} max={100} value={s.end} disabled={i === segs.length - 1}
                onChange={(e) => setEnd(i, Number(e.target.value))} aria-label={`Kết thúc đoạn ${i + 1} (%)`} style={{ width: 70 }} />
              <input type="text" value={s.prompt} onChange={(e) => setText(i, e.target.value)} placeholder="Hành động đoạn này" style={{ flex: 1 }} aria-label={`Prompt đoạn ${i + 1}`} />
              <button type="button" className="btn ghost sm" onClick={() => remove(i)} disabled={segs.length <= 1} aria-label={`Xóa đoạn ${i + 1}`}><Icon name="x" className="sm" /></button>
            </div>
          ))}
          <div><button type="button" className="btn sm" onClick={add} disabled={segs.length >= 6 || segs[segs.length - 1].end - segs[segs.length - 1].start < 2}><Icon name="plus" className="sm" />Thêm đoạn</button></div>
        </div>
        <div className="cols-2">
          <Field label="Khung hình"><Segmented label="Khung hình" value={size} onChange={setSize}
            options={[{ value: "512x896", label: "512×896" }, { value: "448x768", label: "448×768" }, { value: "768x512", label: "768×512" }]} /></Field>
          <Field label="Âm thanh"><Segmented label="Âm thanh" value={audio} onChange={setAudio}
            options={[{ value: "generated", label: "Sinh cùng clip" }, { value: "none", label: "Không" }]} /></Field>
        </div>
        <div className="cols-4">
          <Field label="Số frame" hint={frameOk ? "8n+1" : "phải là 8n+1"}><input type="number" value={frames} step={8} min={9} max={257} onChange={(e) => setFrames(Number(e.target.value))} /></Field>
          <Field label="fps"><input type="number" value={fps} min={8} max={60} onChange={(e) => setFps(Number(e.target.value))} /></Field>
          <Field label="Seed"><input type="number" value={seed} min={0} onChange={(e) => setSeed(Number(e.target.value))} /></Field>
          <Field label="Ngân sách" hint="0 = không giới hạn"><input type="number" value={budget} min={0} onChange={(e) => setBudget(Number(e.target.value))} /></Field>
        </div>
        {m.error && <Alert tone="bad">{err(m.error)}</Alert>}
        {m.data && <Alert tone="ok">Đã xếp hàng job <code>{m.data.job_id.slice(0, 8)}</code>.</Alert>}
        <button className="btn primary lg block" disabled={!src || !prompt.trim() || !segOk || !frameOk || m.isPending} onClick={() => m.mutate()}>
          {m.isPending ? <Spinner /> : <Icon name="film" />}Tạo video
        </button>
      </div>
    </Card>
  );
}

function Jobs() {
  const [open, setOpen] = useState<string | null>(null);
  const q = useQuery({
    queryKey: ["videos"], queryFn: listVideos,
    refetchInterval: (query) => (query.state.data?.items.some((j) => ACTIVE.includes(j.state)) ? 3000 : 15000),
  });
  const items = q.data?.items ?? [];
  return (
    <Card title="Clip đã tạo" sub={`${q.data?.total ?? 0} job`}>
      {q.error && <Alert tone="bad">{err(q.error)}</Alert>}
      {q.data && items.length === 0 && <Empty icon="film" title="Chưa có clip">Tạo clip ở form bên cạnh.</Empty>}
      <div className="stack">
        {items.map((j) => {
          const st = STATE[j.state] ?? { text: j.state, tone: "" as Tone };
          return (
            <div key={j.id} className="stack-sm">
              <button className="hcard" style={{ textAlign: "left" }} onClick={() => setOpen(open === j.id ? null : j.id)} aria-expanded={open === j.id}>
                <div className="hc-body">
                  <div className="row" style={{ gap: 6 }}>
                    <Badge tone={st.tone}>{st.text}</Badge>
                    <Badge>{j.config.duration_s}s</Badge>
                    {j.repair_count > 0 && <Badge tone="info">sửa {j.repair_count}/{j.max_repairs}</Badge>}
                    {ACTIVE.includes(j.state) && <Spinner />}
                  </div>
                  <div className="hc-title">{j.global_prompt}</div>
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
  const q = useQuery({ queryKey: ["video", id], queryFn: () => getVideo(id), refetchInterval: (query) => (ACTIVE.includes(query.state.data?.state ?? "") ? 3000 : false) });
  const [reason, setReason] = useState<RejectReason>("motion_identity");
  const [note, setNote] = useState("");
  const refresh = () => { qc.invalidateQueries({ queryKey: ["video", id] }); qc.invalidateQueries({ queryKey: ["videos"] }); qc.invalidateQueries({ queryKey: ["video-metrics"] }); };
  const review = useMutation({ mutationFn: (d: "approve" | "reject") => reviewVideo(id, d === "approve" ? { decision: d, note } : { decision: d, reason, note }), onSuccess: refresh });
  const ex = useMutation({ mutationFn: () => exportVideo(id, { target: "auto", label: true }) });
  const del = useMutation({ mutationFn: () => deleteVideoJob(id), onSuccess: refresh });
  const retry = useMutation({ mutationFn: (seed: number) => retryVideo(id, { seed }), onSuccess: refresh });
  const j: VideoJob | undefined = q.data;
  if (!j) return <div className="row"><Spinner /> Đang tải…</div>;
  const rendered = [...(j.attempts ?? [])].reverse().find((a) => a.state === "rendered");
  const canRetry = (j.state === "rejected" || j.state === "failed") && j.repair_count < j.max_repairs;
  return (
    <div className="card flat"><div className="card-b stack">
      {j.error && <Alert tone="bad">{j.error}</Alert>}
      {rendered?.output_asset_id && (
        <video controls preload="metadata" src={assetUrl(rendered.output_asset_id)} style={{ width: "100%", maxHeight: 420, borderRadius: 10, background: "#000" }} />
      )}
      {j.state === "review_required" && (
        <div className="stack-sm">
          <span className="section-label">Duyệt cả chuyển động và âm thanh</span>
          <div className="row">
            <button className="btn primary" disabled={review.isPending} onClick={() => review.mutate("approve")}><Icon name="check" />Duyệt</button>
            <select value={reason} onChange={(e) => setReason(e.target.value as RejectReason)} aria-label="Lý do loại">
              {REASONS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
            <button className="btn danger" disabled={review.isPending} onClick={() => review.mutate("reject")}><Icon name="x" />Loại</button>
          </div>
          <input type="text" value={note} onChange={(e) => setNote(e.target.value)} placeholder="Ghi chú (tùy chọn)" aria-label="Ghi chú duyệt" />
        </div>
      )}
      {j.state === "approved" && (
        <div className="stack-sm">
          <button className="btn" disabled={ex.isPending} onClick={() => ex.mutate()}><Icon name="download" />Xuất bản (1080×1920 + nhãn AI + C2PA)</button>
          {ex.error && <Alert tone="bad">{err(ex.error)}</Alert>}
          {ex.data && <Alert tone={ex.data.status === "released" ? "ok" : "bad"}>{ex.data.status === "released" ? <a href={exportFileUrl(ex.data.id)} download>Tải mp4 đã gắn nhãn</a> : "Bị chặn phát hành"} · C2PA: {ex.data.c2pa.signed ? (ex.data.c2pa.verified ? `đã ký + xác minh${ex.data.c2pa.trusted ? "" : " (chứng thư chưa được tin cậy công cộng)"}` : "ký nhưng xác minh lỗi") : `chưa ký (${ex.data.c2pa.error ?? ""})`}</Alert>}
        </div>
      )}
      {j.state === "returned_to_image" && <Alert tone="warn">Lỗi ở ảnh nguồn: quay về tab Tạo ảnh, tạo và duyệt ảnh mới rồi tạo video mới.</Alert>}
      {j.state === "needs_human" && <Alert tone="bad">Đã hết lượt sửa hoặc ngân sách — cần người xử lý.</Alert>}
      {canRetry && <div><button className="btn" disabled={retry.isPending} onClick={() => retry.mutate(Math.floor(Math.random() * 2 ** 31))}><Icon name="refresh" />Sửa tự động (seed mới) · còn {j.max_repairs - j.repair_count} lượt</button></div>}
      {(review.error || retry.error || del.error) && <Alert tone="bad">{err(review.error ?? retry.error ?? del.error)}</Alert>}
      {!["queued", "running"].includes(j.state) && (
        <div><button className="btn sm danger" disabled={del.isPending} onClick={() => confirmDelete("clip video này cùng các lượt render") && del.mutate()}><Icon name="x" className="sm" />Xóa clip</button></div>
      )}
      <dl className="kv">
        <dt>Chi phí</dt><dd>{fmt(j.cost_spent)}{j.budget_limit ? ` / ${j.budget_limit}` : ""}</dd>
        <dt>Workflow</dt><dd><code>{j.workflow_version}</code></dd>
        <dt>Model</dt><dd>{j.model_version ?? "—"}</dd>
        <dt>Lượt render</dt>
        <dd>{(j.attempts ?? []).map((a) => `#${a.attempt_no} ${a.kind === "repair" ? "sửa" : "đầu"} · ${a.state}${a.total_seconds ? ` · ${fmt(a.total_seconds, 0)}s` : ""}${a.review?.decision ? ` · ${a.review.decision}${a.review.reason ? `:${a.review.reason}` : ""}` : ""}`).join("  |  ")}</dd>
      </dl>
    </div></div>
  );
}

function Metrics() {
  const q = useQuery({ queryKey: ["video-metrics"], queryFn: videoMetrics, refetchInterval: 20000 });
  const m = q.data;
  if (!m) return null;
  return (
    <Card title="Số liệu pilot" sub="Ngưỡng đề xuất: tỷ lệ đạt ≥ 80%; p95 và ngân sách do chủ dự án chốt">
      <dl className="kv">
        <dt>Tỷ lệ đạt</dt><dd>{m.approval_rate == null ? "—" : `${(m.approval_rate * 100).toFixed(0)}% (${m.approved}/${m.reviewed} clip đã duyệt)`}</dd>
        <dt>Độ trễ p50 / p95</dt><dd>{fmt(m.latency_s.p50, 0)} s / {fmt(m.latency_s.p95, 0)} s (n={m.latency_s.n})</dd>
        <dt>Chi phí / clip đạt</dt><dd>{fmt(m.cost_per_approved)} (tổng {fmt(m.total_cost)}, gồm lượt hỏng/sửa)</dd>
        <dt>Lượt render</dt><dd>{m.attempts_rendered} xong · {m.attempts_failed} lỗi</dd>
        <dt>Lý do loại</dt><dd>{Object.keys(m.reject_reasons).length ? Object.entries(m.reject_reasons).map(([k, v]) => `${k}: ${v}`).join(" · ") : "—"}</dd>
      </dl>
    </Card>
  );
}
