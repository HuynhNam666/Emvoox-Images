import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { assetUrl, listCharacters, listJobs, thumbUrl, type HistoryJob } from "./api";
import { useQueryClient } from "@tanstack/react-query";
import { reviewImageJob } from "./videoApi";
import { Alert, Badge, Card, Empty, Icon, PageHead, Spinner, type Tone } from "./ui";

const STATE: Record<string, { text: string; tone: Tone }> = {
  queued: { text: "Đang chờ", tone: "info" }, running: { text: "Đang tạo", tone: "primary" },
  review_required: { text: "Chờ duyệt", tone: "warn" }, approved: { text: "Đã duyệt", tone: "ok" },
  rejected: { text: "Bị loại", tone: "bad" }, failed: { text: "Thất bại", tone: "bad" }, cancelled: { text: "Đã hủy", tone: "" },
};
const FILTERS = [["", "Tất cả"], ["review_required", "Chờ duyệt"], ["approved", "Đã duyệt"], ["failed", "Thất bại"], ["running", "Đang tạo"], ["queued", "Đang chờ"]] as const;
const PAGE = 24;

const when = (iso: string | null) => (iso ? new Date(iso).toLocaleString("vi-VN", { dateStyle: "short", timeStyle: "short" }) : "—");
const aspectLabel = (a?: string) => (a === "1:1-small" ? "768²" : a ?? "—");

export function History({ onReuse }: { onReuse: (job: HistoryJob) => void }) {
  const [state, setState] = useState("");
  const [charId, setCharId] = useState("");
  const [limit, setLimit] = useState(PAGE);
  const [open, setOpen] = useState<HistoryJob | null>(null);
  const chars = useQuery({ queryKey: ["characters"], queryFn: listCharacters });
  const q = useQuery({
    queryKey: ["history", state, charId, limit],
    queryFn: () => listJobs({ limit, state: state || undefined, character_id: charId || undefined }),
    refetchInterval: (query) => (query.state.data?.items.some((j) => j.state === "queued" || j.state === "running") ? 4000 : 20000),
  });
  const items = q.data?.items ?? [];
  const total = q.data?.total ?? 0;

  return (
    <>
      <PageHead title="Lịch sử tạo ảnh" sub="Mọi job tạo ảnh đã lưu trong PostgreSQL. Bấm vào một thẻ để xem tham số, prompt và dùng lại." />
      <div className="stack">
        <div className="toolbar">
          <div className="seg" role="group" aria-label="Lọc theo trạng thái" style={{ background: "var(--surface)" }}>
            {FILTERS.map(([v, l]) => <button key={v} type="button" aria-pressed={state === v} onClick={() => { setState(v); setLimit(PAGE); }}>{l}</button>)}
          </div>
          <select value={charId} onChange={(e) => { setCharId(e.target.value); setLimit(PAGE); }} aria-label="Lọc theo nhân vật">
            <option value="">Mọi nhân vật</option>
            {chars.data?.map((c) => <option key={c.id} value={c.id}>{c.code} · {c.name}</option>)}
          </select>
          <span className="spacer" />
          <Badge>{total} job</Badge>
          {q.isFetching && <Spinner />}
        </div>

        {q.error && <Alert tone="bad">{(q.error as Error).message}</Alert>}
        {q.isLoading && <div className="row"><Spinner /> Đang tải…</div>}
        {q.data && items.length === 0 && (
          <Card><Empty icon="clock" title="Chưa có job nào">{state || charId ? "Không có job khớp bộ lọc." : "Tạo ảnh ở tab “Tạo ảnh”, lịch sử sẽ hiện ở đây."}</Empty></Card>
        )}

        <div className="gallery big">
          {items.map((j) => {
            const st = STATE[j.state] ?? { text: j.state, tone: "" as Tone };
            const first = j.outputs[0];
            const c = j.config;
            return (
              <button key={j.id} className="hcard" onClick={() => setOpen(j)} aria-label={`Xem job ${j.id.slice(0, 8)}`}>
                {first ? <img src={thumbUrl(first.asset_id)} alt="" loading="lazy" /> : (
                  <div className="ph">
                    {j.state === "queued" || j.state === "running" ? <Spinner /> : <Icon name={j.state === "failed" ? "alert" : "image"} />}
                  </div>
                )}
                <div className="hc-body">
                  <div className="row" style={{ gap: 6 }}>
                    <Badge tone={st.tone}>{st.text}</Badge>
                    {j.outputs.length > 1 && <Badge>×{j.outputs.length}</Badge>}
                    {j.character && <Badge tone="primary">{j.character.code}</Badge>}
                    {c.lora_version_id && <Badge tone="info">LoRA</Badge>}
                    {c.use_references && <Badge tone="info">IP</Badge>}
                  </div>
                  <div className="hc-title">{j.prompt_original}</div>
                  <div>{when(j.created_at)} · {aspectLabel(c.aspect_ratio)} · {c.steps} bước · seed {c.seed}</div>
                </div>
              </button>
            );
          })}
        </div>

        {items.length < total && (
          <div className="row" style={{ justifyContent: "center" }}>
            <button className="btn" onClick={() => setLimit(limit + PAGE)}>Xem thêm ({total - items.length} job nữa)</button>
          </div>
        )}
      </div>
      {open && <Detail job={items.find((j) => j.id === open.id) ?? open} onClose={() => setOpen(null)} onReuse={(j) => { setOpen(null); onReuse(j); }} />}
    </>
  );
}

function Detail({ job, onClose, onReuse }: { job: HistoryJob; onClose: () => void; onReuse: (j: HistoryJob) => void }) {
  const [copied, setCopied] = useState(false);
  const qc = useQueryClient();
  const [reviewErr, setReviewErr] = useState<string | null>(null);
  const review = async (d: "approved" | "rejected") => {
    try { setReviewErr(null); await reviewImageJob(job.id, d); qc.invalidateQueries({ queryKey: ["history"] }); qc.invalidateQueries({ queryKey: ["approved-images"] }); }
    catch (e) { setReviewErr((e as Error).message); }
  };
  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [onClose]);
  const st = STATE[job.state] ?? { text: job.state, tone: "" as Tone };
  const c = job.config;
  const prompt = job.prompt_compiled || job.prompt_original;
  const copy = async () => { try { await navigator.clipboard.writeText(prompt); setCopied(true); setTimeout(() => setCopied(false), 1500); } catch { /* không có quyền clipboard */ } };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" role="dialog" aria-modal="true" aria-label="Chi tiết job" onClick={(e) => e.stopPropagation()}>
        <div className="modal-h">
          <strong>Job <code>{job.id.slice(0, 8)}</code></strong>
          <Badge tone={st.tone}>{st.text}</Badge>
          <small>{when(job.created_at)}</small>
          <span className="spacer" />
          {["review_required", "approved", "rejected"].includes(job.state) && job.outputs.length > 0 && (<>
            <button className="btn sm" disabled={job.state === "approved"} onClick={() => review("approved")}><Icon name="check" className="sm" />Duyệt ảnh</button>
            <button className="btn sm danger" disabled={job.state === "rejected"} onClick={() => review("rejected")}><Icon name="x" className="sm" />Loại</button>
          </>)}
          <button className="btn primary sm" onClick={() => onReuse(job)}><Icon name="refresh" className="sm" />Dùng lại tham số</button>
          <button className="btn ghost sm" onClick={onClose} aria-label="Đóng"><Icon name="x" /></button>
        </div>
        <div className="modal-b">
          <div className="stack">
            {job.outputs.length === 0 && <Empty icon="image" title="Chưa có ảnh">{job.state === "failed" ? "Job thất bại, xem lỗi bên phải." : "Job chưa tạo xong ảnh."}</Empty>}
            <div className="gallery" style={{ gridTemplateColumns: job.outputs.length > 1 ? "repeat(2, minmax(0, 1fr))" : "1fr" }}>
              {job.outputs.map((o) => (
                <figure key={o.asset_id} className="shot" style={{ margin: 0 }}>
                  <a href={assetUrl(o.asset_id)} target="_blank" rel="noreferrer"><img src={assetUrl(o.asset_id)} alt={`Ảnh seed ${o.seed}`} /></a>
                  <figcaption className="shot-bar">
                    <Badge>seed {o.seed}</Badge><span className="spacer" />
                    <a className="btn sm" href={assetUrl(o.asset_id)} download={`emvoox-${o.asset_id.slice(0, 8)}.png`}><Icon name="download" className="sm" />Tải</a>
                  </figcaption>
                </figure>
              ))}
            </div>
          </div>
          <div className="stack">
            {job.error && <Alert tone="bad"><strong>Lỗi:</strong> {job.error}</Alert>}
            {reviewErr && <Alert tone="bad">{reviewErr}</Alert>}
            <div className="stack-sm">
              <div className="row"><span className="section-label">Prompt gửi model</span><span className="spacer" />
                <button className="btn ghost sm" onClick={copy}><Icon name={copied ? "check" : "copy"} className="sm" />{copied ? "Đã chép" : "Chép"}</button></div>
              <div className="promptline">{prompt}</div>
            </div>
            {job.prompt_original !== prompt && <div className="stack-sm"><span className="section-label">Mô tả gốc</span><div className="promptline">{job.prompt_original}</div></div>}
            {c.negative_prompt && <div className="stack-sm"><span className="section-label">Negative prompt</span><div className="promptline">{c.negative_prompt}</div></div>}
            <dl className="kv">
              <dt>Nhân vật</dt><dd>{job.character ? `${job.character.code} · ${job.character.name} (v${job.character.version})` : "—"}</dd>
              <dt>Khung hình</dt><dd>{aspectLabel(c.aspect_ratio)}</dd>
              <dt>Seed</dt><dd>{c.seed}{c.num_images > 1 ? ` (+${c.num_images - 1})` : ""}</dd>
              <dt>Số bước</dt><dd>{c.steps}</dd>
              <dt>Guidance</dt><dd>{c.guidance_scale}</dd>
              <dt>Số ảnh</dt><dd>{c.num_images}</dd>
              <dt>LoRA</dt><dd>{c.lora_version_id ? <><code>{String(c.lora_version_id).slice(0, 8)}</code> · scale {c.lora_scale}{c.evaluation ? " · đánh giá" : ""}</> : "không"}</dd>
              <dt>IP-Adapter</dt><dd>{c.use_references ? `bật · scale ${c.reference_scale}` : "không"}</dd>
            </dl>
          </div>
        </div>
      </div>
    </div>
  );
}
