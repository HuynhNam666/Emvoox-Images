import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { assetUrl } from "./api";
import { applyBench, createBench, getBench, listBench, listModels, nextRating, sendRating } from "./visualApi";
import { confirmDelete, deleteBenchmark } from "./manageApi";
import { Alert, Badge, Card, Empty, Field, Icon, PageHead, Spinner } from "./ui";

const err = (e: unknown) => (e as Error).message;
const f = (x: number | null | undefined, d = 2) => (x == null ? "—" : x.toFixed(d));

export function Benchmark() {
  const [run, setRun] = useState<string>("");
  return (
    <>
      <PageHead title="Benchmark chấm mù" sub="30 cảnh × 4 lần tạo × các lựa chọn model (báo cáo mục 1.2). Hai người chấm độc lập, ẩn tên công cụ; trọng số: nhất quán 40% · đúng cảnh 30% · hình thể 20% · thẩm mỹ 10%. Lỗi sai nhân vật/hành động cốt lõi là lỗi loại." />
      <div className="stack">
        <NewRun onCreated={setRun} />
        <Runs run={run} onPick={setRun} />
        {run && <Rate runId={run} />}
        {run && <Results runId={run} onDeleted={() => setRun("")} />}
      </div>
    </>
  );
}

function NewRun({ onCreated }: { onCreated: (id: string) => void }) {
  const qc = useQueryClient();
  const models = useQuery({ queryKey: ["vmodels"], queryFn: listModels });
  const [name, setName] = useState("Vòng 1 — prompt nền");
  const [round, setRound] = useState(1);
  const [picked, setPicked] = useState<string[]>([]);
  const [variants, setVariants] = useState(4);
  const m = useMutation({
    mutationFn: () => createBench({ name, round_no: round, model_keys: picked, variants }),
    onSuccess: (d) => { qc.invalidateQueries({ queryKey: ["bench"] }); onCreated(d.run_id); },
  });
  const avail = models.data?.models.filter((x) => x.route !== "manual" && x.available) ?? [];
  return (
    <Card title="Lượt benchmark mới" sub="Vòng 1: chỉ prompt (năng lực nền). Vòng 2: có ảnh tham chiếu/LoRA (quy trình sản xuất). Cần nhân vật A01, A02.">
      <div className="stack">
        <div className="cols-3">
          <Field label="Tên"><input value={name} onChange={(e) => setName(e.target.value)} /></Field>
          <Field label="Vòng"><select value={round} onChange={(e) => setRound(Number(e.target.value))}><option value={1}>1 — prompt</option><option value={2}>2 — tham chiếu</option></select></Field>
          <Field label="Số lần tạo / cảnh"><input type="number" min={1} max={4} value={variants} onChange={(e) => setVariants(Number(e.target.value))} /></Field>
        </div>
        <Field label="Model so sánh (chỉ model khả dụng)" hint="Báo cáo: 5 lựa chọn → 600 ảnh mỗi vòng">
          <div className="row">{avail.map((x) => (
            <label key={x.key} className="row" style={{ gap: 4 }}>
              <input type="checkbox" checked={picked.includes(x.key)} onChange={(e) => setPicked(e.target.checked ? [...picked, x.key] : picked.filter((k) => k !== x.key))} />{x.key}</label>))}
            {avail.length === 0 && <span>Chưa có model khả dụng — xem tab Model.</span>}</div>
        </Field>
        {m.error && <Alert tone="bad">{err(m.error)}</Alert>}
        {m.data && <Alert tone="ok">Đã tạo {m.data.jobs_created} job (~{m.data.images_expected} ảnh), ước tính {m.data.estimated_cost} USD (API).{m.data.skipped.length ? ` Bỏ qua ${m.data.skipped.length}: ${m.data.skipped[0].reason}` : ""}</Alert>}
        <button className="btn primary" disabled={!picked.length || m.isPending} onClick={() => m.mutate()}>{m.isPending ? <Spinner /> : <Icon name="play" />}Chạy benchmark</button>
      </div>
    </Card>
  );
}

function Runs({ run, onPick }: { run: string; onPick: (id: string) => void }) {
  const q = useQuery({ queryKey: ["bench"], queryFn: listBench });
  return (
    <Card title="Các lượt benchmark">
      {q.data?.length === 0 && <Empty icon="layers" title="Chưa có lượt nào">Tạo lượt đầu tiên ở trên.</Empty>}
      <div className="row">{q.data?.map((r) => (
        <button key={r.id} className="btn" aria-pressed={run === r.id} onClick={() => onPick(r.id)} style={run === r.id ? { outline: "2px solid var(--primary)" } : undefined}>
          {r.name} <Badge>vòng {r.round_no}</Badge></button>))}</div>
    </Card>
  );
}

function Rate({ runId }: { runId: string }) {
  const qc = useQueryClient();
  const [rater, setRater] = useState(() => { try { return localStorage.getItem("emvoox.rater") ?? ""; } catch { return ""; } });
  const [s, setS] = useState({ consistency: 3, scene_fit: 3, anatomy: 3, aesthetics: 3 });
  const [core, setCore] = useState(false);
  const q = useQuery({ queryKey: ["next", runId, rater], queryFn: () => nextRating(runId, rater), enabled: !!rater });
  const m = useMutation({
    mutationFn: () => sendRating({ run_id: runId, candidate_id: q.data!.candidate_id, rater, ...s, core_error: core }),
    onSuccess: () => { setS({ consistency: 3, scene_fit: 3, anatomy: 3, aesthetics: 3 }); setCore(false); qc.invalidateQueries({ queryKey: ["next", runId, rater] }); qc.invalidateQueries({ queryKey: ["bench-detail", runId] }); },
  });
  const d = q.data;
  const row = (k: keyof typeof s, label: string) => (
    <Field key={k} label={`${label}: ${s[k]}`}><input type="range" min={1} max={5} value={s[k]} onChange={(e) => setS({ ...s, [k]: Number(e.target.value) })} /></Field>);
  return (
    <Card title="Chấm ảnh (ẩn tên công cụ)" sub="Mỗi ảnh cần hai người chấm khác nhau. Nhập tên của bạn; không xem trước kết quả của người kia.">
      <div className="stack">
        <Field label="Người chấm"><input value={rater} onChange={(e) => { setRater(e.target.value); try { localStorage.setItem("emvoox.rater", e.target.value); } catch { /* bỏ qua */ } }} placeholder="Tên (vd: Nam)" style={{ maxWidth: 220 }} /></Field>
        {q.isFetching && <Spinner />}
        {d?.done && <Alert tone="ok">Bạn đã chấm hết các ảnh có thể chấm.</Alert>}
        {d && !d.done && d.brief && (
          <div className="split">
            <img src={assetUrl(d.asset_id!)} alt="Ảnh cần chấm" style={{ maxWidth: "100%", maxHeight: 520, borderRadius: 10 }} />
            <div className="stack">
              <Alert tone="info"><strong>Yêu cầu cảnh:</strong> {d.brief.characters.join(" + ")} — {d.brief.action}; cảm xúc {d.brief.emotion}; {d.brief.camera.size}; {d.brief.lighting}. <br /><small>Còn {d.remaining} ảnh.</small></Alert>
              {row("consistency", "Nhất quán nhận dạng (40%)")}{row("scene_fit", "Đúng cảnh (30%)")}{row("anatomy", "Hình thể & chi tiết (20%)")}{row("aesthetics", "Thẩm mỹ (10%)")}
              <label className="row"><input type="checkbox" checked={core} onChange={(e) => setCore(e.target.checked)} />Lỗi loại: sai nhân vật hoặc sai hành động cốt lõi</label>
              {m.error && <Alert tone="bad">{err(m.error)}</Alert>}
              <button className="btn primary" disabled={m.isPending} onClick={() => m.mutate()}><Icon name="check" />Gửi điểm &amp; ảnh tiếp</button>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}

function Results({ runId, onDeleted }: { runId: string; onDeleted: () => void }) {
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ["bench-detail", runId], queryFn: () => getBench(runId), refetchInterval: 10000 });
  const ap = useMutation({ mutationFn: () => applyBench(runId), onSuccess: () => qc.invalidateQueries({ queryKey: ["vmodels"] }) });
  const del = useMutation({ mutationFn: () => deleteBenchmark(runId), onSuccess: () => { qc.invalidateQueries({ queryKey: ["bench"] }); onDeleted(); } });
  const d = q.data;
  if (!d) return null;
  return (
    <Card title="Kết quả" sub={`${d.pass_rule} · job ${d.jobs.done}/${d.jobs.total} (lỗi ${d.jobs.failed}) · chấm ${d.ratings}/${d.ratings_needed}`}>
      <div style={{ overflowX: "auto" }}>
        <table className="tbl">
          <thead><tr><th>Model</th><th>Ảnh</th><th>Đã đủ 2 người chấm</th><th>Đạt</th><th>Lỗi loại</th><th>Tỉ lệ đạt</th><th>Điểm TB (0–100)</th><th>QC tự động đạt</th>
            <th>Chi phí</th><th>USD/ảnh đạt</th><th>p50/p95 (s)</th><th>VRAM đỉnh</th></tr></thead>
          <tbody>{d.models.map((m) => (
            <tr key={m.model_key}><td><code>{m.model_key}</code></td><td>{m.images}</td><td>{m.rated_by_two}</td><td>{m.passed}</td><td>{m.core_errors}</td>
              <td>{f(m.pass_rate)}</td><td>{f(m.mean_weighted, 1)}</td><td>{m.qc_auto_pass}</td><td>{f(m.cost, 4)}</td><td>{f(m.cost_per_pass, 4)}</td>
              <td>{f(m.latency_s.p50, 1)} / {f(m.latency_s.p95, 1)}</td><td>{m.gpu_peak_mb_max ? `${Math.round(m.gpu_peak_mb_max)} MB` : "—"}</td></tr>))}</tbody>
        </table>
      </div>
      <div className="row" style={{ marginTop: 12 }}>
        <button className="btn" disabled={ap.isPending} onClick={() => ap.mutate()}>Ghi kết quả vào Model Registry (cho Model Router)</button>
        {ap.data && <Badge tone="ok">đã ghi: {ap.data.applied.join(", ") || "chưa đủ 20 ảnh/ model"}</Badge>}
        {ap.error && <Alert tone="bad">{err(ap.error)}</Alert>}
        <button className="btn danger" disabled={del.isPending} onClick={() => confirmDelete("lượt benchmark này cùng mọi ảnh và điểm chấm") && del.mutate()}><Icon name="x" className="sm" />Xóa lượt benchmark</button>
        {del.error && <Alert tone="bad">{err(del.error)}</Alert>}
      </div>
      <Alert tone="info">Chỉ chốt model khi đủ mẫu: quy mô báo cáo là 30 cảnh × 4 lần × 5 lựa chọn = 600 ảnh mỗi vòng. Kết quả ít ảnh chỉ mang tính thăm dò.</Alert>
    </Card>
  );
}
