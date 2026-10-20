import { useQuery, useQueryClient } from "@tanstack/react-query";
import { getReport, listModels, patchModel, type ModelInfo, type ModelReport } from "./visualApi";
import { Alert, Badge, Card, Icon, PageHead, Spinner, type Tone } from "./ui";

const f = (x: number | null | undefined, d = 2) => (x == null ? "—" : x.toFixed(d));
const mb = (x: number | null | undefined) => (x == null ? "—" : `${Math.round(x).toLocaleString("vi-VN")} MB`);
const VERIFY: Record<string, { text: string; tone: Tone }> = {
  verified: { text: "đã xác minh", tone: "ok" }, connector_unverified: { text: "kết nối chưa gọi thử", tone: "warn" },
  unverified: { text: "chưa xác minh", tone: "bad" }, manual: { text: "nhập thủ công", tone: "info" },
};

export function Reports() {
  const q = useQuery({ queryKey: ["report"], queryFn: getReport, refetchInterval: 30000 });
  const r = q.data;
  return (
    <>
      <PageHead title="Báo cáo vận hành" sub="Chi phí, thời gian, bộ nhớ và RAM từ các lượt chạy thật. Ô “—” nghĩa là chưa có dữ liệu hoặc không đo được."
        actions={<div className="row">
          <a className="btn sm" href="/api/v1/reports/report.md"><Icon name="download" className="sm" />Báo cáo (.md)</a>
          <a className="btn sm" href="/api/v1/reports/attempts.csv"><Icon name="download" className="sm" />Từng lượt (.csv)</a>
        </div>} />
      {q.isLoading && <div className="row"><Spinner /> Đang tải…</div>}
      {q.error && <Alert tone="bad">{(q.error as Error).message}</Alert>}
      {r && (
        <div className="stack">
          <Card title="Chi phí ảnh" sub={r.visual.formula}>
            <dl className="kv">
              <dt>Job / được duyệt</dt><dd>{r.visual.totals.jobs} / {r.visual.totals.approved} · {r.visual.totals.reviews} lượt duyệt</dd>
              <dt>C tạo / C sửa / C GPU</dt><dd>{f(r.visual.totals.c_create, 4)} / {f(r.visual.totals.c_repair, 4)} / {f(r.visual.totals.c_gpu, 4)} USD</dd>
              <dt>C QA / C lưu trữ / C huấn luyện</dt><dd>{f(r.visual.totals.c_qa, 4)} / {f(r.visual.totals.c_storage, 4)} / {f(r.visual.totals.c_training, 4)} USD</dd>
              <dt>Tổng</dt><dd><strong>{f(r.visual.totals.c_total, 4)} USD</strong></dd>
            </dl>
            <Alert tone="info">{r.visual.rates.note}</Alert>
          </Card>
          <Card title="Theo model" sub="Chi phí/ảnh đạt = tổng chi phí phân bổ / số ảnh được duyệt">
            <div style={{ overflowX: "auto" }}>
              <table className="tbl">
                <thead><tr><th>Model</th><th>Lượt</th><th>Lỗi</th><th>Ảnh</th><th>Đạt</th><th>Tỉ lệ</th><th>USD/ảnh đạt</th><th>Giờ GPU</th><th>Thời gian p50/p95 (s)</th>
                  <th>Chờ hàng p50</th><th>VRAM đỉnh</th><th>VRAM torch</th><th>RAM đỉnh</th><th>RSS tiến trình</th></tr></thead>
                <tbody>{r.visual.models.map((m: ModelReport) => (
                  <tr key={m.model_key}>
                    <td><code>{m.model_key}</code></td><td>{m.attempts}</td><td>{m.failed}</td><td>{m.images}</td><td>{m.approved}</td>
                    <td>{f(m.approval_rate)}</td><td>{f(m.cost.per_approved, 4)}</td><td>{f(m.gpu_hours, 3)}</td>
                    <td>{f(m.latency_s.wall.p50, 1)} / {f(m.latency_s.wall.p95, 1)}</td><td>{f(m.latency_s.queue.p50, 1)}</td>
                    <td>{mb(m.memory_mb.gpu_peak.max)}</td><td>{mb(m.memory_mb.torch_vram_peak.max)}</td>
                    <td>{mb(m.memory_mb.ram_peak_system.max)}</td><td>{mb(m.memory_mb.process_rss_peak.max)}</td>
                  </tr>))}
                </tbody>
              </table>
              {r.visual.models.length === 0 && <p>Chưa có lượt chạy nào.</p>}
            </div>
          </Card>
          <Card title="Video (ComfyUI + LTX-2.3)">
            <dl className="kv">
              <dt>Lượt render</dt><dd>{r.video.rendered} xong · {r.video.failed} lỗi · đã xem {r.video.reviewed} · đạt {r.video.approved} (tỉ lệ {f(r.video.approval_rate)})</dd>
              <dt>Thời gian tổng p50 / p95</dt><dd>{f(r.video.latency_s.total?.p50, 1)} / {f(r.video.latency_s.total?.p95, 1)} s</dd>
              <dt>Chờ render / tải file (p50)</dt><dd>{f(r.video.latency_s.render_wait?.p50, 1)} s / {f(r.video.latency_s.download?.p50, 1)} s</dd>
              <dt>VRAM / RAM đỉnh (ComfyUI)</dt><dd>{mb(r.video.memory_mb.gpu_peak?.max)} / {mb(r.video.memory_mb.ram_peak?.max)}</dd>
              <dt>Giờ GPU · chi phí</dt><dd>{f(r.video.gpu_hours, 3)} h · {f(r.video.cost.total)} ({r.video.cost.unit}) · / clip đạt {f(r.video.cost.per_approved)}</dd>
            </dl>
          </Card>
          <Card title="Máy chủ API hiện tại" sub="GPU nằm ở container worker/ComfyUI nên API thường không thấy GPU; xem tab Model để biết GPU của worker.">
            <dl className="kv">
              {r.system.gpus.map((g) => <><dt>GPU</dt><dd key={g.name}>{g.name}: {mb(g.used_mb)} / {mb(g.total_mb)} · tải {g.util}%</dd></>)}
              {r.system.ram && <><dt>RAM</dt><dd>{mb(r.system.ram.used_mb)} / {mb(r.system.ram.total_mb)} (còn {mb(r.system.ram.available_mb)}) · {r.system.cpu_count} lõi</dd></>}
              {Object.entries(r.system.disks).map(([p, d]) => d && <><dt>Đĩa {p}</dt><dd key={p}>trống {d.free_gb} / {d.total_gb} GB</dd></>)}
            </dl>
          </Card>
        </div>
      )}
    </>
  );
}

export function Models() {
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ["vmodels"], queryFn: listModels, refetchInterval: 30000 });
  const w = q.data?.worker as Record<string, unknown> | null | undefined;
  return (
    <>
      <PageHead title="Model Registry" sub="Danh mục model theo báo cáo mục 1.3. Model Router chỉ dùng model đang bật, đủ năng lực và khả dụng; mỗi dòng ghi mức xác minh trung thực." />
      <div className="stack">
        <Alert tone={w?.cuda ? "ok" : "warn"}>{w?.cuda ? <>Worker GPU: <strong>{String(w.name)}</strong> · {String(w.total_gb)} GB VRAM · torch {String(w.torch)} · FaceNet {w.face_ok ? "sẵn sàng" : `chưa dùng được (${String(w.face_error)})`}</> :
          "Worker GPU chưa báo cáo CUDA: các model tự chạy sẽ không được chọn."}</Alert>
        {q.isLoading && <div className="row"><Spinner /> Đang tải…</div>}
        <Card>
          <div style={{ overflowX: "auto" }}>
            <table className="tbl">
              <thead><tr><th>Model</th><th>Loại</th><th>Giấy phép</th><th>Giá tham khảo</th><th>Cần VRAM</th><th>Năng lực</th><th>Xác minh</th><th>Khả dụng</th><th>Benchmark</th><th>Bật</th></tr></thead>
              <tbody>{q.data?.models.map((m: ModelInfo) => {
                const v = VERIFY[m.verification] ?? { text: m.verification, tone: "" as Tone };
                const b = m.benchmark as Record<string, number | null>;
                return (
                  <tr key={m.key}>
                    <td><strong>{m.name}</strong><br /><small>{m.vendor} · <code>{m.key}</code></small></td>
                    <td><Badge tone={m.route === "api" ? "info" : m.route === "local" ? "primary" : ""}>{m.route}</Badge></td>
                    <td>{m.license ?? "—"}</td>
                    <td><small>{JSON.stringify(m.pricing).slice(0, 90)}</small></td>
                    <td>{m.min_vram_gb ? `${m.min_vram_gb} GB` : "—"}</td>
                    <td><small>{Object.entries(m.capabilities).filter(([, x]) => x).map(([k]) => k).join(", ")}</small></td>
                    <td><Badge tone={v.tone}>{v.text}</Badge><br /><small>{m.checked_at}</small></td>
                    <td>{m.available ? <Badge tone="ok">có</Badge> : <Badge tone="bad">không</Badge>}<br /><small>{m.availability}</small></td>
                    <td><small>{b?.pass_rate != null ? `đạt ${f(b.pass_rate)} · $${f(b.cost_per_pass, 3)}/ảnh đạt` : "chưa chạy"}</small></td>
                    <td><input type="checkbox" checked={m.enabled} aria-label={`Bật ${m.key}`} onChange={async (e) => { await patchModel(m.key, e.target.checked); qc.invalidateQueries({ queryKey: ["vmodels"] }); }} /></td>
                  </tr>);
              })}</tbody>
            </table>
          </div>
        </Card>
        <Alert tone="info">Giá là giá công khai tham khảo (báo cáo hoặc nguồn thứ cấp), không phải hóa đơn. “kết nối chưa gọi thử”: bộ kết nối viết theo tài liệu công khai nhưng chưa gọi với khóa thật — chạy một job thật để xác nhận.</Alert>
      </div>
    </>
  );
}
