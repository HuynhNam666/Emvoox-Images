import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { assetUrl, listCharacters, thumbUrl } from "./api";
import {
  bulkItems, createDataset, createVersion_, listDatasets, listItems, listVersions, manifestUrl, patchItem, uploadFiles,
  type DsGroup, type DsItem, type ImportReport, type ItemStatus, type Split,
} from "./datasetApi";
import { Alert, Badge, Card, Empty, Field, Icon, PageHead, Spinner, type Tone } from "./ui";

const GROUPS: { key: DsGroup; label: string; hint: string }[] = [
  { key: "identity", label: "1. Khuôn mặt (Identity)", hint: "Cận mặt chính diện, nghiêng trái/phải, góc ¾, ngẩng/cúi" },
  { key: "body", label: "2. Toàn thân (Body)", hint: "Toàn thân, bán thân, mặt trước/sau/hai bên" },
  { key: "expression", label: "3. Biểu cảm (Expression)", hint: "Bình thường, vui, buồn, giận, ngạc nhiên, lo lắng" },
  { key: "pose", label: "4. Tư thế (Pose)", hint: "Đứng, ngồi, đi, chạy, cúi, cầm đồ vật" },
  { key: "outfit", label: "5. Trang phục (Outfit)", hint: "Từng bộ đồ nhiều góc, toàn thân + chi tiết" },
  { key: "appearance", label: "6. Tóc & phụ kiện (Appearance)", hint: "Kiểu tóc, kính, trang sức, giày, vật dụng đặc trưng" },
  { key: "style", label: "7. Phong cách (Style)", hint: "Anime, manhua, 2D giả 3D, ảnh thật… (train Style LoRA riêng)" },
  { key: "scene", label: "8. Bối cảnh (Scene)", hint: "Phòng, đường phố, văn phòng, ngày/đêm" },
];
const STATUS_LABEL: Record<ItemStatus, string> = { pending: "Chờ duyệt", approved: "Đã duyệt", rejected: "Đã loại" };
const STATUS_TONE: Record<ItemStatus, Tone> = { pending: "", approved: "ok", rejected: "bad" };
const STATUSES = Object.keys(STATUS_LABEL) as ItemStatus[];
const err = (e: unknown) => (e as Error).message;
const groupLabel = (k?: DsGroup) => GROUPS.find((x) => x.key === k)?.label;

export function Datasets() {
  const qc = useQueryClient();
  const list = useQuery({ queryKey: ["datasets"], queryFn: listDatasets });
  const chars = useQuery({ queryKey: ["characters"], queryFn: listCharacters });
  const [sel, setSel] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [group, setGroup] = useState<DsGroup>("identity");
  const [charId, setCharId] = useState("");
  const [source, setSource] = useState("");
  const [license, setLicense] = useState("");

  const create = useMutation({
    mutationFn: () => createDataset({ name: name.trim(), group, character_id: charId || null, source: source || undefined, license: license || undefined }),
    onSuccess: (d) => { qc.invalidateQueries({ queryKey: ["datasets"] }); setSel(d.id); setName(""); },
  });
  const g = GROUPS.find((x) => x.key === group)!;

  return (
    <>
      <PageHead title="Dataset" sub="Quản lý ảnh huấn luyện theo 8 nhóm: upload, sửa caption, duyệt, chia train/val/test rồi khóa thành version bất biến." />
      <div className="split-wide">
        <div className="stack">
          <Card title="Danh sách dataset" sub={list.data ? `${list.data.length} dataset` : undefined}>
            {list.isLoading && <div className="row"><Spinner /> Đang tải…</div>}
            {list.error && <Alert tone="bad">{err(list.error)}</Alert>}
            {list.data?.length === 0 && <Empty icon="database" title="Chưa có dataset">Tạo dataset đầu tiên ở khung bên dưới.</Empty>}
            <div className="list">
              {list.data?.map((d) => (
                <button key={d.id} className="list-item" aria-current={sel === d.id} onClick={() => setSel(d.id)} style={{ alignItems: "flex-start" }}>
                  <span style={{ minWidth: 0, display: "grid", gap: 6, width: "100%" }}>
                    <span className="t">{d.name}</span>
                    <span className="s">{groupLabel(d.group)}</span>
                    <span className="row" style={{ gap: 6 }}>
                      <Badge tone="ok">{d.counts.approved ?? 0} duyệt</Badge>
                      <Badge>{d.counts.pending ?? 0} chờ</Badge>
                      <Badge tone="bad">{d.counts.rejected ?? 0} loại</Badge>
                      <span className="spacer" />
                      <Badge tone="primary">v{d.latest_version}</Badge>
                    </span>
                  </span>
                </button>
              ))}
            </div>
          </Card>

          <Card title="Tạo dataset">
            <div className="stack">
              <Field label="Tên"><input placeholder="vd A02 — khuôn mặt" value={name} onChange={(e) => setName(e.target.value)} /></Field>
              <Field label="Nhóm" hint={`Cần có: ${g.hint}`}>
                <select value={group} onChange={(e) => setGroup(e.target.value as DsGroup)}>{GROUPS.map((x) => <option key={x.key} value={x.key}>{x.label}</option>)}</select>
              </Field>
              <Field label="Nhân vật">
                <select value={charId} onChange={(e) => setCharId(e.target.value)}>
                  <option value="">Không gắn (dùng chung / phong cách)</option>
                  {chars.data?.map((c) => <option key={c.id} value={c.id}>{c.code} · {c.name}</option>)}
                </select>
              </Field>
              <Field label="Nguồn"><input placeholder="URL / tác giả" value={source} onChange={(e) => setSource(e.target.value)} /></Field>
              <Field label="License / quyền sử dụng"><input placeholder="vd CC BY 4.0, ảnh tự sinh…" value={license} onChange={(e) => setLicense(e.target.value)} /></Field>
              <button className="btn primary" disabled={!name.trim() || create.isPending} onClick={() => create.mutate()}><Icon name="plus" className="sm" />Tạo dataset</button>
              {create.error && <Alert tone="bad">{err(create.error)}</Alert>}
            </div>
          </Card>
        </div>

        <div>
          {sel ? <Detail key={sel} id={sel} /> : <Card><Empty icon="layers" title="Chọn hoặc tạo một dataset">Ảnh, caption và các version sẽ hiện ở đây.</Empty></Card>}
        </div>
      </div>
    </>
  );
}

function Detail({ id }: { id: string }) {
  const qc = useQueryClient();
  const ds = useQuery({ queryKey: ["datasets"], queryFn: listDatasets, select: (all) => all.find((d) => d.id === id) });
  const [filter, setFilter] = useState<ItemStatus | "">("");
  const items = useQuery({ queryKey: ["items", id, filter], queryFn: () => listItems(id, filter || undefined) });
  const versions = useQuery({ queryKey: ["versions", id], queryFn: () => listVersions(id) });
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [progress, setProgress] = useState<number | null>(null);
  const [report, setReport] = useState<ImportReport | null>(null);
  const [trigger, setTrigger] = useState("");
  const [conflict, setConflict] = useState<string | null>(null);

  const refresh = () => { qc.invalidateQueries({ queryKey: ["items", id] }); qc.invalidateQueries({ queryKey: ["datasets"] }); };
  const up = useMutation({
    mutationFn: (files: File[]) => { setProgress(0); return uploadFiles(id, files, setProgress); },
    onSuccess: (r) => { setReport(r); refresh(); },
    onSettled: () => setProgress(null),
  });
  const bulk = useMutation({ mutationFn: (b: Parameters<typeof bulkItems>[1]) => bulkItems(id, b), onSuccess: refresh });
  const mkVersion = useMutation({
    mutationFn: (harmonize: boolean) => createVersion_(id, harmonize),
    onMutate: () => setConflict(null),
    onError: (e) => {
      try {
        const m = JSON.parse((e as Error).message);
        if (m?.code === "near_duplicate_split_conflict") setConflict(m.message);
      } catch { /* lỗi thường: hiển thị qua mkVersion.error */ }
    },
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["versions", id] }); qc.invalidateQueries({ queryKey: ["datasets"] }); refresh(); },
  });

  const list = items.data?.items ?? [];
  const toggle = (i: string) => setPicked((s) => { const n = new Set(s); if (n.has(i)) n.delete(i); else n.add(i); return n; });
  const ids = [...picked];
  const d = ds.data;

  return (
    <div className="stack">
      <Card title={d?.name} sub={<>{groupLabel(d?.group)} · Nguồn: {d?.source || "—"} · License: {d?.license || "—"}</>}
        actions={d && <><Badge tone="ok">{d.counts.approved ?? 0} duyệt</Badge><Badge>{d.counts.pending ?? 0} chờ</Badge><Badge tone="bad">{d.counts.rejected ?? 0} loại</Badge></>}>
        <div className="stack">
          <label className="dropzone">
            <Icon name="upload" />
            <span><strong>Chọn ảnh hoặc ZIP để upload</strong></span>
            <small>JPG / PNG / WebP · ZIP: ảnh kèm file .txt cùng tên làm caption</small>
            <input type="file" multiple accept="image/png,image/jpeg,image/webp,.zip" disabled={up.isPending}
              onChange={(e) => { if (e.target.files?.length) up.mutate([...e.target.files]); e.target.value = ""; }} />
          </label>
          {progress !== null && <div className="stack-sm"><div className="progress" aria-label="Tiến độ upload"><i style={{ width: `${progress}%` }} /></div><small>Đang upload {progress}%</small></div>}
          {up.error && <Alert tone="bad">{err(up.error)}</Alert>}
          {report && (
            <Alert tone={report.rejected.length ? "warn" : "ok"}>
              <strong>Báo cáo import:</strong> nhập {report.imported} · trùng {report.duplicates.length} · gần trùng {report.near_duplicates.length} · độ phân giải thấp (tự loại) {report.low_resolution.length} · bị từ chối {report.rejected.length}
              {report.rejected.map((r) => <div key={r.file}>✗ {r.file}: {r.reason}</div>)}
              {report.near_duplicates.map((r) => <div key={r.file}>≈ {r.file} gần trùng ảnh đã có</div>)}
            </Alert>
          )}
        </div>
      </Card>

      <Card title="Ảnh" sub={items.data ? `${list.length} ảnh đang hiển thị` : undefined}>
        <div className="stack">
          <div className="toolbar">
            <select value={filter} onChange={(e) => { setFilter(e.target.value as ItemStatus | ""); setPicked(new Set()); }} aria-label="Lọc theo trạng thái">
              <option value="">Tất cả</option>{STATUSES.map((s) => <option key={s} value={s}>{STATUS_LABEL[s]}</option>)}
            </select>
            <button className="btn sm" onClick={() => setPicked(new Set(list.map((i) => i.id)))}>Chọn tất cả</button>
            <button className="btn sm ghost" onClick={() => setPicked(new Set())}>Bỏ chọn</button>
            <span className="divider-v" />
            <Badge tone={picked.size ? "primary" : ""}>{picked.size} đã chọn</Badge>
            <button className="btn sm" disabled={!picked.size} onClick={() => bulk.mutate({ item_ids: ids, status: "approved" })}><Icon name="check" className="sm" />Duyệt</button>
            <button className="btn sm danger" disabled={!picked.size} onClick={() => bulk.mutate({ item_ids: ids, status: "rejected" })}><Icon name="x" className="sm" />Loại</button>
            {(["train", "val", "test"] as Split[]).map((s) => <button key={s} className="btn sm" disabled={!picked.size} onClick={() => bulk.mutate({ item_ids: ids, split: s })}>→ {s}</button>)}
            <span className="divider-v" />
            <input placeholder="Trigger (vd emv_a02)" value={trigger} onChange={(e) => setTrigger(e.target.value)} aria-label="Trigger token" />
            <button className="btn sm" disabled={!picked.size || !trigger.trim()} onClick={() => bulk.mutate({ item_ids: ids, caption_prepend: trigger.trim() })}>Thêm vào đầu caption</button>
          </div>
          {bulk.error && <Alert tone="bad">{err(bulk.error)}</Alert>}
          {items.isLoading && <div className="row"><Spinner /> Đang tải…</div>}
          {items.data && list.length === 0 && <Empty icon="image" title="Chưa có ảnh trong mục này">Upload ảnh hoặc ZIP ở phía trên.</Empty>}
          <div className="gallery">
            {list.map((i) => <Card2 key={i.id + i.caption + i.status + i.split} item={i} picked={picked.has(i.id)} onPick={() => toggle(i.id)} onSaved={refresh} />)}
          </div>
        </div>
      </Card>

      <Card title="Version bất biến" sub="Khóa ảnh đã duyệt thành snapshot để train; sửa dữ liệu sau đó sẽ tạo version mới."
        actions={<button className="btn primary" disabled={mkVersion.isPending || !d?.counts.approved} onClick={() => mkVersion.mutate(false)}>{mkVersion.isPending ? <Spinner /> : <Icon name="layers" className="sm" />}Khóa version</button>}>
        <div className="stack">
          {conflict && <Alert tone="warn">{conflict} <button className="btn sm" style={{ marginLeft: 8 }} onClick={() => mkVersion.mutate(true)}>Gộp cùng split</button></Alert>}
          {mkVersion.error && !conflict && <Alert tone="bad">{err(mkVersion.error)}</Alert>}
          {versions.data?.length === 0 && <small>Chưa có version nào.</small>}
          {versions.data?.map((v) => (
            <div key={v.id} className="card flat" style={{ padding: 12 }}>
              <div className="row">
                <Badge tone="primary">v{v.version}</Badge>
                <span>{v.item_count} ảnh</span>
                <small>{Object.entries(v.split_counts).map(([k, n]) => `${k}: ${n}`).join(" · ")}</small>
                <span className="spacer" />
                <small>hash <code>{v.manifest_hash.slice(0, 12)}</code></small>
                <a className="btn sm" href={manifestUrl(v.id)}><Icon name="download" className="sm" />manifest.jsonl</a>
              </div>
              {v.warnings.map((w) => <div key={w} style={{ marginTop: 8 }}><Alert tone="warn">{w}</Alert></div>)}
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function Card2({ item, picked, onPick, onSaved }: { item: DsItem; picked: boolean; onPick: () => void; onSaved: () => void }) {
  const [cap, setCap] = useState(item.caption);
  const save = useMutation({ mutationFn: (b: Parameters<typeof patchItem>[1]) => patchItem(item.id, b), onSuccess: onSaved });
  return (
    <figure className={`dcard ${item.status}${picked ? " picked" : ""}`} style={{ margin: 0 }}>
      <label className="dc-top">
        <input type="checkbox" checked={picked} onChange={onPick} style={{ margin: 0 }} />
        <span title={item.filename ?? ""}>{item.filename}</span>
        <span style={{ marginLeft: "auto" }}><Badge tone={STATUS_TONE[item.status]}>{STATUS_LABEL[item.status]}</Badge></span>
      </label>
      <a href={assetUrl(item.asset_id)} target="_blank" rel="noreferrer"><img src={thumbUrl(item.asset_id)} alt={item.filename ?? "ảnh dataset"} loading="lazy" /></a>
      <div className="dc-body">
        <textarea rows={2} value={cap} placeholder="Caption…" aria-label="Caption" onChange={(e) => setCap(e.target.value)}
          onBlur={() => { if (cap !== item.caption) save.mutate({ caption: cap }); }} />
        <div className="dc-row">
          <select value={item.status} aria-label="Trạng thái" onChange={(e) => save.mutate({ status: e.target.value as ItemStatus })}>
            {STATUSES.map((s) => <option key={s} value={s}>{STATUS_LABEL[s]}</option>)}</select>
          <select value={item.split} aria-label="Split" onChange={(e) => save.mutate({ split: e.target.value as Split })}><option>train</option><option>val</option><option>test</option></select>
          <label className="check" style={{ fontSize: 12 }}><input type="checkbox" checked={item.synthetic} onChange={(e) => save.mutate({ synthetic: e.target.checked })} />AI</label>
        </div>
        <div className="dc-row">
          <small>{item.width}×{item.height}</small>
          {item.flags.map((f) => <Badge key={f} tone="warn">{f.startsWith("near_duplicate") ? "gần trùng" : f === "low_resolution" ? "độ phân giải thấp" : f}</Badge>)}
        </div>
        {save.error && <Alert tone="bad">{err(save.error)}</Alert>}
      </div>
    </figure>
  );
}
