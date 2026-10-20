import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { listCharacters } from "./api";
import { listScenes, type SceneInfo } from "./visualApi";
import {
  confirmDelete, createCostume, createScene, deleteCostume, deleteScene, listCostumes, newSceneVersion, patchCostume, patchScene,
  type Costume,
} from "./manageApi";
import { Alert, Badge, Card, Empty, Field, Icon, PageHead, Spinner } from "./ui";

const err = (e: unknown) => (e as Error).message;
const SPEC_FIELDS: [string, string, string][] = [
  ["layout", "Bố cục", "vd a narrow indoor corridor"], ["openings", "Cửa / vị trí đồ vật", "vd door on the right wall near the end"],
  ["time_of_day", "Thời điểm", "vd late morning"], ["light_direction", "Hướng sáng", "vd camera left (window)"],
  ["palette", "Bảng màu", "vd warm off-white walls"],
];

export function Scenes() {
  return (
    <>
      <PageHead title="Bối cảnh & trang phục" sub="Scene Library và trang phục theo nhân vật. Sửa bối cảnh tạo version mới (job cũ vẫn trỏ về version đã dùng); xóa bị chặn khi còn job đang dùng." />
      <div className="split">
        <SceneLib />
        <CostumeLib />
      </div>
    </>
  );
}

function SceneForm({ initial, onSubmit, pending, label }: {
  initial?: SceneInfo; onSubmit: (b: { code: string; name: string; spec: Record<string, unknown> }) => void; pending: boolean; label: string;
}) {
  const cur = initial?.versions.find((v) => v.version === initial.current_version);
  const [code, setCode] = useState(initial?.code ?? "");
  const [name, setName] = useState(initial?.name ?? "");
  const [spec, setSpec] = useState<Record<string, string>>(Object.fromEntries(SPEC_FIELDS.map(([k]) => [k, String(cur?.spec?.[k] ?? "")])));
  const [props, setProps] = useState(Array.isArray(cur?.spec?.props) ? (cur!.spec.props as string[]).join(", ") : "");
  return (
    <div className="stack">
      <div className="cols-2">
        <Field label="Mã"><input value={code} disabled={!!initial} onChange={(e) => setCode(e.target.value)} placeholder="vd B02" /></Field>
        <Field label="Tên"><input value={name} onChange={(e) => setName(e.target.value)} /></Field>
      </div>
      {SPEC_FIELDS.map(([k, l, ph]) => (
        <Field key={k} label={l}><input value={spec[k]} placeholder={ph} onChange={(e) => setSpec({ ...spec, [k]: e.target.value })} /></Field>))}
      <Field label="Đạo cụ (cách nhau dấu phẩy)"><input value={props} onChange={(e) => setProps(e.target.value)} placeholder="wall lamp, wooden bench" /></Field>
      <button className="btn primary" disabled={!code.trim() || !name.trim() || pending}
        onClick={() => onSubmit({ code: code.trim(), name: name.trim(), spec: { ...spec, props: props.split(",").map((s) => s.trim()).filter(Boolean) } })}>
        {pending ? <Spinner /> : <Icon name="check" className="sm" />}{label}
      </button>
    </div>
  );
}

function SceneLib() {
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ["scenes"], queryFn: listScenes });
  const [edit, setEdit] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const done = () => { setEdit(null); setAdding(false); qc.invalidateQueries({ queryKey: ["scenes"] }); };
  const add = useMutation({ mutationFn: createScene, onSuccess: done });
  const upd = useMutation({ mutationFn: ({ code, b }: { code: string; b: { code: string; name: string; spec: Record<string, unknown> } }) => newSceneVersion(code, b), onSuccess: done });
  const ren = useMutation({ mutationFn: ({ code, name }: { code: string; name: string }) => patchScene(code, name), onSuccess: done });
  const del = useMutation({ mutationFn: (code: string) => deleteScene(code), onSuccess: done });
  return (
    <Card title="Bối cảnh" sub={`${q.data?.length ?? 0} bối cảnh`} actions={<button className="btn sm" onClick={() => setAdding(!adding)}><Icon name="plus" className="sm" />Thêm</button>}>
      <div className="stack">
        {adding && <div className="card flat"><div className="card-b"><SceneForm label="Tạo bối cảnh" pending={add.isPending} onSubmit={(b) => add.mutate(b)} /></div></div>}
        {[add.error, upd.error, ren.error, del.error].filter(Boolean).map((e, i) => <Alert key={i} tone="bad">{err(e)}</Alert>)}
        {q.data?.length === 0 && !adding && <Empty icon="image" title="Chưa có bối cảnh">Bấm “Thêm” để tạo.</Empty>}
        {q.data?.map((s) => (
          <div key={s.code} className="stack-sm">
            <div className="row">
              <Badge tone="primary">{s.code}</Badge><strong>{s.name}</strong><Badge>v{s.current_version}</Badge><span className="spacer" />
              <button className="btn sm" onClick={() => setEdit(edit === s.code ? null : s.code)}><Icon name="refresh" className="sm" />Sửa</button>
              <button className="btn sm" onClick={() => { const n = window.prompt("Tên mới cho bối cảnh", s.name); if (n && n.trim()) ren.mutate({ code: s.code, name: n.trim() }); }}>Đổi tên</button>
              <button className="btn sm danger" disabled={del.isPending} onClick={() => confirmDelete(`bối cảnh ${s.code}`) && del.mutate(s.code)}><Icon name="x" className="sm" />Xóa</button>
            </div>
            {edit === s.code && (
              <div className="card flat"><div className="card-b">
                <SceneForm initial={s} label="Lưu thành version mới" pending={upd.isPending} onSubmit={(b) => upd.mutate({ code: s.code, b })} />
              </div></div>
            )}
          </div>
        ))}
      </div>
    </Card>
  );
}

function CostumeLib() {
  const qc = useQueryClient();
  const chars = useQuery({ queryKey: ["characters"], queryFn: listCharacters });
  const [cid, setCid] = useState("");
  const list = useQuery({ queryKey: ["costumes", cid], queryFn: () => listCostumes(cid), enabled: !!cid });
  const [code, setCode] = useState("");
  const [desc, setDesc] = useState("");
  const refresh = () => qc.invalidateQueries({ queryKey: ["costumes", cid] });
  const add = useMutation({ mutationFn: () => createCostume(cid, { code: code.trim(), description_en: desc.trim() }), onSuccess: () => { setCode(""); setDesc(""); refresh(); } });
  const upd = useMutation({ mutationFn: ({ id, description_en }: { id: string; description_en: string }) => patchCostume(id, { description_en }), onSuccess: refresh });
  const del = useMutation({ mutationFn: (id: string) => deleteCostume(id), onSuccess: refresh });
  return (
    <Card title="Trang phục theo nhân vật">
      <div className="stack">
        <Field label="Nhân vật">
          <select value={cid} onChange={(e) => setCid(e.target.value)}><option value="">— chọn —</option>
            {chars.data?.map((c) => <option key={c.id} value={c.id}>{c.code} · {c.name}</option>)}</select>
        </Field>
        {cid && (
          <>
            {list.data?.length === 0 && <Empty icon="user" title="Chưa có trang phục">Thêm bên dưới.</Empty>}
            {list.data?.map((c: Costume) => (
              <div key={c.id} className="row">
                <Badge tone="primary">{c.code}</Badge><span style={{ flex: 1 }}>{c.description_en}</span>
                <button className="btn sm" onClick={() => { const n = window.prompt("Mô tả trang phục (tiếng Anh)", c.description_en); if (n && n.trim()) upd.mutate({ id: c.id, description_en: n.trim() }); }}>Sửa</button>
                <button className="btn sm danger" disabled={del.isPending} onClick={() => confirmDelete(`trang phục ${c.code}`) && del.mutate(c.id)}><Icon name="x" className="sm" />Xóa</button>
              </div>
            ))}
            <div className="cols-2">
              <Field label="Mã"><input value={code} onChange={(e) => setCode(e.target.value)} placeholder="vd WHITE_SHIRT" /></Field>
              <Field label="Mô tả (tiếng Anh)"><input value={desc} onChange={(e) => setDesc(e.target.value)} placeholder="a plain white shirt" /></Field>
            </div>
            <button className="btn primary" disabled={!code.trim() || !desc.trim() || add.isPending} onClick={() => add.mutate()}><Icon name="plus" className="sm" />Thêm trang phục</button>
            {[add.error, upd.error, del.error].filter(Boolean).map((e, i) => <Alert key={i} tone="bad">{err(e)}</Alert>)}
          </>
        )}
      </div>
    </Card>
  );
}
