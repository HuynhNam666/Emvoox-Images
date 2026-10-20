import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { assetUrl, createCharacter, createVersion, getCharacter, listCharacters, thumbUrl, type Identity } from "./api";
import { confirmDelete, deleteCharacter, patchCharacter } from "./manageApi";
import { Alert, Badge, Card, Empty, Field, Icon, PageHead, Spinner } from "./ui";

const EMPTY: Identity = { description_vi: "", identity_en: "", negative_prompt: "" };

export function Library() {
  const qc = useQueryClient();
  const chars = useQuery({ queryKey: ["characters"], queryFn: listCharacters });
  const [selected, setSelected] = useState<string | null>(null);
  const [code, setCode] = useState("");
  const [name, setName] = useState("");

  const create = useMutation({
    mutationFn: () => createCharacter(code.trim(), name.trim(), EMPTY),
    onSuccess: (c) => { qc.invalidateQueries({ queryKey: ["characters"] }); setSelected(c.id); setCode(""); setName(""); },
  });

  return (
    <>
      <PageHead title="Thư viện nhân vật" sub="Mỗi nhân vật có hồ sơ nhận dạng và ảnh chuẩn. Mọi thay đổi tạo version mới; job cũ vẫn trỏ về version đã dùng." />
      <div className="split-wide">
        <div className="stack">
          <Card title="Nhân vật" sub={chars.data ? `${chars.data.length} hồ sơ` : undefined}>
            {chars.isLoading && <div className="row"><Spinner /> Đang tải…</div>}
            {chars.error && <Alert tone="bad">{(chars.error as Error).message}</Alert>}
            {chars.data?.length === 0 && <Empty icon="users" title="Chưa có nhân vật">Thêm nhân vật đầu tiên ở khung bên dưới.</Empty>}
            <div className="list">
              {chars.data?.map((c) => {
                const ref = c.current_version?.references[0];
                return (
                  <button key={c.id} className="list-item" aria-current={selected === c.id} onClick={() => setSelected(c.id)}>
                    {ref ? <img className="thumb" src={thumbUrl(ref)} alt="" /> : <span className="thumb ph"><Icon name="user" /></span>}
                    <span style={{ minWidth: 0 }}>
                      <div className="t">{c.code} · {c.name}</div>
                      <div className="s">v{c.current_version?.version} · {c.current_version?.references.length ?? 0} ảnh chuẩn</div>
                    </span>
                  </button>
                );
              })}
            </div>
          </Card>

          <Card title="Thêm nhân vật">
            <div className="stack">
              <div className="cols-2">
                <Field label="Mã"><input placeholder="vd A03" value={code} onChange={(e) => setCode(e.target.value)} /></Field>
                <Field label="Tên"><input placeholder="Tên hiển thị" value={name} onChange={(e) => setName(e.target.value)} /></Field>
              </div>
              <button className="btn primary" disabled={!code.trim() || !name.trim() || create.isPending} onClick={() => create.mutate()}>
                <Icon name="plus" className="sm" />Tạo nhân vật
              </button>
              {create.error && <Alert tone="bad">{(create.error as Error).message}</Alert>}
            </div>
          </Card>
        </div>

        <div>
          {selected ? <Detail id={selected} onDeleted={() => setSelected(null)} /> : (
            <Card><Empty icon="user" title="Chọn một nhân vật">Hồ sơ, ảnh chuẩn và lịch sử version sẽ hiện ở đây.</Empty></Card>
          )}
        </div>
      </div>
    </>
  );
}

function Detail({ id, onDeleted }: { id: string; onDeleted: () => void }) {
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ["character", id], queryFn: () => getCharacter(id) });
  const [draft, setDraft] = useState<{ id: string; v: Identity } | null>(null);
  const cur = q.data?.current_version;
  const identity = draft && draft.id === cur?.id ? draft.v : cur?.identity;

  const refresh = () => { qc.invalidateQueries({ queryKey: ["character", id] }); qc.invalidateQueries({ queryKey: ["characters"] }); };
  const save = useMutation({ mutationFn: () => createVersion(id, { identity }), onSuccess: () => { setDraft(null); refresh(); } });
  const rename = useMutation({ mutationFn: (b: { name?: string; code?: string }) => patchCharacter(id, b), onSuccess: refresh });
  const remove = useMutation({ mutationFn: () => deleteCharacter(id), onSuccess: () => { qc.invalidateQueries({ queryKey: ["characters"] }); onDeleted(); } });
  const dropRef = useMutation({
    mutationFn: (asset: string) => createVersion(id, { reference_asset_ids: cur!.references.filter((a) => a !== asset) }),
    onSuccess: refresh,
  });

  if (q.isLoading || !q.data || !cur || !identity) return <Card><div className="row"><Spinner /> Đang tải…</div></Card>;
  const edit = (patch: Partial<Identity>) => setDraft({ id: cur.id, v: { ...identity, ...patch } });
  const changed = JSON.stringify(identity) !== JSON.stringify(cur.identity);

  return (
    <div className="stack">
      <Card title={<>{q.data.code} · {q.data.name}</>} sub="Hồ sơ nhận dạng" actions={<div className="row">
        <Badge tone="primary">version {cur.version}</Badge>
        <button className="btn sm" onClick={() => { const n = window.prompt("Tên hiển thị", q.data.name); if (n && n.trim()) rename.mutate({ name: n.trim() }); }}>Đổi tên</button>
        <button className="btn sm" onClick={() => { const c = window.prompt("Mã nhân vật (chữ, số, _ -)", q.data.code); if (c && c.trim() && c.trim() !== q.data.code) rename.mutate({ code: c.trim() }); }}>Đổi mã</button>
        <button className="btn sm danger" disabled={remove.isPending} onClick={() => confirmDelete(`nhân vật ${q.data.code} cùng toàn bộ version và ảnh chuẩn`) && remove.mutate()}><Icon name="x" className="sm" />Xóa</button>
      </div>}>
        <div className="stack">
          <Field label="Mô tả gốc (tiếng Việt)">
            <textarea rows={2} value={identity.description_vi} onChange={(e) => edit({ description_vi: e.target.value })} />
          </Field>
          <Field label="Nhận dạng gửi model (tiếng Anh)" hint="Đoạn này được ghép vào đầu mỗi prompt của nhân vật.">
            <textarea rows={2} value={identity.identity_en} onChange={(e) => edit({ identity_en: e.target.value })} />
          </Field>
          <Field label="Negative prompt">
            <input value={identity.negative_prompt} onChange={(e) => edit({ negative_prompt: e.target.value })} />
          </Field>
          <div className="row">
            <button className="btn primary" disabled={!changed || save.isPending} onClick={() => save.mutate()}>
              {save.isPending ? <Spinner /> : <Icon name="check" className="sm" />}Lưu thành version mới
            </button>
            {changed && <small>Có thay đổi chưa lưu</small>}
          </div>
          {save.error && <Alert tone="bad">{(save.error as Error).message}</Alert>}
          {(rename.error || remove.error) && <Alert tone="bad">{((rename.error ?? remove.error) as Error).message}</Alert>}
        </div>
      </Card>

      <Card title="Ảnh chuẩn" sub={`${cur.references.length} ảnh · dùng cho IP-Adapter ở trang Tạo ảnh`}>
        {cur.references.length === 0 ? (
          <Empty icon="image" title="Chưa có ảnh chuẩn">Vào “Tạo ảnh”, chọn nhân vật này rồi bấm “Đặt làm ảnh chuẩn” trên ảnh ưng ý.</Empty>
        ) : (
          <div className="gallery">
            {cur.references.map((a) => (
              <figure key={a} className="shot" style={{ margin: 0 }}>
                <a href={assetUrl(a)} target="_blank" rel="noreferrer"><img src={thumbUrl(a)} alt="Ảnh chuẩn" /></a>
                <figcaption className="shot-bar">
                  <span className="spacer" />
                  <button className="btn sm ghost danger" disabled={dropRef.isPending} onClick={() => dropRef.mutate(a)}><Icon name="x" className="sm" />Bỏ</button>
                </figcaption>
              </figure>
            ))}
          </div>
        )}
        {dropRef.error && <div style={{ marginTop: 10 }}><Alert tone="bad">{(dropRef.error as Error).message}</Alert></div>}
      </Card>

      <Card title="Lịch sử version">
        <div className="stack-sm">
          {q.data.versions?.map((v) => (
            <div key={v.id} className="row">
              <Badge tone={v.id === cur.id ? "primary" : ""}>v{v.version}</Badge>
              <small>{v.references.length} ảnh chuẩn</small>
              <span className="spacer" />
              <small>{v.created_at?.slice(0, 16).replace("T", " ")}</small>
              {v.id === cur.id && <Badge tone="ok">hiện tại</Badge>}
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
