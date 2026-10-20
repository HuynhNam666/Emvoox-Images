import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Character, api, assetUrl } from "../api";

interface AssetList { total: number; items: { id: string; provenance: Record<string, unknown>; created_at: string }[] }

export default function Characters() {
  const [q, setQ] = useState("");
  const chars = useQuery({ queryKey: ["characters", q], queryFn: () => api.get<Character[]>(`/characters${q ? `?q=${encodeURIComponent(q)}` : ""}`) });
  const gallery = useQuery({ queryKey: ["assets", "generated"], queryFn: () => api.get<AssetList>("/assets?kind=generated&limit=24") });
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Thư viện nhân vật</h1>
      <input className="input max-w-sm" placeholder="Tìm theo tên hoặc mã…" value={q} onChange={(e) => setQ(e.target.value)} />
      {chars.isLoading && <p>Đang tải…</p>}
      {chars.data?.length === 0 && <p className="text-sm text-slate-500">Chưa có nhân vật. Tạo ảnh rồi chọn “Lưu làm ảnh chuẩn”.</p>}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {chars.data?.map((c) => {
          const canon = c.current_version?.references.find((r) => r.is_canonical)?.asset_id;
          return (
            <div key={c.id} className="card space-y-2">
              {canon ? <img src={assetUrl(canon)} className="aspect-[3/4] w-full rounded object-cover" /> : <div className="aspect-[3/4] rounded bg-slate-100" />}
              <div className="font-medium">{c.code} — {c.name}</div>
              <div className="text-xs text-slate-500">Phiên bản nhận dạng v{c.current_version?.version ?? "-"}</div>
            </div>
          );
        })}
      </div>
      <div>
        <h2 className="mb-2 font-medium">Ảnh đã tạo gần đây ({gallery.data?.total ?? 0})</h2>
        <div className="grid grid-cols-3 gap-2 sm:grid-cols-4 lg:grid-cols-6">
          {gallery.data?.items.map((a) => (
            <a key={a.id} href={assetUrl(a.id, "content")} target="_blank"><img src={assetUrl(a.id)} className="aspect-square w-full rounded object-cover" loading="lazy" /></a>
          ))}
        </div>
      </div>
    </div>
  );
}
