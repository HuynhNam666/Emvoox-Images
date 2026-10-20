import { useQuery } from "@tanstack/react-query";
import { ModelVersion, api } from "../api";

const KIND: Record<string, string> = { base: "Base checkpoint", character_lora: "Character LoRA", style_lora: "Style LoRA" };

export default function Models() {
  const q = useQuery({ queryKey: ["models"], queryFn: () => api.get<ModelVersion[]>("/models") });
  const ready = useQuery({ queryKey: ["ready"], queryFn: () => api.get<Record<string, unknown>>("/health/ready") });
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Model Registry & Cài đặt</h1>
      <div className="card text-sm">
        <h2 className="mb-2 font-medium">Kết nối</h2>
        <pre className="text-xs">{JSON.stringify(ready.data, null, 2)}</pre>
        <p className="mt-2 text-xs text-slate-500">Mật khẩu database và token không bao giờ được trả về giao diện.</p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full rounded border bg-white text-left text-sm">
          <thead className="bg-slate-100 text-xs uppercase text-slate-600">
            <tr><th className="p-2">Tên</th><th>Loại</th><th>Family</th><th>Nguồn / revision</th><th>License</th><th>Trạng thái</th><th>Duyệt</th></tr>
          </thead>
          <tbody>
            {q.data?.map((m) => (
              <tr key={m.id} className="border-t">
                <td className="p-2 font-medium">{m.name}</td><td>{KIND[m.kind] ?? m.kind}</td><td>{m.family}</td>
                <td className="font-mono text-xs">{m.hf_id ?? "—"} {m.revision ? `@${m.revision.slice(0, 8)}` : "(chưa ghim revision)"}</td>
                <td className="text-xs">{m.license ?? "—"}</td><td>{m.status}</td><td>{m.review_status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
