import { useQuery } from "@tanstack/react-query";
import { api } from "../api";

interface Dash {
  characters: number; datasets: number; model_versions: number; running_jobs: number;
  recent_jobs: { id: string; state: string; created_at: string; prompt: string }[];
}

export default function Dashboard() {
  const q = useQuery({ queryKey: ["dashboard"], queryFn: () => api.get<Dash>("/dashboard"), refetchInterval: 5000 });
  if (q.isLoading) return <p>Đang tải…</p>;
  if (q.error) return <p className="text-red-600">Không tải được số liệu: {(q.error as Error).message}</p>;
  const d = q.data!;
  const tiles: [string, number][] = [
    ["Nhân vật", d.characters], ["Dataset", d.datasets], ["LoRA / model version", d.model_versions], ["Job đang chạy", d.running_jobs],
  ];
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Tổng quan</h1>
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {tiles.map(([k, v]) => (
          <div key={k} className="card"><div className="text-xs text-slate-500">{k}</div><div className="text-2xl font-bold">{v}</div></div>
        ))}
      </div>
      <div className="card">
        <h2 className="mb-2 font-medium">Tác vụ gần đây</h2>
        {d.recent_jobs.length === 0 ? <p className="text-sm text-slate-500">Chưa có job nào. Vào “Tạo ảnh nhân vật” để bắt đầu.</p> : (
          <ul className="divide-y text-sm">
            {d.recent_jobs.map((j) => (
              <li key={j.id} className="flex justify-between py-2"><span className="truncate pr-4">{j.prompt}</span><span className="text-slate-500">{j.state}</span></li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
