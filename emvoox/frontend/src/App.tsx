import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import Dashboard from "./pages/Dashboard";
import Generate from "./pages/Generate";
import Characters from "./pages/Characters";
import Datasets from "./pages/Datasets";
import Training from "./pages/Training";
import Models from "./pages/Models";

const PAGES = [
  ["dashboard", "Tổng quan", Dashboard],
  ["generate", "Tạo ảnh nhân vật", Generate],
  ["characters", "Thư viện nhân vật", Characters],
  ["datasets", "Dataset", Datasets],
  ["training", "Huấn luyện LoRA", Training],
  ["models", "Model & Cài đặt", Models],
] as const;

type Ready = { database: boolean; redis: boolean; gpu_worker: boolean; image_provider: string };

export default function App() {
  const [page, setPage] = useState<(typeof PAGES)[number][0]>("dashboard");
  const ready = useQuery({ queryKey: ["ready"], queryFn: () => api.get<Ready>("/health/ready"), refetchInterval: 15000 });
  const Current = PAGES.find((p) => p[0] === page)![2];
  const r = ready.data;
  return (
    <div className="flex min-h-screen">
      <aside className="w-56 shrink-0 border-r border-slate-200 bg-white p-3">
        <div className="mb-4 text-lg font-bold text-indigo-700">EMVOOX Studio</div>
        <nav className="space-y-1">
          {PAGES.map(([key, label]) => (
            <button key={key} onClick={() => setPage(key)}
              className={`block w-full rounded px-3 py-2 text-left text-sm ${page === key ? "bg-indigo-50 font-medium text-indigo-700" : "hover:bg-slate-100"}`}>
              {label}
            </button>
          ))}
        </nav>
        <div className="mt-6 space-y-1 text-xs text-slate-600">
          <Dot ok={r?.database} label="Database" />
          <Dot ok={r?.redis} label="Redis" />
          <Dot ok={r?.gpu_worker} label="Worker AI" />
          {r?.image_provider === "mock" && (
            <div className="rounded bg-amber-100 p-2 text-amber-800">Đang chạy provider MOCK — ảnh không phải do model thật tạo.</div>
          )}
          {r && !r.gpu_worker && <div className="text-slate-500">Worker AI không khả dụng: job sẽ chờ trong hàng đợi.</div>}
        </div>
      </aside>
      <main className="flex-1 p-6"><Current /></main>
    </div>
  );
}

function Dot({ ok, label }: { ok?: boolean; label: string }) {
  const c = ok === undefined ? "bg-slate-300" : ok ? "bg-emerald-500" : "bg-red-500";
  return <div className="flex items-center gap-2"><span className={`h-2 w-2 rounded-full ${c}`} />{label}</div>;
}
