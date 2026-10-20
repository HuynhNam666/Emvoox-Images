import React from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";

import "./styles.css";
import { Generate } from "./Generate";
import { Library } from "./Library";
import { Datasets } from "./Datasets";
import { Training } from "./Training";
import { History } from "./History";
import { Video } from "./Video";
import { Visual } from "./Visual";
import { Models, Reports } from "./Reports";
import { Benchmark } from "./Benchmark";
import { Scenes } from "./Scenes";
import { authStatus, login } from "./visualApi";
import type { HistoryJob } from "./api";
import { Icon, type IconName } from "./ui";

const qc = new QueryClient();

type Tab = "visual" | "gen" | "hist" | "video" | "lib" | "ds" | "train" | "scenes" | "models" | "bench" | "report";
const NAV: { key: Tab; label: string; icon: IconName }[] = [
  { key: "visual", label: "Ảnh theo shot", icon: "film" },
  { key: "gen", label: "Tạo ảnh (SDXL)", icon: "sparkles" },
  { key: "hist", label: "Lịch sử", icon: "clock" },
  { key: "video", label: "Video", icon: "film" },
  { key: "lib", label: "Nhân vật", icon: "users" },
  { key: "ds", label: "Dataset", icon: "database" },
  { key: "train", label: "Training", icon: "cpu" },
  { key: "scenes", label: "Bối cảnh & trang phục", icon: "image" },
  { key: "models", label: "Model", icon: "layers" },
  { key: "bench", label: "Benchmark", icon: "star" },
  { key: "report", label: "Báo cáo", icon: "clock" },
];

function safeGet(key: string): string | null {
  try { return localStorage.getItem(key); } catch { return null; }
}
function safeSet(key: string, value: string) {
  try { localStorage.setItem(key, value); } catch { /* chế độ riêng tư: bỏ qua */ }
}

function Health() {
  const { data, error, isLoading } = useQuery({
    queryKey: ["ready"],
    queryFn: async () => {
      const r = await fetch("/api/v1/health/ready");
      if (!r.ok) throw new Error(String(r.status));
      return r.json() as Promise<{ status: string }>;
    },
    retry: false,
    refetchInterval: 15000,
  });
  const api = isLoading ? "wait" : error ? "bad" : "ok";
  return (
    <div className="status-card" aria-live="polite">
      <div className="status-row"><span>API &amp; PostgreSQL</span><span className={`dot ${api}`} title={error ? "API/DB chưa sẵn sàng" : data?.status} /></div>
      <div className="status-row"><span>Worker GPU</span><span style={{ color: "#8b90b8" }}>kiểm tra khi chạy</span></div>
    </div>
  );
}

function App() {
  const [tab, setTab] = React.useState<Tab>(() => (safeGet("emvoox.tab") as Tab) || "visual");
  const [theme, setTheme] = React.useState<"light" | "dark">(() => {
    const saved = safeGet("emvoox.theme");
    if (saved === "light" || saved === "dark") return saved;
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });
  React.useEffect(() => { document.documentElement.dataset.theme = theme; safeSet("emvoox.theme", theme); }, [theme]);
  const [visited, setVisited] = React.useState<Set<Tab>>(() => new Set([tab]));
  // "Dùng lại tham số" từ Lịch sử: gửi job sang trang Tạo ảnh (n tăng để Generate nhận biết lần mới)
  const [reuse, setReuse] = React.useState<{ n: number; job: HistoryJob } | null>(null);
  const pages: Record<Tab, React.ReactNode> = {
    visual: <Visual />,
    gen: <Generate reuse={reuse} />,
    hist: <History onReuse={(job) => { setReuse({ n: Date.now(), job }); go("gen"); }} />,
    video: <Video />,
    lib: <Library />, ds: <Datasets />, train: <Training />,
    scenes: <Scenes />, models: <Models />, bench: <Benchmark />, report: <Reports />,
  };
  const go = (t: Tab) => { setTab(t); setVisited((v) => (v.has(t) ? v : new Set(v).add(t))); safeSet("emvoox.tab", t); };

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Icon name="sparkles" /></div>
          <div>
            <div className="brand-name">EMVOOX</div>
            <div className="brand-sub">Character Studio</div>
          </div>
        </div>
        <nav className="nav" aria-label="Điều hướng chính">
          <div className="nav-label">Không gian làm việc</div>
          {NAV.map((n) => (
            <button key={n.key} className="nav-item" aria-current={tab === n.key ? "page" : undefined} onClick={() => go(n.key)}>
              <Icon name={n.icon} /><span>{n.label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-foot">
          <Health />
          <button className="theme-toggle" onClick={() => setTheme(theme === "dark" ? "light" : "dark")} aria-label="Đổi giao diện sáng/tối">
            <Icon name={theme === "dark" ? "sun" : "moon"} className="sm" />{theme === "dark" ? "Giao diện sáng" : "Giao diện tối"}
          </button>
        </div>
      </aside>
      <main className="main">
        {/* Giữ các trang đã mở luôn tồn tại (chỉ ẩn) để không mất form/kết quả/job đang chạy khi đổi tab. */}
        {(Object.keys(pages) as Tab[]).map((k) => visited.has(k) && (
          <div key={k} hidden={tab !== k}>{pages[k]}</div>
        ))}
      </main>
    </div>
  );
}

function Gate() {
  const st = useQuery({ queryKey: ["auth"], queryFn: authStatus, retry: false });
  const [token, setToken] = React.useState("");
  const [msg, setMsg] = React.useState<string | null>(null);
  if (st.isLoading) return <div className="row" style={{ padding: 24 }}>Đang kiểm tra phiên…</div>;
  if (st.data?.required && !st.data.authenticated) {
    const go = async (e: React.FormEvent) => {
      e.preventDefault();
      try { await login(token); setMsg(null); st.refetch(); } catch (x) { setMsg((x as Error).message); }
    };
    return (
      <form onSubmit={go} style={{ maxWidth: 360, margin: "15vh auto", display: "grid", gap: 12 }}>
        <h2>EMVOOX Studio</h2>
        <label className="field">Token truy cập
          <input type="password" value={token} onChange={(e) => setToken(e.target.value)} autoFocus autoComplete="current-password" /></label>
        {msg && <div role="alert" style={{ color: "var(--bad)" }}>{msg}</div>}
        <button className="btn primary" type="submit" disabled={!token}>Đăng nhập</button>
      </form>
    );
  }
  return <App />;
}

createRoot(document.getElementById("root")!).render(
  <QueryClientProvider client={qc}>
    <Gate />
  </QueryClientProvider>,
);
