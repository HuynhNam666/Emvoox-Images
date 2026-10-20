export class ApiError extends Error {
  constructor(public code: string, message: string, public status: number) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api/v1${path}`, init);
  if (!res.ok) {
    let code = "http_error";
    let message = `Lỗi ${res.status}`;
    try {
      const body = await res.json();
      const d = body.detail;
      if (d && typeof d === "object" && !Array.isArray(d)) {
        code = d.code ?? code;
        message = d.message ?? message;
      } else if (Array.isArray(d)) {
        code = "validation_error";
        message = d.map((x: { loc: string[]; msg: string }) => `${x.loc.slice(1).join(".")}: ${x.msg}`).join("; ");
      }
    } catch { /* non-JSON error body */ }
    throw new ApiError(code, message, res.status);
  }
  return res.json() as Promise<T>;
}

const json = (method: string, body?: unknown, headers: Record<string, string> = {}): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json", ...headers },
  body: body === undefined ? undefined : JSON.stringify(body),
});

export const api = {
  get: <T,>(p: string) => request<T>(p),
  post: <T,>(p: string, body?: unknown, headers?: Record<string, string>) => request<T>(p, json("POST", body, headers)),
  patch: <T,>(p: string, body: unknown) => request<T>(p, json("PATCH", body)),
  upload: <T,>(p: string, form: FormData) => request<T>(p, { method: "POST", body: form }),
};

export const assetUrl = (id: string, kind: "content" | "thumbnail" | "export" = "thumbnail") =>
  `/api/v1/assets/${id}/${kind}`;

export interface ModelVersion {
  id: string; kind: string; name: string; family: string; hf_id: string | null; revision: string | null;
  license: string | null; status: string; review_status: string; parent_base_id: string | null;
  trigger_token: string | null; capabilities: Record<string, unknown>;
}
export interface Job {
  id: string; state: string; prompt_compiled: string; config: Record<string, number | string>;
  progress_step: number | null; progress_total: number | null; error_code: string | null;
  error_message: string | null; outputs: { asset_id: string; output_index: number; seed: number }[];
}
export interface Character {
  id: string; code: string; name: string;
  current_version: null | { id: string; version: number; trigger_token: string | null;
    references: { asset_id: string; is_canonical: boolean }[] };
}
export interface DatasetOut {
  id: string; name: string; objective: string; trigger_token: string | null; item_count: number;
  approved_count: number; versions: { id: string; version: number; item_count: number; warnings: string[] }[];
}
export interface DatasetItem {
  id: string; asset_id: string; caption: string; status: string; split: string | null;
  quality_flags: string[]; license: string | null;
}
export interface TrainingRun {
  id: string; state: string; current_step: number; total_steps: number | null; error_code: string | null;
  error_message: string | null; trigger_token: string; output_model_version_id: string | null;
  config: Record<string, unknown>;
}
