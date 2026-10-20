import { check, post } from "./api";

export interface QcCheck { name: string; status: "pass" | "warn" | "fail" | "skipped"; value: number | string | null; threshold: number | null; message: string }
export interface Candidate {
  id: string; attempt_id: string; asset_id: string; seed: number | null; qc_verdict: "pass" | "review" | "fail" | "pending"; qc_score: number | null;
  qc: { verdict?: string; score?: number; checks?: QcCheck[] };
}
export interface Metrics {
  wall_s?: number; queue_s?: number; infer_s?: number; load_s?: number; per_image_s?: number; steps?: number; n_images?: number;
  gpu_name?: string; gpu_total_mb?: number; gpu_peak_mb?: number; torch_vram_peak_mb?: number; ram_peak_mb?: number; rss_peak_mb?: number;
  uncertain_billing?: boolean; cpu_offload?: boolean;
}
export interface Attempt {
  id: string; no: number; kind: string; model_key: string; route: string; state: string; error: string | null; instruction: string | null;
  cost_api: number; cost_gpu: number; metrics: Metrics; remote_id: string | null;
}
export interface PlanItem { model_key: string; route: string; est_cost: number; preferred: boolean; verification: string; reason: string }
export interface VisualJob {
  id: string; project_id: string; episode_id: string | null; shot_id: string; state: string; error: string | null; aspect_ratio: string;
  num_variants: number; seed: number; pinned_model_key: string | null; budget_limit: number; cost_spent: number; repair_count: number;
  max_repairs: number; auto_repair: boolean; selected_candidate_id: string | null; reviewer: string | null; reviewed_at: string | null;
  review_note: string | null; created_at: string | null; thumb_asset_id?: string | null; model_key?: string | null;
  prompt?: string; negative_prompt?: string; route_plan?: PlanItem[]; task?: string; continuity_warnings?: string[];
  characters?: { code: string; version: number; costume: string | null }[]; scene?: { code: string; name: string; version: number } | null;
  attempts?: Attempt[]; candidates?: Candidate[]; review_history?: { at: string; reviewer: string; decision: string; note?: string; reason?: string }[];
}
export interface ModelInfo {
  key: string; name: string; vendor: string; family: string; route: "local" | "api" | "manual"; capabilities: Record<string, boolean>;
  license: string | null; commercial_ok: boolean | null; pricing: Record<string, unknown>; min_vram_gb: number | null; tasks: string[];
  verification: string; checked_at: string | null; notes: string | null; enabled: boolean; benchmark: Record<string, unknown>;
  available: boolean; availability: string;
}
export interface ExportRec {
  id: string; job_id: string; kind: string; status: "released" | "blocked"; target: string; export_asset_id: string | null;
  label: { text?: string; font_px?: number; margin_px?: number; position?: string } | null;
  c2pa: { signed?: boolean; verified?: boolean; trusted?: boolean; validation_state?: string; error?: string; note?: string };
  invisible_watermark: string; sha256: string | null;
}
export interface SceneInfo { code: string; name: string; current_version: number | null; versions: { id: string; version: number; spec: Record<string, unknown> }[] }

export const listVisual = async (state?: string): Promise<{ total: number; items: VisualJob[] }> =>
  check(await fetch(`/api/v1/visual/jobs?limit=40${state ? `&state=${state}` : ""}`));
export const getVisual = async (id: string): Promise<VisualJob> => check(await fetch(`/api/v1/visual/jobs/${id}`));
export const listModels = async (): Promise<{ worker: Record<string, unknown> | null; models: ModelInfo[] }> => check(await fetch("/api/v1/visual/models"));
export const patchModel = (key: string, enabled: boolean) =>
  fetch(`/api/v1/visual/models/${key}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ enabled }) }).then(check);
export const listScenes = async (): Promise<SceneInfo[]> => check(await fetch("/api/v1/scenes"));
export const createVisual = (body: unknown, key: string): Promise<{ job_id: string; state: string; route_plan?: PlanItem[]; excluded?: { model_key: string; reason: string }[] }> =>
  fetch("/api/v1/visual/jobs", { method: "POST", headers: { "Content-Type": "application/json", "Idempotency-Key": key }, body: JSON.stringify(body) }).then(check);
export const reviewVisual = (id: string, body: Record<string, unknown>): Promise<VisualJob> => post(`/api/v1/visual/jobs/${id}/review`, body);
export const exportVisual = (id: string, body: { target: string; label: boolean }): Promise<ExportRec> => post(`/api/v1/visual/jobs/${id}/export`, body);
export const exportVideo = (id: string, body: { target: string; label: boolean }): Promise<ExportRec> => post(`/api/v1/video/jobs/${id}/export`, body);
export const exportFileUrl = (id: string) => `/api/v1/exports/${id}/file`;

export interface ModelReport {
  model_key: string; attempts: number; failed: number; images: number; approved: number; approval_rate: number | null; fail_reasons: Record<string, number>;
  cost: { c_create: number; c_repair: number; c_gpu: number; c_qa_alloc: number; c_storage_alloc: number; c_training_alloc: number; total: number; per_approved: number | null };
  gpu_hours: number;
  latency_s: Record<"wall" | "queue" | "infer" | "per_image", { n: number; p50: number | null; p95: number | null; max: number | null }>;
  memory_mb: Record<"gpu_peak" | "torch_vram_peak" | "ram_peak_system" | "process_rss_peak", { n: number; p50: number | null; p95: number | null; max: number | null }>;
}
export interface Report {
  generated_at: string;
  visual: { formula: string; rates: { note: string }; totals: Record<string, number>; models: ModelReport[] };
  video: {
    attempts: number; rendered: number; failed: number; reviewed: number; approved: number; approval_rate: number | null;
    latency_s: Record<string, { n: number; p50: number | null; p95: number | null; max: number | null }>;
    memory_mb: Record<string, { n: number; max: number | null }>; gpu_hours: number; cost: { total: number; per_approved: number | null; unit: string };
  };
  system: { gpus: { name: string; total_mb: number; used_mb: number; util: number }[]; ram?: { total_mb: number; used_mb: number; available_mb: number };
    cpu_count?: number; disks: Record<string, { total_gb: number; free_gb: number } | null> };
}
export const getReport = async (): Promise<Report> => check(await fetch("/api/v1/reports/summary"));
export const getSystem = async (): Promise<{ api_host: Report["system"]; worker: Record<string, unknown> | null }> => check(await fetch("/api/v1/system/stats"));

export interface BenchRun { id: string; name: string; round_no: number; model_keys: string[]; variants: number; scenes: number }
export interface BenchModelRow {
  model_key: string; images: number; rated_by_two: number; passed: number; core_errors: number; pass_rate: number | null; mean_weighted: number | null;
  qc_auto_pass: number; cost: number; cost_per_pass: number | null; latency_s: { p50: number | null; p95: number | null }; gpu_peak_mb_max: number | null;
}
export interface BenchDetail {
  id: string; name: string; round_no: number; variants: number; jobs: { total: number; done: number; failed: number }; images: number; ratings: number;
  ratings_needed: number; pass_rule: string; weights: Record<string, number>; models: BenchModelRow[];
}
export interface NextRating {
  done: boolean; remaining?: number; candidate_id?: string; asset_id?: string;
  brief?: { group: string; action: string; emotion: string; camera: { size: string; angle: string }; lighting: string; characters: string[] };
}
export const listBench = async (): Promise<BenchRun[]> => check(await fetch("/api/v1/benchmark/runs"));
export const getBench = async (id: string): Promise<BenchDetail> => check(await fetch(`/api/v1/benchmark/runs/${id}`));
export const createBench = (body: unknown): Promise<{ run_id: string; jobs_created: number; images_expected: number; estimated_cost: number; skipped: { model_key: string; scene?: string; reason: string }[] }> =>
  post("/api/v1/benchmark/runs", body);
export const nextRating = async (id: string, rater: string): Promise<NextRating> => check(await fetch(`/api/v1/benchmark/runs/${id}/next?rater=${encodeURIComponent(rater)}`));
export const sendRating = (body: unknown) => post("/api/v1/benchmark/ratings", body);
export const applyBench = (id: string): Promise<{ applied: string[] }> => post(`/api/v1/benchmark/runs/${id}/apply?min_rated=20`, {});

export const authStatus = async (): Promise<{ required: boolean; authenticated: boolean }> => check(await fetch("/api/v1/auth/status"));
export const login = (token: string) => post("/api/v1/auth/login", { token });
