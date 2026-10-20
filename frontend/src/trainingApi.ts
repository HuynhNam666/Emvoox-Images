import { check, post } from "./api";

export interface ModelVersion {
  id: string; kind: "base" | "character_lora" | "style_lora" | "ip_adapter"; name: string; family: string; model_id: string | null;
  revision: string | null; license: string | null; status: string; trainable: boolean; sha256: string | null;
  character_id: string | null; trigger_token: string | null; metrics: Record<string, unknown>; review_note: string | null;
}
export interface TrainConfig {
  resolution: 512 | 768 | 1024; rank: number; learning_rate: number; max_train_steps?: number; num_train_epochs?: number;
  train_batch_size: number; gradient_accumulation_steps: number; checkpointing_steps: number;
  mixed_precision: "fp16" | "bf16" | "no"; gradient_checkpointing: boolean; use_vae_fp16_fix: boolean; seed: number;
}
export interface RunRequest { dataset_version_id: string; extra_dataset_version_ids?: string[]; base_model_version_id: string; trigger_token: string; objective: "character" | "style"; config: TrainConfig }
export interface Check { name: string; status: "ok" | "warn" | "fail"; message: string }
export interface Preflight { overall: "ok" | "warn" | "fail"; checks: Check[]; resolved_steps: number; n_train: number }
export interface Run {
  id: string; state: string; phase: string | null; step: number; max_steps: number; trigger_token: string; error: string | null;
  config: TrainConfig; output_model_version_id: string | null; validation_asset_ids: string[];
  trainer_revision: string | null; base_revision: string | null; manifest_hash: string | null;
  created_at: string | null; log_tail?: string;
}
export interface Metric { step: number; loss: number; lr: number | null }

export const listModels = async (kind?: string): Promise<ModelVersion[]> =>
  check(await fetch(`/api/v1/models${kind ? `?kind=${kind}` : ""}`));
export const preflight = (r: RunRequest): Promise<Preflight> => post("/api/v1/training-runs/preflight", r);
export const startRun = (r: RunRequest): Promise<Run> => post("/api/v1/training-runs", r);
export const listRuns = async (): Promise<Run[]> => check(await fetch("/api/v1/training-runs"));
export const getRun = async (id: string): Promise<Run> => check(await fetch(`/api/v1/training-runs/${id}?tail=25`));
export const getMetrics = async (id: string): Promise<Metric[]> => check(await fetch(`/api/v1/training-runs/${id}/metrics`));
export const runAction = (id: string, a: "cancel" | "resume" | "validate") => post(`/api/v1/training-runs/${id}/${a}`, {});
export const reviewModel = (id: string, decision: "approved" | "rejected", note: string) =>
  post(`/api/v1/model-versions/${id}/review`, { decision, note });
