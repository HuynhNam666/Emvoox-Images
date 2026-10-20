import { check, post } from "./api";

export interface Segment { start: number; end: number; prompt: string }
export type RejectReason =
  | "keyframe_identity" | "motion_identity" | "wrong_action" | "interaction_error" | "visual_artifact" | "audio_issue" | "other";

export interface VideoAttempt {
  id: string; attempt_no: number; kind: "initial" | "repair"; state: string; prompt_id: string | null;
  output_asset_id: string | null; error: string | null; total_seconds: number | null; cost: number;
  params: Record<string, unknown>; review: { decision?: string; reason?: string; note?: string };
}
export interface VideoJob {
  id: string; state: string; error: string | null; source_keyframe_asset_id: string; source_generation_job_id: string | null;
  global_prompt: string; motion_segments: Segment[]; continuity_constraints: string[];
  config: { width: number; height: number; frame_count: number; fps: number; seed: number; duration_s: number };
  workflow_version: string; model_version: string | null; audio_mode: string; budget_limit: number; cost_spent: number;
  repair_count: number; max_repairs: number; output_asset_id: string | null; review_reason: string | null; created_at: string | null;
  attempts?: VideoAttempt[];
}
export interface VideoRequest {
  source_keyframe_asset_id: string; global_prompt: string; motion_segments: Segment[]; continuity_constraints?: string[];
  width: number; height: number; frame_count: number; fps: number; seed: number; audio_mode: "generated" | "none"; budget_limit: number;
}
export interface VideoMetrics {
  attempts_total: number; attempts_rendered: number; attempts_failed: number; reviewed: number; approved: number;
  approval_rate: number | null; latency_s: { p50: number | null; p95: number | null; n: number };
  total_cost: number; cost_per_approved: number | null; reject_reasons: Record<string, number>;
}
export interface VideoHealth {
  ok: boolean; comfyui_url: string; workflow_version?: string; workflow_verified?: boolean;
  missing_nodes?: string[]; warning?: string; error?: string;
}

export const createVideo = (b: VideoRequest, key: string): Promise<{ job_id: string; state: string }> =>
  fetch("/api/v1/video/jobs", {
    method: "POST", headers: { "Content-Type": "application/json", "Idempotency-Key": key }, body: JSON.stringify(b),
  }).then(check);
export const listVideos = async (): Promise<{ total: number; items: VideoJob[] }> => check(await fetch("/api/v1/video/jobs?limit=30"));
export const getVideo = async (id: string): Promise<VideoJob> => check(await fetch(`/api/v1/video/jobs/${id}`));
export const reviewVideo = (id: string, body: { decision: "approve" | "reject"; reason?: RejectReason; note?: string }): Promise<VideoJob> =>
  post(`/api/v1/video/jobs/${id}/review`, body);
export const retryVideo = (id: string, body: { seed?: number }): Promise<unknown> => post(`/api/v1/video/jobs/${id}/retry`, body);
export const videoMetrics = async (): Promise<VideoMetrics> => check(await fetch("/api/v1/video/metrics"));
export const videoHealth = async (): Promise<VideoHealth> => check(await fetch("/api/v1/video/health"));
export const reviewImageJob = (id: string, decision: "approved" | "rejected"): Promise<unknown> =>
  post(`/api/v1/generation-jobs/${id}/review`, { decision });
