export interface JobOutput { asset_id: string; seed: number; index: number }
export interface Job {
  id: string;
  state: "queued" | "running" | "review_required" | "approved" | "rejected" | "failed" | "cancelled";
  error: string | null;
  outputs: JobOutput[];
}

export interface GenerationBody {
  character_version_id?: string | null;
  prompt_original: string;
  prompt_compiled: string | null;
  negative_prompt: string;
  aspect_ratio: "9:16" | "1:1" | "16:9" | "1:1-small";
  num_images: number;
  seed: number;
  steps: number;
  guidance_scale: number;
  lora_version_id?: string;
  lora_scale?: number;
  evaluation?: boolean;
  use_references?: boolean;
  reference_scale?: number;
}

export async function check(r: Response) {
  if (!r.ok) {
    let detail = String(r.status);
    try { detail = (await r.json()).detail ?? detail; } catch { /* ignore */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return r.json();
}

export async function createJob(body: GenerationBody, key: string): Promise<{ job_id: string; state: string }> {
  return check(await fetch("/api/v1/generation-jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json", "Idempotency-Key": key },
    body: JSON.stringify(body),
  }));
}

export async function getJob(id: string): Promise<Job> {
  return check(await fetch(`/api/v1/generation-jobs/${id}`));
}

export const assetUrl = (id: string) => `/api/v1/assets/${id}/content`;

export interface Identity { description_vi: string; identity_en: string; negative_prompt: string }
export interface CharVersion { id: string; version: number; identity: Identity; references: string[]; created_at: string | null }
export interface Character { id: string; code: string; name: string; current_version: CharVersion | null; versions?: CharVersion[] }

export const thumbUrl = (id: string) => `/api/v1/assets/${id}/thumbnail`;

export const listCharacters = async (): Promise<Character[]> => check(await fetch("/api/v1/characters"));
export const getCharacter = async (id: string): Promise<Character> => check(await fetch(`/api/v1/characters/${id}`));

export const post = (url: string, body: unknown) =>
  fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(check);

export const createCharacter = (code: string, name: string, identity: Identity): Promise<Character> =>
  post("/api/v1/characters", { code, name, identity });
export const createVersion = (id: string, body: { identity?: Identity; reference_asset_ids?: string[] }): Promise<Character> =>
  post(`/api/v1/characters/${id}/versions`, body);
export const addReference = (id: string, asset_id: string): Promise<Character> =>
  post(`/api/v1/characters/${id}/references`, { asset_id });


// ---- Lịch sử tạo ảnh ----
export interface HistoryJob {
  id: string; state: Job["state"]; error: string | null; created_at: string | null;
  prompt_original: string; prompt_compiled: string | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  config: Record<string, any>;
  character: { id: string; code: string; name: string; version: number } | null;
  outputs: JobOutput[];
}
export const listJobs = async (p: { limit: number; state?: string; character_id?: string }): Promise<{ total: number; items: HistoryJob[] }> =>
  check(await fetch(`/api/v1/generation-jobs?limit=${p.limit}${p.state ? `&state=${p.state}` : ""}${p.character_id ? `&character_id=${p.character_id}` : ""}`));
