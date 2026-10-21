import { check } from "./api";

const send = (method: "PATCH" | "DELETE" | "POST", url: string, body?: unknown) =>
  fetch(url, { method, headers: body !== undefined ? { "Content-Type": "application/json" } : undefined, body: body !== undefined ? JSON.stringify(body) : undefined }).then(check);

// Nhân vật
export const patchCharacter = (id: string, b: { name?: string; code?: string }) => send("PATCH", `/api/v1/characters/${id}`, b);
export const deleteCharacter = (id: string) => send("DELETE", `/api/v1/characters/${id}`);
// Dataset
export const patchDataset = (id: string, b: { name?: string; source?: string; license?: string; detach_character?: boolean }) => send("PATCH", `/api/v1/datasets/${id}`, b);
export const deleteDataset = (id: string) => send("DELETE", `/api/v1/datasets/${id}`);
export const deleteItems = (datasetId: string, ids: string[]) => send("POST", `/api/v1/datasets/${datasetId}/items/delete`, { item_ids: ids });
// Bối cảnh / trang phục
export interface Costume { id: string; code: string; description_en: string; reference_asset_id: string | null }
export const createScene = (b: { code: string; name: string; spec: Record<string, unknown> }) => send("POST", "/api/v1/scenes", b);
export const newSceneVersion = (code: string, b: { code: string; name: string; spec: Record<string, unknown> }) => send("POST", `/api/v1/scenes/${code}/versions`, b);
export const patchScene = (code: string, name: string) => send("PATCH", `/api/v1/scenes/${code}`, { name });
export const deleteScene = (code: string) => send("DELETE", `/api/v1/scenes/${code}`);
export const listCostumes = async (characterId: string): Promise<Costume[]> => check(await fetch(`/api/v1/characters/${characterId}/costumes`));
export const createCostume = (characterId: string, b: { code: string; description_en: string }) => send("POST", `/api/v1/characters/${characterId}/costumes`, b);
export const patchCostume = (id: string, b: { code?: string; description_en?: string }) => send("PATCH", `/api/v1/costumes/${id}`, b);
export const deleteCostume = (id: string) => send("DELETE", `/api/v1/costumes/${id}`);
// Job
export const deleteVisualJob = (id: string) => send("DELETE", `/api/v1/visual/jobs/${id}`);
export const deleteVideoJob = (id: string) => send("DELETE", `/api/v1/video/jobs/${id}`);
export const deleteGenerationJob = (id: string) => send("DELETE", `/api/v1/generation-jobs/${id}`);
export const deleteBenchmark = (id: string) => send("DELETE", `/api/v1/benchmark/runs/${id}`);
export const deleteExport = (id: string) => send("DELETE", `/api/v1/exports/${id}`);

/** Hộp xác nhận trước thao tác xóa (không thể hoàn tác). */
export const confirmDelete = (what: string) => window.confirm(`Xóa ${what}?\nThao tác này không hoàn tác được.`);

/** Tải một ảnh lên kho asset (JPG/PNG/WebP). Trả id asset. */
export async function uploadAsset(file: File, kind = "reference"): Promise<string> {
  const fd = new FormData();
  fd.append("file", file);
  const r = await check(await fetch(`/api/v1/assets?kind=${kind}`, { method: "POST", body: fd }));
  return r.id as string;
}
