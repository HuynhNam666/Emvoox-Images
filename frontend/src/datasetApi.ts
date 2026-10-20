import { check, post } from "./api";

export type DsGroup = "identity" | "body" | "expression" | "pose" | "outfit" | "appearance" | "style" | "scene";
export type ItemStatus = "pending" | "approved" | "rejected";
export type Split = "train" | "val" | "test";
export interface Dataset { id: string; name: string; group: DsGroup; character_id: string | null; source: string | null; license: string | null; counts: Partial<Record<ItemStatus, number>>; latest_version: number }
export interface DsItem { id: string; asset_id: string; filename: string | null; caption: string; status: ItemStatus; split: Split; synthetic: boolean; flags: string[]; width: number | null; height: number | null }
export interface ImportReport { imported: number; duplicates: string[]; near_duplicates: { file: string; similar_to: string }[]; low_resolution: string[]; rejected: { file: string; reason: string }[] }
export interface DsVersion { id: string; version: number; manifest_hash: string; item_count: number; split_counts: Record<string, number>; warnings: string[]; created_at: string | null }

export const listDatasets = async (): Promise<Dataset[]> => check(await fetch("/api/v1/datasets"));
export const createDataset = (b: { name: string; group: DsGroup; character_id?: string | null; source?: string; license?: string }): Promise<Dataset> => post("/api/v1/datasets", b);
export const listItems = async (id: string, status?: ItemStatus): Promise<{ total: number; items: DsItem[] }> =>
  check(await fetch(`/api/v1/datasets/${id}/items?limit=200${status ? `&status=${status}` : ""}`));
export const patchItem = (id: string, b: Partial<Pick<DsItem, "caption" | "status" | "split" | "synthetic">>): Promise<DsItem> =>
  fetch(`/api/v1/dataset-items/${id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b) }).then(check);
export const bulkItems = (dsId: string, b: { item_ids: string[]; status?: ItemStatus; split?: Split; caption_prepend?: string; caption_append?: string }) =>
  post(`/api/v1/datasets/${dsId}/items/bulk`, b);
export const listVersions = async (id: string): Promise<DsVersion[]> => check(await fetch(`/api/v1/datasets/${id}/versions`));
export const createVersion_ = (id: string, harmonize: boolean): Promise<DsVersion> => post(`/api/v1/datasets/${id}/versions`, { harmonize_splits: harmonize });
export const manifestUrl = (versionId: string) => `/api/v1/dataset-versions/${versionId}/manifest`;

export function uploadFiles(dsId: string, files: File[], onProgress: (pct: number) => void): Promise<ImportReport> {
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    const x = new XMLHttpRequest();
    x.open("POST", `/api/v1/datasets/${dsId}/uploads`);
    x.upload.onprogress = (e) => e.lengthComputable && onProgress(Math.round((e.loaded / e.total) * 100));
    x.onload = () => (x.status < 300 ? resolve(JSON.parse(x.responseText)) : reject(new Error(x.responseText || String(x.status))));
    x.onerror = () => reject(new Error("lỗi mạng khi upload"));
    x.send(fd);
  });
}
