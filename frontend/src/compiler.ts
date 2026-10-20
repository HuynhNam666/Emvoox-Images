// Prompt Compiler dạng template (không cần API key). Không dịch tiếng Việt tự do:
// mô tả gốc được giữ nguyên, prompt gửi model được ghép từ preset có cấu trúc + phần tiếng Anh người dùng tự nhập.

export interface Option { vi: string; en: string }
export interface Field { key: FieldKey; label: string; options: Option[] }
export type FieldKey = "emotion" | "action" | "outfit" | "scene" | "camera" | "lighting" | "style";

export const FIELDS: Field[] = [
  { key: "emotion", label: "Cảm xúc", options: [
    { vi: "Bình thường", en: "neutral expression" }, { vi: "Vui", en: "smiling, happy" },
    { vi: "Buồn", en: "sad expression" }, { vi: "Giận", en: "angry expression" },
    { vi: "Ngạc nhiên", en: "surprised expression" }, { vi: "Lo lắng", en: "worried expression" } ] },
  { key: "action", label: "Tư thế / hành động", options: [
    { vi: "Đứng", en: "standing" }, { vi: "Ngồi", en: "sitting" }, { vi: "Đi", en: "walking" },
    { vi: "Chạy", en: "running" }, { vi: "Cúi", en: "bending forward" }, { vi: "Cầm đồ vật", en: "holding an object" } ] },
  { key: "outfit", label: "Trang phục", options: [
    { vi: "Áo sơ mi trắng", en: "white shirt" }, { vi: "Áo dài", en: "Vietnamese ao dai" },
    { vi: "Đồ công sở", en: "business attire" }, { vi: "Áo hoodie", en: "casual hoodie" } ] },
  { key: "scene", label: "Bối cảnh", options: [
    { vi: "Thư viện", en: "in a library" }, { vi: "Đường phố ban ngày", en: "on a city street, daytime" },
    { vi: "Đường phố ban đêm", en: "on a city street at night" }, { vi: "Văn phòng", en: "in an office" },
    { vi: "Phòng ngủ", en: "in a bedroom" } ] },
  { key: "camera", label: "Góc máy", options: [
    { vi: "Cận mặt", en: "close-up portrait" }, { vi: "Bán thân", en: "medium shot, waist up" },
    { vi: "Toàn thân", en: "full body shot" }, { vi: "Nghiêng ¾", en: "three-quarter view" } ] },
  { key: "lighting", label: "Ánh sáng", options: [
    { vi: "Cửa sổ tự nhiên", en: "soft window light" }, { vi: "Hoàng hôn", en: "golden hour light" },
    { vi: "Studio", en: "studio lighting" }, { vi: "Ban đêm", en: "night lighting" } ] },
  { key: "style", label: "Phong cách", options: [
    { vi: "Điện ảnh chân thực", en: "cinematic photorealistic photo" }, { vi: "Anime", en: "anime style illustration" },
    { vi: "Manhua", en: "manhua style illustration" }, { vi: "2D giả 3D", en: "2.5D stylized render" } ] },
];

export type Selection = Partial<Record<FieldKey, string>>; // giá trị = en

export function compilePrompt(identityEn: string, sel: Selection, extraEn: string): string {
  const order: FieldKey[] = ["camera", "emotion", "action", "outfit", "scene", "lighting", "style"];
  const parts = [identityEn.trim(), ...order.map((k) => sel[k]), extraEn.trim()];
  return parts.filter(Boolean).join(", ");
}

export const DEFAULT_NEGATIVE = "blurry, low quality, deformed hands, extra fingers, bad anatomy, watermark, text";
