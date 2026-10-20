"""Template-based prompt compiler (no LLM, no API key).

Free Vietnamese text is NOT machine-translated: it is preserved verbatim in `prompt_original`.
Structured fields are mapped through bilingual presets. Free text is appended to the compiled prompt
only when it is already ASCII/English; otherwise a warning tells the user to edit the English prompt.
"""
from dataclasses import dataclass, field

COMPILER_VERSION = "template-1"

PRESETS: dict[str, dict[str, str]] = {
    "hair": {
        "toc_den_ngang_vai": "shoulder-length black hair", "toc_den_dai": "long black hair",
        "toc_ngan": "short hair", "toc_buoc_cao": "hair tied up in a high ponytail", "toc_xoan": "wavy hair",
    },
    "emotion": {
        "vui": "smiling, joyful expression", "buon": "sad, melancholic expression", "tu_tin": "confident expression",
        "ngac_nhien": "surprised expression", "binh_than": "calm neutral expression",
    },
    "camera": {
        "can_canh": "close-up portrait", "ban_than": "medium shot, waist up", "toan_than": "full body shot",
        "goc_thap": "low angle shot", "goc_cao": "high angle shot",
    },
    "lighting": {
        "anh_sang_cua_so": "soft window light", "hoang_hon": "golden hour light", "ban_dem": "night, neon lighting",
        "studio": "studio lighting", "tu_nhien": "natural daylight",
    },
    "style": {
        "dien_anh": "cinematic, photorealistic, film grain", "anime": "anime style, clean line art",
        "tranh_son_dau": "oil painting", "hoat_hinh_3d": "3d animation style",
    },
    "setting": {
        "thu_vien": "in a library", "quan_ca_phe": "in a cafe", "pho_phuong": "on a city street",
        "van_phong": "in an office", "cong_vien": "in a park", "phong_ngu": "in a bedroom",
    },
}

DEFAULT_NEGATIVE = "lowres, blurry, deformed, extra fingers, bad anatomy, watermark, text, cropped"


@dataclass
class CompiledPrompt:
    prompt: str
    negative_prompt: str
    warnings: list[str] = field(default_factory=list)
    version: str = COMPILER_VERSION


def _is_ascii(s: str) -> bool:
    return all(ord(c) < 128 for c in s)


def compile_prompt(fields: dict, *, trigger_token: str | None = None, description_en: str | None = None) -> CompiledPrompt:
    """fields: preset keys per category + optional free text keys (age, outfit, action, extra_en)."""
    parts: list[str] = []
    warnings: list[str] = []
    # order: identity -> action/emotion -> outfit -> setting -> camera -> lighting -> style
    if trigger_token:
        parts.append(trigger_token)
    age = fields.get("age")
    if age:
        parts.append(f"{int(age)} years old person")
    if fields.get("hair"):
        parts.append(PRESETS["hair"].get(fields["hair"], fields["hair"] if _is_ascii(fields["hair"]) else ""))
    for key in ("action", "emotion"):
        v = fields.get(key)
        if v:
            mapped = PRESETS.get(key, {}).get(v)
            if mapped:
                parts.append(mapped)
            elif _is_ascii(v):
                parts.append(v)
            else:
                warnings.append(f"Trường '{key}' là tiếng Việt tự do, chưa được dịch; hãy chọn preset hoặc sửa prompt tiếng Anh.")
    outfit = fields.get("outfit")
    if outfit:
        if _is_ascii(outfit):
            parts.append(f"wearing {outfit}")
        else:
            warnings.append("Trang phục là tiếng Việt tự do, chưa được dịch; hãy nhập tiếng Anh ở prompt đã biên dịch.")
    for key in ("setting", "camera", "lighting", "style"):
        v = fields.get(key)
        if v:
            mapped = PRESETS[key].get(v)
            if mapped:
                parts.append(mapped)
            else:
                warnings.append(f"Giá trị '{v}' của '{key}' không thuộc preset.")
    if description_en:
        parts.append(description_en.strip())
    if not parts:
        warnings.append("Chưa có trường nào dịch được; hãy nhập prompt tiếng Anh trực tiếp.")
    return CompiledPrompt(", ".join(p for p in parts if p), DEFAULT_NEGATIVE, warnings)
