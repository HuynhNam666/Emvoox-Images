"""Prompt Compiler (báo cáo mục 3.4–3.5): scene_spec + tài sản có phiên bản => prompt/cấu hình theo từng model.

Thứ tự (mục 3.2): nhận dạng → hành động/cảm xúc → trang phục → bối cảnh → cỡ cảnh/góc máy → ánh sáng → phong cách → yếu tố cần giữ.
Mô tả ổn định (nhận dạng, trang phục, bối cảnh) tách khỏi nội dung thay đổi (hành động, cảm xúc, camera) để tái sử dụng qua các shot.
Mã A01/B02 chỉ là mã quản trị: model không hiểu, nên luôn được thay bằng mô tả + ảnh tham chiếu thật.
"""
from dataclasses import dataclass, field


@dataclass
class Context:
    """Ngữ cảnh đã resolve và ghim phiên bản của một shot (lưu trong VisualJob.scene_spec['_resolved'])."""
    characters: list[dict] = field(default_factory=list)   # {code,name,version,identity_en,negative,costume_en,position,refs:[{asset_id,role,label}]}
    scene: dict | None = None                               # {code,name,version,text,refs:[...]}
    action: str = ""
    emotion: str = ""
    camera: dict = field(default_factory=dict)             # {size, angle}
    lighting: str = ""
    style: str = ""
    keep: list[str] = field(default_factory=list)
    text_in_image: str = ""
    extra_refs: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @staticmethod
    def from_dict(d: dict) -> "Context":
        return Context(**{k: d.get(k, v) for k, v in Context().__dict__.items()})


def _join(parts: list[str]) -> str:
    return " ".join(p.strip().rstrip(".") + "." for p in parts if p and p.strip())


def ref_plan(ctx: Context, max_refs: int) -> list[dict]:
    """Danh sách tham chiếu theo thứ tự ưu tiên (mặt trước, trang phục, bối cảnh), cắt theo giới hạn của model.
    Mỗi phần tử: {asset_id, role, label}. Thứ tự này cũng là 'Image N' trong câu gán vai trò."""
    refs: list[dict] = []
    for c in ctx.characters:
        refs.extend(c.get("refs", [])[:2])
    for c in ctx.characters:
        if c.get("costume_ref"):
            refs.append(c["costume_ref"])
    if ctx.scene and ctx.scene.get("refs"):
        refs.extend(ctx.scene["refs"][:1])
    refs.extend(ctx.extra_refs)
    seen, out = set(), []
    for r in refs:
        if r["asset_id"] not in seen:
            seen.add(r["asset_id"])
            out.append(r)
    return out[:max_refs]


def compile_prompt(ctx: Context, caps: dict, refs: list[dict], instruction: str = "") -> tuple[str, str]:
    """Trả (prompt, negative_prompt). negative chỉ có nghĩa với model hỗ trợ (caps.negative_prompt)."""
    parts: list[str] = []
    # 0) Gán vai trò từng ảnh tham chiếu (chỉ model đa tham chiếu): tránh trộn đặc điểm giữa nhân vật.
    if caps.get("multi_ref") and refs:
        lines = []
        for i, r in enumerate(refs, 1):
            lines.append(f"Image {i}: {r.get('label') or r['role']}")
        parts.append("Reference images — " + "; ".join(lines) + ". Keep each character's face, hair and body exactly as in their own reference; do not mix traits between characters")
    # 1) nhận dạng
    multi = len(ctx.characters) > 1
    for i, c in enumerate(ctx.characters, 1):
        pos = f" ({c['position']})" if c.get("position") else ""
        prefix = f"Character {i}{pos}: " if multi else ""
        parts.append(prefix + c["identity_en"])
    # 2) hành động + cảm xúc
    parts.append(ctx.action)
    if ctx.emotion:
        parts.append(f"Expression: {ctx.emotion}")
    # 3) trang phục
    costumes = [c["costume_en"] for c in ctx.characters if c.get("costume_en")]
    if costumes:
        parts.append("Wearing: " + "; ".join(costumes))
    # 4) bối cảnh
    if ctx.scene:
        parts.append(ctx.scene["text"])
    # 5) cỡ cảnh + góc máy
    cam = ctx.camera or {}
    if cam.get("size") or cam.get("angle"):
        parts.append(", ".join(x for x in (cam.get("size"), cam.get("angle")) if x))
    # 6) ánh sáng
    if ctx.lighting:
        parts.append(ctx.lighting)
    # 7) phong cách
    if ctx.style:
        parts.append(ctx.style)
    # 8) yếu tố cần giữ
    keep = list(ctx.keep)
    if ctx.characters:
        keep = ["facial identity", "hairstyle", "clothing"] + keep if not keep else keep
    if keep:
        parts.append("Preserve: " + ", ".join(keep))
    if ctx.text_in_image:
        parts.append(f'Visible text exactly: "{ctx.text_in_image}"')
    if instruction:
        parts.append(f"Correction: {instruction}")
    prompt = _join(parts)
    neg = ""
    if caps.get("negative_prompt"):
        negs = [c.get("negative", "") for c in ctx.characters if c.get("negative")]
        neg = ", ".join(dict.fromkeys(x.strip() for n in negs for x in n.split(",") if x.strip())) or \
            "blurry, deformed, extra fingers, bad anatomy, watermark, text artifacts, duplicate person"
    return prompt, neg
