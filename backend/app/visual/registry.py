"""Model Registry (báo cáo mục 1.3): danh mục model/tuyến tạo ảnh kèm năng lực, giấy phép, giá, ngày kiểm tra.

Quy ước trung thực:
- `verification="verified"`: trọng số/giấy phép đã đối chiếu trên Hugging Face (20/10/2026) và pipeline có trong diffusers.
- `connector_unverified`: bộ kết nối API đã viết theo tài liệu công khai nhưng CHƯA gọi thử với khóa thật.
- `unverified`: tên/endpoint trong báo cáo chưa xác minh được (vd "FLUX 3 Image", "Nano Banana 2.1").
- `manual`: không có API (Midjourney) — chỉ nhập ảnh thủ công.
Giá là giá công khai tham khảo (từ báo cáo hoặc nguồn thứ cấp), phải đối chiếu trang giá chính thức trước khi dùng để ra quyết định.
"""
from sqlalchemy.orm import Session

from app.models import ProviderModel

CHECKED = "2026-10-20"

# Nhóm tác vụ theo mục 1.3
T_CONCEPT, T_ACTION, T_FACE, T_CLOSE, T_MULTI, T_TEXT, T_SERIES = (
    "concept", "action_simple", "action_face", "closeup", "multi_person", "text_poster", "series_bulk")

MODELS: list[dict] = [
    dict(key="sdxl-1.0", name="SDXL 1.0 (đối chứng, LoRA + IP-Adapter)", vendor="Stability AI", family="sdxl", route="local",
         capabilities=dict(txt2img=True, ref_identity=True, ip_adapter=True, lora=True, training=True, pose_control=False, text_render=False,
                           negative_prompt=True, multi_ref=False, edit=False),
         license="CreativeML Open RAIL++-M", commercial_ok=True, pricing=dict(unit="gpu_second"), min_vram_gb=8,
         tasks=[T_CONCEPT, T_ACTION, T_FACE, T_CLOSE, T_SERIES], verification="verified",
         notes="Đã chạy thật trên RTX 4060 8GB. DPM++ 2M Karras, 25–35 bước, CFG 5–7 (mục 3.2). LoRA SDXL không dùng cho model khác."),
    dict(key="flux2-klein-4b", name="FLUX.2 Klein 4B (distilled, tự chạy)", vendor="Black Forest Labs", family="flux", route="local",
         capabilities=dict(txt2img=True, edit=True, multi_ref=True, ref_identity=True, lora=False, training=False, text_render=False,
                           negative_prompt=False, pose_control=False),
         license="Apache-2.0", commercial_ok=True, pricing=dict(unit="gpu_second"), min_vram_gb=13,
         tasks=[T_CONCEPT, T_ACTION, T_FACE, T_SERIES], verification="verified",
         notes="diffusers Flux2KleinPipeline, 4 bước, guidance 1.0 (tài liệu BFL). Bản distilled; tinh chỉnh dùng bản Base."),
    dict(key="flux2-klein-base-4b", name="FLUX.2 Klein Base 4B (để tinh chỉnh LoRA)", vendor="Black Forest Labs", family="flux", route="local",
         capabilities=dict(txt2img=True, edit=True, multi_ref=True, ref_identity=True, lora=True, training=True, text_render=False,
                           negative_prompt=False, pose_control=False),
         license="Apache-2.0", commercial_ok=True, pricing=dict(unit="gpu_second"), min_vram_gb=16,
         tasks=[T_ACTION, T_FACE, T_SERIES], verification="verified",
         notes="Ứng viên thử nghiệm cho tuyến tự chạy series lớn (mục 1.3) — chưa được chứng minh là tối ưu. Nhiều bước hơn bản distilled."),
    dict(key="qwen-image", name="Qwen-Image (tạo ảnh, chữ tốt)", vendor="Alibaba Qwen", family="qwen", route="local",
         capabilities=dict(txt2img=True, edit=False, multi_ref=False, ref_identity=False, lora=True, training=False, text_render=True,
                           negative_prompt=True, pose_control=False),
         license="Apache-2.0", commercial_ok=True, pricing=dict(unit="gpu_second"), min_vram_gb=24,
         tasks=[T_CONCEPT, T_TEXT], verification="verified", notes="20B tham số: cần GPU lớn hoặc offload CPU (chậm)."),
    dict(key="qwen-image-edit", name="Qwen-Image-Edit 2511 (sửa ảnh, đa tham chiếu)", vendor="Alibaba Qwen", family="qwen", route="local",
         capabilities=dict(txt2img=False, edit=True, multi_ref=True, ref_identity=True, lora=True, training=False, text_render=True,
                           negative_prompt=True, pose_control=False),
         license="Apache-2.0", commercial_ok=True, pricing=dict(unit="gpu_second"), min_vram_gb=24,
         tasks=[T_FACE, T_CLOSE, T_MULTI, T_TEXT], verification="verified", notes="diffusers QwenImageEditPlusPipeline."),
    dict(key="z-image-turbo", name="Z-Image Turbo (nhanh, tự chạy)", vendor="Tongyi-MAI (Alibaba)", family="z-image", route="local",
         capabilities=dict(txt2img=True, edit=False, multi_ref=False, ref_identity=False, lora=True, training=False, text_render=True,
                           negative_prompt=False, pose_control=False),
         license="Apache-2.0", commercial_ok=True, pricing=dict(unit="gpu_second"), min_vram_gb=16,
         tasks=[T_CONCEPT, T_TEXT], verification="verified",
         notes='Báo cáo ghi "Z-Image FLUTurbo": không xác minh được tên này; đối chiếu với Tongyi-MAI/Z-Image-Turbo. Không có tham chiếu danh tính.'),
    dict(key="flux2-klein-4b-api", name="FLUX.2 Klein 4B (API BFL)", vendor="Black Forest Labs", family="flux", route="api",
         capabilities=dict(txt2img=True, edit=True, multi_ref=True, ref_identity=True, negative_prompt=False),
         license="Điều khoản API BFL", commercial_ok=True,
         pricing=dict(unit="megapixel", first_mp=0.014, extra_mp=0.001, note="báo cáo: từ 0,014 USD/ảnh; bảng giá thứ cấp"), min_vram_gb=None,
         tasks=[T_CONCEPT, T_ACTION, T_FACE, T_SERIES], verification="connector_unverified",
         notes="POST {BFL_BASE_URL}/flux-2-klein-4b, header x-key, thăm dò polling_url."),
    dict(key="flux2-pro", name="FLUX.2 Pro (API BFL)", vendor="Black Forest Labs", family="flux", route="api",
         capabilities=dict(txt2img=True, edit=True, multi_ref=True, ref_identity=True, negative_prompt=False, pose_control=False),
         license="Điều khoản API BFL", commercial_ok=True,
         pricing=dict(unit="megapixel", first_mp=0.030, extra_mp=0.015, ref_mp=0.015, note="nguồn thứ cấp, mâu thuẫn giữa các nguồn"),
         min_vram_gb=None, tasks=[T_ACTION, T_FACE, T_MULTI], verification="connector_unverified",
         notes="Đa tham chiếu (input_image, input_image_2...)."),
    dict(key="flux3-image", name="FLUX 3 Image (API BFL, theo báo cáo)", vendor="Black Forest Labs", family="flux", route="api",
         capabilities=dict(txt2img=True, edit=True, multi_ref=True, ref_identity=True, negative_prompt=False),
         license="Điều khoản API BFL", commercial_ok=True,
         pricing=dict(unit="image_by_mp", mp1=0.048, mp4=0.100, note="theo báo cáo EMVOOX1 mục 1.1"), min_vram_gb=None,
         tasks=[T_ACTION, T_FACE, T_CLOSE, T_MULTI], verification="unverified",
         notes="Không xác minh được endpoint công khai: đặt BFL_FLUX3_ENDPOINT khi đã có thì model mới khả dụng."),
    dict(key="gemini-nano-banana", name="Gemini Nano Banana (API Google)", vendor="Google", family="gemini", route="api",
         capabilities=dict(txt2img=True, edit=True, multi_ref=True, ref_identity=True, text_render=True, negative_prompt=False),
         license="Điều khoản Gemini API (đóng)", commercial_ok=True,
         pricing=dict(unit="image_by_res", k1=0.0336, k2=0.0504, note="báo cáo: đầu ra 1K/2K, chưa gồm phí đầu vào"), min_vram_gb=None,
         tasks=[T_FACE, T_CLOSE, T_TEXT, T_ACTION], verification="connector_unverified",
         notes='Báo cáo ghi "Nano Banana 2.1": id chính xác chưa xác minh — đặt GEMINI_IMAGE_MODEL. Mặc định gemini-2.5-flash-image.'),
    dict(key="seedream-4-5", name="Seedream 4.5 (API BytePlus)", vendor="ByteDance BytePlus", family="seedream", route="api",
         capabilities=dict(txt2img=True, edit=True, multi_ref=True, ref_identity=True, text_render=True, negative_prompt=False),
         license="Điều khoản BytePlus ModelArk", commercial_ok=True, pricing=dict(unit="image", per_image=0.04, note="nguồn thứ cấp ~0,04 USD/ảnh"),
         min_vram_gb=None, tasks=[T_ACTION, T_FACE, T_CLOSE, T_TEXT], verification="connector_unverified",
         notes="POST {SEEDREAM_BASE_URL}/images/generations; id model cấu hình SEEDREAM_MODEL."),
    dict(key="gpt-image", name="GPT Image (API OpenAI)", vendor="OpenAI", family="gpt-image", route="api",
         capabilities=dict(txt2img=True, edit=True, multi_ref=True, ref_identity=True, text_render=True, negative_prompt=False),
         license="Điều khoản OpenAI API", commercial_ok=True,
         pricing=dict(unit="image_by_size_quality", medium_sq=0.034, high_sq=0.133, low_sq=0.009, tall_factor=1.5,
                      note="ước tính từ nguồn thứ cấp; thực tế tính theo token"),
         min_vram_gb=None, tasks=[T_TEXT, T_FACE], verification="connector_unverified", notes="/images/generations và /images/edits (image[])."),
    dict(key="midjourney", name="Midjourney (concept, nhập thủ công)", vendor="Midjourney", family="midjourney", route="manual",
         capabilities=dict(txt2img=True, edit=False, multi_ref=False, ref_identity=False), license="Điều khoản Midjourney", commercial_ok=None,
         pricing=dict(unit="subscription", plans_usd=[10, 30, 60, 120]), min_vram_gb=None, tasks=[T_CONCEPT], verification="manual",
         notes="Không có API chính thức/không tự chạy trọng số: tạo ngoài rồi nhập ảnh qua /visual/jobs/{id}/manual-import."),
]


def seed_registry(db: Session) -> None:
    """Thêm model mới; KHÔNG ghi đè cột người dùng đã sửa (enabled, benchmark)."""
    have = {m.key: m for m in db.query(ProviderModel).all()}
    for spec in MODELS:
        m = have.get(spec["key"])
        if m is None:
            db.add(ProviderModel(checked_at=CHECKED, **spec))
        else:
            for k, v in spec.items():
                if k not in ("enabled", "benchmark"):
                    setattr(m, k, v)
            m.checked_at = CHECKED
    db.commit()


def estimate_api_cost(m: ProviderModel, width: int, height: int, n: int, n_refs: int = 0, k: str = "1K") -> float:
    """Ước tính USD cho n ảnh. Chỉ là ước tính trước khi gọi; chi phí ghi nhận lấy từ phản hồi/giá công khai ở provider."""
    p = m.pricing or {}
    mp = width * height / 1e6
    unit = p.get("unit")
    if unit == "megapixel":
        first, extra = p.get("first_mp", 0), p.get("extra_mp", 0)
        per = first + max(0.0, mp - 1.0) * extra + n_refs * p.get("ref_mp", 0) * 1.0
        return round(per * n, 5)
    if unit == "image_by_mp":
        per = p["mp1"] if mp <= 1.2 else (p["mp4"] if mp >= 3.5 else p["mp1"] + (p["mp4"] - p["mp1"]) * (mp - 1) / 3)
        return round(per * n, 5)
    if unit == "image_by_res":
        return round(p["k1" if k == "1K" else "k2"] * n, 5)
    if unit == "image":
        return round(p["per_image"] * n, 5)
    if unit == "image_by_size_quality":
        base = p["medium_sq"] * (p.get("tall_factor", 1.5) if width != height else 1.0)
        return round(base * n, 5)
    return 0.0
