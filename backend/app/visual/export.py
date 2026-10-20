"""Xuất bản cuối + nhãn AI + C2PA (báo cáo mục 3.6).

Quy trình: hoàn thiện ảnh (về sRGB, đưa về khung đích không kéo méo) → chèn nhãn hiển thị → ký C2PA SAU MỌI thay đổi điểm ảnh
→ đọc lại và xác minh → lưu bản cuối. Nếu cấu hình bắt buộc (REQUIRE_C2PA) mà ký/xác minh lỗi thì bản xuất bị CHẶN, không phát hành.

Nhãn: "Hình ảnh được tạo bằng AI" / "...chỉnh sửa bằng AI". Cỡ chữ/lề (28–36 px ở rộng 1080, lề ≥ 48 px) là thông số nội bộ đề xuất,
KHÔNG phải kích thước bắt buộc của luật. Watermark ẩn: không tự nhúng (cần bộ mã hoá + bộ phát hiện được hỗ trợ); chỉ ghi nhận dấu
của nhà cung cấp nếu có (vd SynthID của Gemini) — không kiểm chứng được sau nén.
C2PA chứng minh nguồn ký và tính toàn vẹn của hồ sơ, không chứng minh cảnh là thật; metadata có thể bị gỡ khi nền tảng xử lý lại.
"""
import io
import json
import os
import tempfile
from pathlib import Path

from PIL import Image, ImageCms, ImageDraw, ImageFont

from app.core.config import settings

TARGETS = {"1080x1920": (1080, 1920), "1920x1080": (1920, 1080)}
MAX_CROP = 0.25
DIGITAL_SOURCE = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf", "/Library/Fonts/Arial.ttf",
]


class ExportError(Exception):
    pass


def _font(px: int) -> ImageFont.FreeTypeFont:
    for f in ([settings.ai_label_font] if settings.ai_label_font else []) + _FONTS:
        if f and os.path.exists(f):
            return ImageFont.truetype(f, px)
    raise ExportError("không tìm thấy font hỗ trợ tiếng Việt (cài fonts-dejavu-core hoặc đặt AI_LABEL_FONT)")


def to_srgb(img: Image.Image) -> Image.Image:
    prof = img.info.get("icc_profile")
    if prof:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(prof))
            return ImageCms.profileToProfile(img, src, ImageCms.createProfile("sRGB"), outputMode="RGB")
        except Exception:  # noqa: BLE001 - profile hỏng: coi như đã là sRGB
            pass
    return img.convert("RGB")


def fit_cover(img: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, float]:
    """Phóng đều rồi cắt giữa (không kéo méo). Trả (ảnh, tỉ lệ phần bị cắt)."""
    tw, th = size
    w, h = img.size
    scale = max(tw / w, th / h)
    nw, nh = max(tw, round(w * scale)), max(th, round(h * scale))
    loss = 1 - (tw * th) / (nw * nh)
    if loss > MAX_CROP:
        raise ExportError(f"khung đích lệch quá nhiều ({loss:.0%} bị cắt > {MAX_CROP:.0%}); tạo lại ảnh đúng tỉ lệ")
    r = img.resize((nw, nh), Image.LANCZOS)
    x, y = (nw - tw) // 2, (nh - th) // 2
    return r.crop((x, y, x + tw, y + th)), round(loss, 4)


def label_layer(size: tuple[int, int], text: str, face_boxes=None) -> tuple[Image.Image, dict]:
    """Lớp RGBA trong suốt chỉ chứa nhãn (dùng cho cả ảnh và video). Nhãn góc trên-trái (tránh vùng phụ đề phía dưới);
    nếu đè lên mặt thì chuyển xuống góc dưới-trái."""
    W, H = size
    scale = W / 1080
    px = max(14, round(32 * scale))          # 28–36 px ở rộng 1080
    margin = round(48 * scale)
    font = _font(px)
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    bbox = probe.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = round(10 * scale)

    def rect(top: bool):
        x0 = margin
        y0 = margin if top else H - margin - th - 2 * pad
        return (x0, y0, x0 + tw + 2 * pad, y0 + th + 2 * pad)

    def hits(r):
        return any(not (r[2] < fb[0] or r[0] > fb[2] or r[3] < fb[1] or r[1] > fb[3]) for fb in (face_boxes or []))

    pos, r = "top-left", rect(True)
    if hits(r) and not hits(rect(False)):
        r, pos = rect(False), "bottom-left"
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    od = ImageDraw.Draw(layer)
    od.rounded_rectangle(r, radius=round(8 * scale), fill=(0, 0, 0, 150))       # nền tối bán trong suốt => tương phản rõ
    od.text((r[0] + pad - bbox[0], r[1] + pad - bbox[1]), text, font=font, fill=(255, 255, 255, 255))
    return layer, {"text": text, "font_px": px, "margin_px": margin, "position": pos, "overlaps_face": hits(r)}


def draw_label(img: Image.Image, text: str, face_boxes=None) -> tuple[Image.Image, dict]:
    layer, info = label_layer(img.size, text, face_boxes)
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB"), info


def render_image(png: bytes, target: str, label_text: str | None, face_boxes=None) -> tuple[bytes, dict]:
    img = Image.open(io.BytesIO(png))
    img.load()
    info: dict = {"source_size": list(img.size)}
    img = to_srgb(img)
    if target != "native":
        if target not in TARGETS:
            raise ExportError(f"target không hỗ trợ: {target}")
        img, loss = fit_cover(img, TARGETS[target])
        info["crop_loss"] = loss
    if label_text:
        img, lab = draw_label(img, label_text, face_boxes)
        info["label"] = lab
    else:
        info["label"] = None
    buf = io.BytesIO()
    img.save(buf, "PNG", icc_profile=None)
    info["size"] = list(img.size)
    return buf.getvalue(), info


def c2pa_configured() -> tuple[bool, str]:
    if not (settings.c2pa_cert_path and settings.c2pa_key_path):
        return False, "chưa cấu hình C2PA_CERT_PATH / C2PA_KEY_PATH"
    if not (Path(settings.c2pa_cert_path).exists() and Path(settings.c2pa_key_path).exists()):
        return False, "không thấy file chứng thư/khóa C2PA"
    try:
        import c2pa  # noqa: F401
    except ImportError:
        return False, "chưa cài c2pa-python"
    return True, "ok"


def manifest_json(title: str, edited: bool, extra: dict) -> str:
    actions = [{"action": "c2pa.created", "digitalSourceType": DIGITAL_SOURCE, "softwareAgent": {"name": "EMVOOX Studio Engine"}}]
    if edited:
        actions.append({"action": "c2pa.edited", "digitalSourceType": DIGITAL_SOURCE, "softwareAgent": {"name": "EMVOOX Studio Engine"}})
    return json.dumps({
        "claim_generator_info": [{"name": "EMVOOX Studio Engine", "version": "1.0"}],
        "title": title,
        "assertions": [
            {"label": "c2pa.actions", "data": {"actions": actions}},
            {"label": "com.emvoox.provenance", "data": extra},
        ],
    })


def sign_and_verify(data: bytes, mime: str, ext: str, manifest: str) -> tuple[bytes | None, dict]:
    """Ký C2PA rồi ĐỌC LẠI từ file đã ký để xác minh. Trả (bytes đã ký | None, báo cáo)."""
    ok, why = c2pa_configured()
    if not ok:
        return None, {"signed": False, "verified": False, "trusted": False, "error": why}
    import c2pa
    cert = Path(settings.c2pa_cert_path).read_bytes()
    key = Path(settings.c2pa_key_path).read_bytes()
    tmp = Path(tempfile.mkdtemp(prefix="emvoox-c2pa-"))
    src, dst = tmp / f"in{ext}", tmp / f"out{ext}"
    src.write_bytes(data)
    try:
        info = c2pa.C2paSignerInfo(c2pa.C2paSigningAlg.ES256, cert, key, settings.c2pa_tsa_url or None)
        with c2pa.Signer.from_info(info) as signer, c2pa.Builder(manifest) as builder:
            builder.sign_file(str(src), str(dst), signer)
        signed = dst.read_bytes()
        with c2pa.Reader(str(dst)) as rd:
            state = rd.get_validation_state()
            active = json.loads(rd.json()).get("active_manifest")
        verified = str(state).lower() in ("valid", "trusted")
        return signed, {"signed": True, "verified": verified, "trusted": str(state).lower() == "trusted",
                        "validation_state": str(state), "active_manifest": active,
                        "note": "chứng thư thử nghiệm => 'Valid' nhưng chưa 'Trusted'" if str(state).lower() == "valid" else ""}
    except Exception as e:  # noqa: BLE001
        return None, {"signed": False, "verified": False, "trusted": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        for f in tmp.glob("*"):
            f.unlink(missing_ok=True)
        tmp.rmdir()


def ffmpeg_bin() -> str | None:
    import shutil
    return settings.ffmpeg_bin or shutil.which("ffmpeg")


def render_video(mp4: bytes, size: tuple[int, int], target: str, label_text: str | None) -> tuple[bytes, dict]:
    """Đưa video về khung đích (cover + cắt giữa, không kéo méo) và chèn nhãn AI bằng ffmpeg; giữ nguyên âm thanh.
    Ký C2PA phải làm SAU bước này (ký sau mọi thay đổi điểm ảnh)."""
    import subprocess
    exe = ffmpeg_bin()
    if not exe:
        raise ExportError("không có ffmpeg (cài ffmpeg hoặc đặt FFMPEG_BIN)")
    tw, th = TARGETS[target] if target in TARGETS else size
    if target in TARGETS:
        loss = 1 - (tw * th) / (max(tw / size[0], th / size[1]) ** 2 * size[0] * size[1])
        if loss > MAX_CROP:
            raise ExportError(f"khung đích lệch quá nhiều ({loss:.0%} bị cắt > {MAX_CROP:.0%})")
    info: dict = {"source_size": list(size), "size": [tw, th], "label": None}
    tmp = Path(tempfile.mkdtemp(prefix="emvoox-vid-"))
    src, dst, png = tmp / "in.mp4", tmp / "out.mp4", tmp / "label.png"
    src.write_bytes(mp4)
    try:
        chain = f"[0:v]scale={tw}:{th}:force_original_aspect_ratio=increase,crop={tw}:{th},setsar=1[v]"
        cmd = [exe, "-y", "-loglevel", "error", "-i", str(src)]
        if label_text:
            layer, info["label"] = label_layer((tw, th), label_text)
            layer.save(png)
            cmd += ["-i", str(png)]
            chain += ";[v][1:v]overlay=0:0:format=auto[outv]"
        else:
            chain += ";[v]null[outv]"
        cmd += ["-filter_complex", chain, "-map", "[outv]", "-map", "0:a?", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                "-c:a", "copy", "-movflags", "+faststart", str(dst)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if r.returncode != 0 or not dst.exists():
            raise ExportError(f"ffmpeg lỗi: {r.stderr[-400:]}")
        return dst.read_bytes(), info
    finally:
        for f in tmp.glob("*"):
            f.unlink(missing_ok=True)
        tmp.rmdir()
