import hashlib
import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import get_settings

ALLOWED_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


class ImageRejected(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class ImageInfo:
    width: int
    height: int
    mime_type: str
    sha256: str
    phash: str


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dhash(img: Image.Image, size: int = 8) -> str:
    """64-bit difference hash as 16 hex chars."""
    g = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = list(g.getdata())
    bits = 0
    for row in range(size):
        for col in range(size):
            bits = (bits << 1) | (1 if px[row * (size + 1) + col] > px[row * (size + 1) + col + 1] else 0)
    return f"{bits:016x}"


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def inspect_image(data: bytes, *, enforce_min_side: bool = False) -> ImageInfo:
    s = get_settings()
    if len(data) > s.max_upload_bytes:
        raise ImageRejected("file_too_large", "File vượt quá giới hạn dung lượng.")
    Image.MAX_IMAGE_PIXELS = s.max_image_pixels
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            if fmt not in ALLOWED_FORMATS:
                raise ImageRejected("unsupported_format", f"Định dạng {fmt} không được hỗ trợ (JPG/PNG/WebP).")
            w, h = probe.size
            if w * h > s.max_image_pixels:
                raise ImageRejected("too_many_pixels", "Ảnh có quá nhiều điểm ảnh.")
            probe.verify()
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            ph = dhash(ImageOps.exif_transpose(img))
    except ImageRejected:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, SyntaxError) as e:
        raise ImageRejected("corrupt_image", f"Ảnh hỏng hoặc không đọc được: {e}") from e
    if enforce_min_side and min(w, h) < s.min_image_side:
        raise ImageRejected("image_too_small", f"Cạnh ngắn nhất {min(w, h)}px < ngưỡng {s.min_image_side}px.")
    return ImageInfo(w, h, ALLOWED_FORMATS[fmt], sha256_hex(data), ph)


def make_thumbnail(data: bytes, max_side: int = 384) -> bytes:
    with Image.open(io.BytesIO(data)) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        img.save(out, "JPEG", quality=82)
        return out.getvalue()


def export_fill(data: bytes, width: int, height: int) -> bytes:
    """Resize + center-crop to exactly width x height while preserving aspect (no stretching)."""
    with Image.open(io.BytesIO(data)) as img:
        img = ImageOps.fit(img.convert("RGB"), (width, height), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        img.save(out, "PNG")
        return out.getvalue()
