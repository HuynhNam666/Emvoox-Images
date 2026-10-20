"""Kiểm tra ảnh, dHash, và đọc ZIP an toàn (không bao giờ ghi file ra đĩa theo tên trong ZIP)."""
import io
import posixpath
import zipfile
from dataclasses import dataclass

from PIL import Image

from app.core.config import settings

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


class ImportError_(ValueError):
    pass


@dataclass
class ImageInfo:
    mime: str
    width: int
    height: int
    phash: str


def inspect_image(data: bytes) -> ImageInfo:
    if len(data) > settings.max_upload_bytes:
        raise ImportError_("file quá lớn")
    try:
        probe = Image.open(io.BytesIO(data))
        w, h = probe.size
        if w * h > settings.max_image_pixels:
            raise ImportError_("ảnh vượt giới hạn số pixel")
        probe.verify()
        img = Image.open(io.BytesIO(data))
        mime = Image.MIME.get(img.format or "")
        if mime not in ALLOWED_MIME:
            raise ImportError_("chỉ nhận JPG/PNG/WebP")
        return ImageInfo(mime, w, h, dhash(img))
    except ImportError_:
        raise
    except Exception:
        raise ImportError_("file không phải ảnh hợp lệ")


def dhash(img: Image.Image) -> str:
    g = img.convert("L").resize((9, 8), Image.LANCZOS)
    px = list(g.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return f"{bits:016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def read_zip(data: bytes) -> list[tuple[str, bytes, str | None]]:
    """Trả [(tên_file_cơ_sở, bytes_ảnh, caption|None)]. Ném ImportError_ nếu ZIP nguy hiểm."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ImportError_("ZIP hỏng")
    infos = [i for i in zf.infolist() if not i.is_dir()]
    if len(infos) > settings.max_zip_entries:
        raise ImportError_(f"ZIP có quá {settings.max_zip_entries} file")
    if sum(i.file_size for i in infos) > settings.max_zip_uncompressed_bytes:
        raise ImportError_("tổng dung lượng giải nén vượt giới hạn")

    texts: dict[str, str] = {}
    images: list[tuple[str, str, bytes]] = []
    budget = settings.max_zip_uncompressed_bytes
    for i in infos:
        name = i.filename.replace("\\", "/")
        parts = name.split("/")
        if name.startswith("/") or ".." in parts or (len(name) > 1 and name[1] == ":"):
            raise ImportError_(f"đường dẫn không an toàn trong ZIP: {i.filename}")
        if (i.external_attr >> 16) & 0o170000 == 0o120000:  # symlink
            raise ImportError_(f"ZIP chứa symlink: {i.filename}")
        if parts[-1].startswith(".") or "__MACOSX" in parts:
            continue
        base = posixpath.basename(name)
        stem, ext = posixpath.splitext(base)
        ext = ext.lower()
        if ext not in IMAGE_EXT and ext != ".txt":
            continue
        with zf.open(i) as f:  # đọc có giới hạn, không tin file_size trong header
            blob = f.read(min(settings.max_upload_bytes, budget) + 1)
        if len(blob) > settings.max_upload_bytes:
            raise ImportError_(f"file quá lớn trong ZIP: {i.filename}")
        budget -= len(blob)
        if budget < 0:
            raise ImportError_("tổng dung lượng giải nén vượt giới hạn")
        key = posixpath.join(posixpath.dirname(name), stem)
        if ext == ".txt":
            texts[key] = blob.decode("utf-8", errors="replace").strip()
        else:
            images.append((key, base, blob))
    return [(base, blob, texts.get(key)) for key, base, blob in images]
