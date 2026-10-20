"""Giao diện chung cho mọi tuyến tạo ảnh (local diffusers, API ngoài, nhập thủ công)."""
import base64
import io
import json
import socket
import time
from dataclasses import dataclass, field
from typing import Callable
from urllib import error as urlerror, request

from PIL import Image


class ProviderError(Exception):
    """code: not_configured | unavailable | http_<n> | timeout | moderated | bad_response | cuda_oom | error.
    retryable: có thể gọi lại cùng tuyến (lỗi trước khi yêu cầu được xử lý). uncertain_billing: có thể đã bị tính phí."""

    def __init__(self, code: str, msg: str, retryable: bool = False, uncertain_billing: bool = False):
        super().__init__(msg)
        self.code, self.retryable, self.uncertain_billing = code, retryable, uncertain_billing


@dataclass
class Ref:
    data: bytes
    role: str = "identity"          # identity | body | costume | scene | pose
    asset_id: str | None = None
    label: str = ""


@dataclass
class ImageRequest:
    prompt: str
    negative_prompt: str = ""
    width: int = 768
    height: int = 1344
    aspect_ratio: str = "9:16"
    seed: int = 42
    n: int = 1
    refs: list[Ref] = field(default_factory=list)
    steps: int | None = None
    guidance: float | None = None
    lora_path: str | None = None
    lora_scale: float = 1.0
    ip_scale: float = 0.6


@dataclass
class ImageOut:
    data: bytes
    seed: int | None
    width: int
    height: int
    meta: dict = field(default_factory=dict)


@dataclass
class GenResult:
    images: list[ImageOut]
    cost_api: float = 0.0
    remote_ids: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)       # steps, vram_peak_torch_mb, infer_s, download_s ...


@dataclass
class Hooks:
    on_remote_id: Callable[[list[str]], None] = lambda ids: None
    resume_ids: list[str] = field(default_factory=list)


class Provider:
    key = ""

    def available(self) -> tuple[bool, str]:
        raise NotImplementedError

    def generate(self, req: ImageRequest, hooks: Hooks) -> GenResult:
        raise NotImplementedError


# ---------- tiện ích HTTP (stdlib) ----------
def http(method: str, url: str, headers: dict | None = None, body: bytes | None = None, timeout: float = 120) -> tuple[int, bytes]:
    req = request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urlerror.HTTPError as e:
        payload = e.read()[:1500].decode("utf8", "replace")
        retry = e.code in (429, 500, 502, 503, 504)
        raise ProviderError(f"http_{e.code}", f"{method} {url.split('?')[0]} -> {e.code}: {payload}", retryable=retry)
    except (socket.timeout, TimeoutError):
        raise ProviderError("timeout", f"{method} {url.split('?')[0]} quá {timeout}s", uncertain_billing=True)
    except urlerror.URLError as e:
        if isinstance(e.reason, (socket.timeout, TimeoutError)):
            raise ProviderError("timeout", f"{method} {url.split('?')[0]} quá {timeout}s", uncertain_billing=True)
        raise ProviderError("unavailable", f"không kết nối được {url.split('?')[0]}: {e.reason}", retryable=True)


def http_json(method: str, url: str, headers: dict | None = None, payload: dict | None = None, timeout: float = 120) -> dict:
    h = {"Content-Type": "application/json", "Accept": "application/json", **(headers or {})}
    _, raw = http(method, url, h, json.dumps(payload).encode() if payload is not None else None, timeout)
    try:
        return json.loads(raw or b"{}")
    except ValueError:
        raise ProviderError("bad_response", f"phản hồi không phải JSON: {raw[:200]!r}")


def multipart(fields: dict[str, str], files: list[tuple[str, str, bytes, str]]) -> tuple[bytes, str]:
    """files: [(field_name, filename, data, content_type)]"""
    import uuid
    b = uuid.uuid4().hex
    out = []
    for k, v in fields.items():
        out.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    for name, fn, data, ct in files:
        out.append(f'--{b}\r\nContent-Disposition: form-data; name="{name}"; filename="{fn}"\r\nContent-Type: {ct}\r\n\r\n'.encode() + data + b"\r\n")
    out.append(f"--{b}--\r\n".encode())
    return b"".join(out), f"multipart/form-data; boundary={b}"


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def png_info(data: bytes) -> tuple[bytes, int, int]:
    """Chuẩn hoá mọi định dạng trả về thành PNG (giữ pixel), trả (png_bytes, w, h). Lỗi => ProviderError bad_response."""
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        w, h = im.size
        if im.format == "PNG":
            return data, w, h
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "PNG")
        return buf.getvalue(), w, h
    except Exception as e:  # noqa: BLE001
        raise ProviderError("bad_response", f"không đọc được ảnh trả về: {e}")


def with_retries(fn, max_retries: int, base_delay: float = 1.5):
    """Gọi lại khi lỗi `retryable` (chưa được xử lý phía nhà cung cấp). Timeout sau khi gửi KHÔNG được gọi lại (rủi ro tính phí trùng)."""
    last = None
    for i in range(max_retries + 1):
        try:
            return fn()
        except ProviderError as e:
            last = e
            if not e.retryable or i == max_retries:
                raise
            time.sleep(base_delay * (2 ** i))
    raise last  # pragma: no cover
