"""Bộ kết nối API ngoài: BFL (FLUX), Gemini, Seedream (BytePlus), OpenAI GPT Image.

Tất cả viết theo tài liệu công khai và CHƯA được gọi thử với khóa thật (verification=connector_unverified trong registry).
Id model, base URL và khóa đều cấu hình qua môi trường (app/core/config.py). Thiếu khóa => available()=False, không giả ảnh.
Chi phí ghi nhận = giá công khai trong registry (cộng usage nếu phản hồi có); phải đối chiếu hóa đơn nhà cung cấp.
"""
import json
import time

from app.core.config import settings
from app.visual.providers.base import (
    GenResult, Hooks, ImageOut, ImageRequest, Provider, ProviderError, b64, http, http_json, multipart, png_info, with_retries,
)
from app.visual.registry import estimate_api_cost


def _dl(url: str, headers: dict | None = None) -> bytes:
    _, raw = http("GET", url, headers or {}, None, settings.api_timeout_s)
    return raw


class _Api(Provider):
    model_key = ""

    def __init__(self, registry_model):
        self.m = registry_model       # ProviderModel (để lấy pricing)

    def _cost(self, req: ImageRequest, n: int, k: str = "1K") -> float:
        return estimate_api_cost(self.m, req.width, req.height, n, len(req.refs), k)


# ------------------------------------------------------------------ BFL
class BFLProvider(_Api):
    """FLUX.2 (Pro / Klein 4B API) và 'FLUX 3 Image' khi có endpoint. Bất đồng bộ: nhận polling_url rồi thăm dò."""
    ENDPOINTS = {"flux2-pro": "flux-2-pro", "flux2-klein-4b-api": "flux-2-klein-4b"}

    def __init__(self, registry_model, key: str):
        super().__init__(registry_model)
        self.key = key

    def _endpoint(self) -> str:
        if self.key == "flux3-image":
            return settings.bfl_flux3_endpoint
        return self.ENDPOINTS[self.key]

    def available(self):
        if not settings.bfl_api_key:
            return False, "thiếu BFL_API_KEY"
        if not self._endpoint():
            return False, "chưa cấu hình endpoint (BFL_FLUX3_ENDPOINT) — tên/endpoint chưa xác minh"
        return True, "ok"

    def _hdr(self):
        return {"x-key": settings.bfl_api_key}

    def _poll(self, polling_url: str) -> tuple[bytes, dict]:
        deadline = time.monotonic() + settings.api_timeout_s
        while time.monotonic() < deadline:
            d = http_json("GET", polling_url, self._hdr(), None, 30)
            st = d.get("status")
            if st == "Ready":
                url = (d.get("result") or {}).get("sample")
                if not url:
                    raise ProviderError("bad_response", f"Ready nhưng không có result.sample: {json.dumps(d)[:300]}")
                return _dl(url), d
            if st in ("Error", "Failed", "Task not found"):
                raise ProviderError("error", f"BFL trả {st}: {json.dumps(d)[:300]}")
            if st and "Moderated" in st:
                raise ProviderError("moderated", f"BFL từ chối nội dung: {st}")
            time.sleep(settings.api_poll_s)
        raise ProviderError("timeout", f"quá {settings.api_timeout_s}s chờ {polling_url} (job có thể vẫn chạy; lưu remote_id)",
                            uncertain_billing=True)

    def generate(self, req: ImageRequest, hooks: Hooks) -> GenResult:
        ep = self._endpoint()
        ids: list[str] = list(hooks.resume_ids)       # polling_url đã gửi từ lần trước => nối lại, KHÔNG gửi trùng
        outs: list[ImageOut] = []
        for i, url in enumerate(ids):
            raw, _ = self._poll(url)
            png, w, h = png_info(raw)
            outs.append(ImageOut(png, req.seed + i, w, h, {"remote": url}))
        for i in range(len(outs), req.n):
            body = {"prompt": req.prompt, "width": req.width, "height": req.height, "seed": req.seed + i, "output_format": "png"}
            for j, r in enumerate(req.refs[:8]):
                body["input_image" if j == 0 else f"input_image_{j + 1}"] = b64(r.data)
            sub = with_retries(lambda: http_json("POST", f"{settings.bfl_base_url.rstrip('/')}/{ep}", self._hdr(), body,
                                                 settings.api_timeout_s), settings.api_max_retries)
            url = sub.get("polling_url")
            if not url:
                raise ProviderError("bad_response", f"không có polling_url: {json.dumps(sub)[:300]}")
            ids.append(url)
            hooks.on_remote_id(ids)
            raw, _ = self._poll(url)
            png, w, h = png_info(raw)
            outs.append(ImageOut(png, req.seed + i, w, h, {"remote": url}))
        return GenResult(outs, self._cost(req, len(outs)), ids)


# ------------------------------------------------------------------ Gemini
class GeminiProvider(_Api):
    def available(self):
        return (True, "ok") if settings.gemini_api_key else (False, "thiếu GEMINI_API_KEY")

    def generate(self, req: ImageRequest, hooks: Hooks) -> GenResult:
        model = settings.gemini_image_model
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        outs = []
        for i in range(req.n):
            parts = [{"text": req.prompt}]
            for r in req.refs[:10]:
                parts.append({"inline_data": {"mime_type": "image/png", "data": b64(r.data)}})
            body = {"contents": [{"parts": parts}],
                    "generationConfig": {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": req.aspect_ratio}}}
            d = with_retries(lambda: http_json("POST", url, {"x-goog-api-key": settings.gemini_api_key}, body, settings.api_timeout_s),
                             settings.api_max_retries)
            img = None
            for c in d.get("candidates", []):
                for p in (c.get("content") or {}).get("parts", []):
                    blob = p.get("inlineData") or p.get("inline_data")
                    if blob and blob.get("data"):
                        import base64
                        img = base64.b64decode(blob["data"])
            if img is None:
                fb = d.get("promptFeedback") or d.get("candidates", [{}])[0].get("finishReason")
                raise ProviderError("moderated" if fb else "bad_response", f"Gemini không trả ảnh: {json.dumps(fb)[:300]}")
            png, w, h = png_info(img)
            outs.append(ImageOut(png, None, w, h, {"model": model, "synthid": "nhà cung cấp tự nhúng (nếu hỗ trợ)"}))
        return GenResult(outs, self._cost(req, len(outs), "1K"), [], {"model_id": model})


# ------------------------------------------------------------------ Seedream
class SeedreamProvider(_Api):
    SIZES = {"9:16": "1440x2560", "16:9": "2560x1440", "1:1": "2048x2048"}

    def available(self):
        return (True, "ok") if settings.seedream_api_key else (False, "thiếu SEEDREAM_API_KEY")

    def generate(self, req: ImageRequest, hooks: Hooks) -> GenResult:
        outs = []
        for i in range(req.n):
            body = {"model": settings.seedream_model, "prompt": req.prompt, "size": self.SIZES.get(req.aspect_ratio, "2K"),
                    "response_format": "b64_json", "watermark": False, "seed": req.seed + i}
            if req.refs:
                body["image"] = [f"data:image/png;base64,{b64(r.data)}" for r in req.refs[:10]]
            d = with_retries(lambda: http_json("POST", f"{settings.seedream_base_url.rstrip('/')}/images/generations",
                                               {"Authorization": f"Bearer {settings.seedream_api_key}"}, body, settings.api_timeout_s),
                             settings.api_max_retries)
            item = (d.get("data") or [{}])[0]
            if item.get("b64_json"):
                import base64
                raw = base64.b64decode(item["b64_json"])
            elif item.get("url"):
                raw = _dl(item["url"])
            else:
                raise ProviderError("bad_response", f"Seedream không trả ảnh: {json.dumps(d)[:300]}")
            png, w, h = png_info(raw)
            outs.append(ImageOut(png, req.seed + i, w, h, {"model": settings.seedream_model}))
        return GenResult(outs, self._cost(req, len(outs)), [], {"model_id": settings.seedream_model})


# ------------------------------------------------------------------ OpenAI
class OpenAIImageProvider(_Api):
    def available(self):
        return (True, "ok") if settings.openai_api_key else (False, "thiếu OPENAI_API_KEY")

    @staticmethod
    def _size(req: ImageRequest) -> str:
        return {"9:16": "1024x1536", "16:9": "1536x1024"}.get(req.aspect_ratio, "1024x1024")

    def generate(self, req: ImageRequest, hooks: Hooks) -> GenResult:
        import base64
        hdr = {"Authorization": f"Bearer {settings.openai_api_key}"}
        base = settings.openai_base_url.rstrip("/")
        outs, extra_cost = [], 0.0
        for i in range(req.n):
            if req.refs:
                body, ctype = multipart(
                    {"model": settings.openai_image_model, "prompt": req.prompt, "size": self._size(req), "quality": "medium"},
                    [("image[]", f"ref{j}.png", r.data, "image/png") for j, r in enumerate(req.refs[:10])])
                _, raw = with_retries(lambda: http("POST", f"{base}/images/edits", {**hdr, "Content-Type": ctype}, body,
                                                   settings.api_timeout_s), settings.api_max_retries)
                d = json.loads(raw)
            else:
                d = with_retries(lambda: http_json("POST", f"{base}/images/generations", hdr,
                                                   {"model": settings.openai_image_model, "prompt": req.prompt, "size": self._size(req),
                                                    "quality": "medium", "n": 1}, settings.api_timeout_s), settings.api_max_retries)
            item = (d.get("data") or [{}])[0]
            if not item.get("b64_json"):
                raise ProviderError("bad_response", f"OpenAI không trả b64_json: {json.dumps(d)[:300]}")
            png, w, h = png_info(base64.b64decode(item["b64_json"]))
            outs.append(ImageOut(png, None, w, h, {"model": settings.openai_image_model, "usage": d.get("usage")}))
        return GenResult(outs, self._cost(req, len(outs)) + extra_cost, [], {"model_id": settings.openai_image_model})
