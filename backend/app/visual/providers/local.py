"""Tuyến tự chạy (diffusers) trong worker GPU: SDXL, FLUX.2 Klein, Qwen-Image(-Edit), Z-Image.

Một GPU, một pipeline nằm trong VRAM mỗi lúc. Chuyển model => nhả model cũ. Chiến lược nạp theo VRAM thật:
đủ chỗ (min_vram_gb + 2) thì .to("cuda"), không thì enable_model_cpu_offload() (chậm hơn nhưng chạy được; ghi vào metrics).
Chỉ import torch/diffusers khi gọi (API không có GPU vẫn import được module này).
"""
import gc
import io
import logging
import threading
import time

from app.core.config import settings
from app.visual.providers.base import GenResult, Hooks, ImageOut, ImageRequest, Provider, ProviderError, Ref

log = logging.getLogger("visual.local")
_lock = threading.Lock()
_resident: dict = {"key": None, "pipe": None, "offload": False}

# (rộng, cao) gốc theo họ model; bội số 16 (FLUX/Z-Image) hoặc 64 (SDXL). Xuất 1080×1920 là bước resize riêng.
SIZES = {
    "sdxl": {"9:16": (768, 1344), "16:9": (1344, 768), "1:1": (1024, 1024)},
    "flux": {"9:16": (720, 1280), "16:9": (1280, 720), "1:1": (1024, 1024)},
    "qwen": {"9:16": (928, 1664), "16:9": (1664, 928), "1:1": (1328, 1328)},
    "z-image": {"9:16": (720, 1280), "16:9": (1280, 720), "1:1": (1024, 1024)},
}

HF_REPOS = {
    "flux2-klein-4b": "black-forest-labs/FLUX.2-klein-4B",
    "flux2-klein-base-4b": "black-forest-labs/FLUX.2-klein-base-4B",
    "qwen-image": "Qwen/Qwen-Image",
    "qwen-image-edit": "Qwen/Qwen-Image-Edit-2511",
    "z-image-turbo": "Tongyi-MAI/Z-Image-Turbo",
}


def size_for(family: str, aspect: str) -> tuple[int, int]:
    return SIZES[family].get(aspect) or SIZES[family]["1:1"]


def release_all() -> None:
    """Nhả mọi pipeline (SDXL + diffusers khác). Không bao giờ ném lỗi."""
    with _lock:
        _resident.update(key=None, pipe=None, offload=False)
    try:
        from app.ml.sdxl import release_pipeline
        release_pipeline()
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _pil(r: Ref):
    from PIL import Image
    return Image.open(io.BytesIO(r.data)).convert("RGB")


def _png(img) -> bytes:
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


class DiffusersLocal(Provider):
    """FLUX.2 Klein (4B / Base), Qwen-Image, Qwen-Image-Edit, Z-Image-Turbo."""

    def __init__(self, key: str, family: str, min_vram_gb: float):
        self.key, self.family, self.min_vram_gb = key, family, min_vram_gb

    def available(self):
        try:
            import torch
        except ImportError:
            return False, "worker thiếu torch"
        if not torch.cuda.is_available():
            return False, "không có CUDA GPU"
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        if total < self.min_vram_gb * 0.5:
            return False, f"GPU {total:.0f}GB quá nhỏ cho {self.key} (cần ≥ {self.min_vram_gb}GB, hoặc ≥ {self.min_vram_gb * 0.5:.0f}GB kèm offload)"
        return True, "ok"

    def _load(self):
        import torch
        import diffusers as D
        with _lock:
            if _resident["key"] == self.key:
                return _resident["pipe"], _resident["offload"]
        release_all()                       # nhả model khác
        try:
            from app.ml import comfy_client
            comfy_client.free_memory()      # ComfyUI dùng chung GPU: nhả VRAM của nó
        except Exception:  # noqa: BLE001
            pass
        cls = {"flux2-klein-4b": "Flux2KleinPipeline", "flux2-klein-base-4b": "Flux2KleinPipeline",
               "qwen-image": "QwenImagePipeline", "qwen-image-edit": "QwenImageEditPlusPipeline",
               "z-image-turbo": "ZImagePipeline"}[self.key]
        pipe = getattr(D, cls).from_pretrained(HF_REPOS[self.key], torch_dtype=torch.bfloat16, cache_dir=settings.model_cache_dir)
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        offload = total < self.min_vram_gb + 2
        if offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        with _lock:
            _resident.update(key=self.key, pipe=pipe, offload=offload)
        return pipe, offload

    def generate(self, req: ImageRequest, hooks: Hooks) -> GenResult:
        import torch
        ok, why = self.available()
        if not ok:
            raise ProviderError("unavailable", why)
        t_load = time.perf_counter()
        try:
            pipe, offload = self._load()
        except ImportError as e:
            raise ProviderError("unavailable", f"thiếu thư viện: {e}")
        except Exception as e:  # noqa: BLE001
            raise ProviderError("error", f"không nạp được {self.key}: {e}")
        load_s = time.perf_counter() - t_load
        torch.cuda.reset_peak_memory_stats()
        refs = [_pil(r) for r in req.refs]
        outs, t0 = [], time.perf_counter()
        steps = req.steps
        try:
            for i in range(req.n):
                g = torch.Generator("cuda").manual_seed(req.seed + i)
                kw = dict(prompt=req.prompt, height=req.height, width=req.width, generator=g)
                if self.key == "flux2-klein-4b":
                    steps = req.steps or 4
                    kw.update(num_inference_steps=steps, guidance_scale=1.0)
                    if refs:
                        kw["image"] = refs
                elif self.key == "flux2-klein-base-4b":
                    steps = req.steps or 50
                    kw.update(num_inference_steps=steps, guidance_scale=req.guidance or 4.0)
                    if refs:
                        kw["image"] = refs
                elif self.key == "qwen-image":
                    steps = req.steps or 40
                    kw.update(num_inference_steps=steps, true_cfg_scale=req.guidance or 4.0, negative_prompt=req.negative_prompt or " ")
                elif self.key == "qwen-image-edit":
                    steps = req.steps or 40
                    kw.update(num_inference_steps=steps, true_cfg_scale=req.guidance or 4.0, negative_prompt=req.negative_prompt or " ")
                    if not refs:
                        raise ProviderError("error", "qwen-image-edit cần ít nhất một ảnh tham chiếu")
                    kw["image"] = refs
                    kw.pop("height"), kw.pop("width")      # Edit giữ tỉ lệ ảnh đầu vào
                elif self.key == "z-image-turbo":
                    steps = req.steps or 9
                    kw.update(num_inference_steps=steps, guidance_scale=0.0)
                img = pipe(**kw).images[0]
                outs.append(ImageOut(_png(img), req.seed + i, img.width, img.height, {"steps": steps}))
        except ProviderError:
            raise
        except torch.cuda.OutOfMemoryError as e:
            release_all()
            raise ProviderError("cuda_oom", f"hết VRAM với {self.key}: {str(e)[:200]}")
        except Exception as e:  # noqa: BLE001
            if "out of memory" in str(e).lower():
                release_all()
                raise ProviderError("cuda_oom", f"hết VRAM với {self.key}: {str(e)[:200]}")
            raise ProviderError("error", f"{self.key}: {e}")
        infer_s = time.perf_counter() - t0
        return GenResult(outs, 0.0, [], {
            "steps": steps, "batch": req.n, "load_s": round(load_s, 2), "infer_s": round(infer_s, 2),
            "torch_vram_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1), "cpu_offload": offload,
            "n_refs": len(refs)})


class SDXLLocal(Provider):
    key = "sdxl-1.0"

    def available(self):
        try:
            import torch
        except ImportError:
            return False, "worker thiếu torch"
        return (True, "ok") if torch.cuda.is_available() else (False, "không có CUDA GPU")

    def generate(self, req: ImageRequest, hooks: Hooks) -> GenResult:
        import torch
        from app.ml import sdxl
        ok, why = self.available()
        if not ok:
            raise ProviderError("unavailable", why)
        aspect = req.aspect_ratio if req.aspect_ratio in sdxl.ASPECT_SIZES else "1:1"
        refs = [r.data for r in req.refs if r.role in ("identity", "body")][: sdxl.MAX_REFERENCES]
        p = sdxl.GenParams(prompt=req.prompt, negative_prompt=req.negative_prompt, aspect_ratio=aspect, steps=req.steps or 30,
                           guidance_scale=req.guidance or 6.0, seed=req.seed, lora_path=req.lora_path, lora_scale=req.lora_scale,
                           ip_images=refs, ip_scale=req.ip_scale)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            try:
                res = sdxl.generate_images(p, req.n)
            except Exception as first:  # noqa: BLE001
                if "out of memory" not in str(first).lower():
                    raise
                sdxl.release_pipeline()
                res = sdxl.generate_images(p, req.n)    # thử lại đúng 1 lần
        except sdxl.ProviderUnavailable as e:
            raise ProviderError("unavailable", str(e))
        except Exception as e:  # noqa: BLE001
            code = "cuda_oom" if "out of memory" in str(e).lower() else "error"
            raise ProviderError(code, f"sdxl: {str(e)[:300]}")
        outs = [ImageOut(data, seed, w, h, {"steps": p.steps}) for seed, data, w, h in res]
        return GenResult(outs, 0.0, [], {
            "steps": p.steps, "batch": req.n, "infer_s": round(time.perf_counter() - t0, 2), "n_refs": len(refs),
            "torch_vram_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1), "guidance": p.guidance_scale})
