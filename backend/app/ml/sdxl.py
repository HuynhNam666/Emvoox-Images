"""Provider SDXL (Diffusers). Chỉ import torch/diffusers khi worker GPU thật sự chạy."""
import io
import logging
import threading
import time
from dataclasses import dataclass, field

from app.core.config import settings

# Kích thước gốc SDXL (bội số 64, ~1MP). Xuất 1080x1920 là bước resize/crop riêng sau này.
ASPECT_SIZES = {"9:16": (768, 1344), "1:1": (1024, 1024), "16:9": (1344, 768), "1:1-small": (768, 768)}

VAE_FP16_FIX = "madebyollin/sdxl-vae-fp16-fix"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_WEIGHT = "ip-adapter-plus-face_sdxl_vit-h.safetensors"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ENCODER_SUBFOLDER = "models/image_encoder"  # CLIP ViT-H/14 (khớp biến thể vit-h)
MAX_REFERENCES = 4


class ProviderUnavailable(RuntimeError):
    pass


@dataclass
class GenParams:
    prompt: str
    negative_prompt: str = ""
    aspect_ratio: str = "9:16"
    steps: int = 28
    guidance_scale: float = 6.0
    seed: int = 42
    lora_path: str | None = None
    lora_scale: float = 1.0
    ip_images: list[bytes] = field(default_factory=list)  # ảnh tham chiếu (bytes); rỗng = không dùng IP-Adapter
    ip_scale: float = 0.6


log = logging.getLogger("sdxl")
_lock = threading.Lock()
_pipe = None
_ip_embed_cache: dict[tuple, tuple] = {}  # khóa theo sha256 các ảnh tham chiếu -> (pos, uncond) trên CPU


def _load():
    global _pipe
    with _lock:
        if _pipe is not None:
            return _pipe
        try:
            import torch
            from diffusers import StableDiffusionXLPipeline
        except ImportError as e:
            raise ProviderUnavailable("Thiếu torch/diffusers trong worker GPU") from e
        if not torch.cuda.is_available():
            raise ProviderUnavailable("Không có CUDA GPU")
        # VAE gốc của SDXL bị diffusers ép lên fp32 khi giải mã (gấp đôi bộ nhớ → OOM ở 1024² trên 8GB).
        # VAE fp16-fix chạy thẳng fp16, cùng kiến trúc, kết quả gần như giống hệt.
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(VAE_FP16_FIX, torch_dtype=torch.float16, cache_dir=settings.model_cache_dir)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            settings.sdxl_model_id,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            cache_dir=settings.model_cache_dir,
        )
        if settings.sdxl_scheduler == "dpmpp_2m_karras":   # báo cáo mục 3.2: DPM++ 2M Karras
            from diffusers import DPMSolverMultistepScheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
        if settings.sdxl_cpu_offload:  # GPU ~8GB: chậm hơn nhưng tránh OOM
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        pipe.vae.enable_tiling()
        if not settings.sdxl_cpu_offload and _low_vram():
            # GPU nhỏ (≤ ~8–11GB): giữ 2 text encoder (~1,6GB) trên CPU, chỉ đưa lên GPU lúc mã hoá prompt.
            pipe.text_encoder.to("cpu")
            pipe.text_encoder_2.to("cpu")
            torch.cuda.empty_cache()
        _pipe = pipe
        return pipe


def _low_vram() -> bool:
    import torch
    return torch.cuda.get_device_properties(0).total_memory < 12 * 2**30


def _encoders_to(pipe, device: str) -> None:
    """Đưa 2 text encoder (~1.6GB) qua lại CPU/GPU: giải phóng VRAM cho IP-Adapter trên GPU 8GB."""
    import torch
    if settings.sdxl_cpu_offload:
        return
    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)
    if device == "cpu":
        torch.cuda.empty_cache()


def _reference_embeds(images: list[bytes]):
    """Embedding tham chiếu cho IP-Adapter plus-face: hidden_states[-2] của CLIP ViT-H, trung bình các ảnh.
    Encoder nạp tạm trên CPU rồi giải phóng (không giữ ~2.4GB RAM)."""
    import gc
    import hashlib

    import torch
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    key = tuple(hashlib.sha256(b).hexdigest() for b in images)
    if key in _ip_embed_cache:
        return _ip_embed_cache[key]
    enc = CLIPVisionModelWithProjection.from_pretrained(
        IP_ADAPTER_REPO, subfolder=IP_ENCODER_SUBFOLDER, cache_dir=settings.model_cache_dir,
        torch_dtype=torch.float16).eval()  # fp16 trên CPU: nửa RAM (~1.3GB) so với fp32
    proc = CLIPImageProcessor()  # mặc định CLIP: resize/crop 224, mean/std OpenAI
    try:
        with torch.no_grad():
            pix = proc([Image.open(io.BytesIO(b)).convert("RGB") for b in images], return_tensors="pt").pixel_values.to(torch.float16)
            pos = enc(pix, output_hidden_states=True).hidden_states[-2].mean(dim=0, keepdim=True)  # (1, 257, 1280)
            unc = enc(torch.zeros_like(pix[:1]), output_hidden_states=True).hidden_states[-2]
    finally:
        del enc
        gc.collect()
    _ip_embed_cache.clear()  # chỉ giữ bộ gần nhất (nhỏ, nhưng tránh phình)
    _ip_embed_cache[key] = (pos, unc)
    return pos, unc


def generate_images(p: GenParams, num_images: int = 1) -> list[tuple[int, bytes, int, int]]:
    """Trả [(seed, png_bytes, w, h)]. Mỗi ảnh seed = p.seed + i để truy vết."""
    if p.aspect_ratio not in ASPECT_SIZES:
        raise ValueError(f"aspect_ratio không hỗ trợ: {p.aspect_ratio}")
    if len(p.ip_images) > MAX_REFERENCES:
        raise ValueError(f"tối đa {MAX_REFERENCES} ảnh tham chiếu")
    w, h = ASPECT_SIZES[p.aspect_ratio]
    pipe = _load()  # ném ProviderUnavailable nếu thiếu torch/CUDA
    import torch

    use_ip = bool(p.ip_images)
    t0 = time.perf_counter()
    def tick(label):
        free, total = torch.cuda.mem_get_info()
        log.warning("[sdxl] %-22s %.1fs  vram used=%.2fGB reserved=%.2fGB", label, time.perf_counter() - t0,
                    (total - free) / 2**30, torch.cuda.memory_reserved() / 2**30)
    out = []
    ip_loaded = False
    lora_loaded = False
    try:
        if p.lora_path:  # adapter chỉ tồn tại trong phạm vi một job; luôn gỡ ở finally
            pipe.load_lora_weights(p.lora_path, adapter_name="job_lora")
            pipe.set_adapters(["job_lora"], adapter_weights=[p.lora_scale])
            lora_loaded = True
        # Luôn mã hoá prompt trước; ở GPU nhỏ encoder được đưa lên GPU tạm thời rồi trả về CPU (tiết kiệm ~1,6GB VRAM).
        low = _low_vram()
        if low:
            _encoders_to(pipe, "cuda")
        pe, npe, ppe, nppe = pipe.encode_prompt(
            prompt=p.prompt, negative_prompt=p.negative_prompt or None, device="cuda",
            num_images_per_prompt=1, do_classifier_free_guidance=True)
        if low:
            _encoders_to(pipe, "cpu")
        tick("encode_prompt")
        kwargs = {"prompt_embeds": pe, "negative_prompt_embeds": npe, "pooled_prompt_embeds": ppe,
                  "negative_pooled_prompt_embeds": nppe}
        if use_ip:
            pos, unc = _reference_embeds(p.ip_images)
            tick("ref embeds")
            pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder=IP_ADAPTER_SUBFOLDER, weight_name=IP_ADAPTER_WEIGHT,
                                 image_encoder_folder=None, cache_dir=settings.model_cache_dir)
            ip_loaded = True
            tick("load_ip_adapter")
            pipe.set_ip_adapter_scale(p.ip_scale)
            # định dạng pipeline đòi hỏi khi CFG bật: cat([negative, positive]) dạng (2, 1, seq, dim)
            kwargs["ip_adapter_image_embeds"] = [torch.cat([unc[None], pos[None]], dim=0).to("cuda", torch.float16)]
        for i in range(num_images):
            seed = p.seed + i
            gen = torch.Generator("cuda").manual_seed(seed)
            img = pipe(
                width=w, height=h,
                num_inference_steps=p.steps,
                guidance_scale=p.guidance_scale,
                generator=gen,
                **kwargs,
            ).images[0]
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            out.append((seed, buf.getvalue(), w, h))
            tick(f"image {i + 1}")
    finally:
        if ip_loaded:
            pipe.unload_ip_adapter()
        if lora_loaded:
            pipe.unload_lora_weights()
        torch.cuda.empty_cache()
    return out


def gpu_info() -> dict:
    """Thông tin GPU thật của worker (dùng cho preflight)."""
    try:
        import torch
    except ImportError:
        return {"cuda": False, "reason": "worker thiếu torch"}
    if not torch.cuda.is_available():
        return {"cuda": False, "reason": "không có CUDA GPU"}
    free, total = torch.cuda.mem_get_info()
    return {"cuda": True, "name": torch.cuda.get_device_name(0),
            "total_gb": round(total / 2**30, 2), "free_gb": round(free / 2**30, 2)}


def release_pipeline() -> None:
    """Giải phóng VRAM (trước khi train / sau OOM). KHÔNG bao giờ ném lỗi: sau lỗi CUDA nặng, context có thể đã hỏng."""
    global _pipe
    with _lock:
        _pipe = None
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
