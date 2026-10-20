"""Image providers. `sdxl` is the real Diffusers pipeline; `mock` is for CI only and labels its output."""
import io
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from app.config import get_settings

ASPECT_SIZES = {"9:16": (768, 1344), "1:1": (1024, 1024), "16:9": (1344, 768)}  # SDXL-friendly buckets


@dataclass
class GenRequest:
    prompt: str
    negative_prompt: str | None
    width: int
    height: int
    steps: int
    guidance_scale: float
    seed: int
    num_images: int
    lora_path: Path | None = None
    lora_scale: float = 1.0


@dataclass
class GenImage:
    data: bytes
    seed: int
    meta: dict = field(default_factory=dict)


class Cancelled(Exception):
    pass


class ProviderUnavailable(Exception):
    pass


class SDXLProvider:
    name = "sdxl"

    def __init__(self) -> None:
        self._pipe = None

    def _load(self):
        try:
            import torch
            from diffusers import StableDiffusionXLPipeline
        except ImportError as e:  # API image has no torch
            raise ProviderUnavailable("torch/diffusers chưa cài (cần image worker GPU).") from e
        if not torch.cuda.is_available():
            raise ProviderUnavailable("Không có CUDA; worker GPU chưa sẵn sàng.")
        s = get_settings()
        cache = str(Path(s.data_dir) / "hf-cache")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            s.sdxl_model_id, revision=s.sdxl_revision, torch_dtype=torch.float16, variant="fp16",
            use_safetensors=True, cache_dir=cache, token=s.hf_token,
        )
        if s.low_vram_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        return pipe

    @property
    def pipe(self):
        if self._pipe is None:  # loaded once per worker process, never per request
            self._pipe = self._load()
        return self._pipe

    def generate(self, req: GenRequest, on_step: Callable[[int, int], None] | None = None) -> list[GenImage]:
        import torch

        pipe = self.pipe
        used_lora = False
        try:
            if req.lora_path is not None:
                pipe.load_lora_weights(str(req.lora_path.parent), weight_name=req.lora_path.name, adapter_name="char")
                pipe.set_adapters(["char"], adapter_weights=[req.lora_scale])
                used_lora = True
            out: list[GenImage] = []
            for i in range(req.num_images):
                seed = req.seed + i
                gen = torch.Generator("cuda").manual_seed(seed)

                def cb(_pipe, step, _t, kw, _i=i):
                    if on_step:
                        on_step(_i * req.steps + step + 1, req.num_images * req.steps)  # raises Cancelled
                    return kw

                result = pipe(
                    prompt=req.prompt, negative_prompt=req.negative_prompt, width=req.width, height=req.height,
                    num_inference_steps=req.steps, guidance_scale=req.guidance_scale, generator=gen,
                    callback_on_step_end=cb,
                )
                buf = io.BytesIO()
                result.images[0].save(buf, "PNG")
                out.append(GenImage(buf.getvalue(), seed, {"scheduler": type(pipe.scheduler).__name__, "dtype": "fp16"}))
            return out
        finally:
            if used_lora:  # no adapter may leak into the next job
                pipe.unload_lora_weights()
            torch.cuda.empty_cache()


class MockProvider:
    """Deterministic placeholder images, explicitly stamped MOCK. Never use for real work."""
    name = "mock"

    def generate(self, req: GenRequest, on_step: Callable[[int, int], None] | None = None) -> list[GenImage]:
        from PIL import Image, ImageDraw

        out = []
        for i in range(req.num_images):
            seed = req.seed + i
            img = Image.new("RGB", (req.width, req.height), ((seed * 37) % 256, (seed * 91) % 256, 120))
            ImageDraw.Draw(img).text((16, 16), f"MOCK PROVIDER seed={seed}", fill="white")
            if on_step:
                on_step((i + 1) * req.steps, req.num_images * req.steps)
            buf = io.BytesIO()
            img.save(buf, "PNG")
            out.append(GenImage(buf.getvalue(), seed, {"mock": True}))
        return out


_provider = None


def get_provider():
    global _provider
    if _provider is None:
        _provider = MockProvider() if get_settings().image_provider == "mock" else SDXLProvider()
    return _provider
