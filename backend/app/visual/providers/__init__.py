"""Nhà máy provider: từ ProviderModel trong registry => đối tượng Provider."""
from app.visual.providers.api_providers import BFLProvider, GeminiProvider, OpenAIImageProvider, SeedreamProvider
from app.visual.providers.base import Provider, ProviderError
from app.visual.providers.local import DiffusersLocal, SDXLLocal

__all__ = ["get_provider", "Provider", "ProviderError"]


def get_provider(m) -> Provider:
    """m: ProviderModel. Model route=manual không có provider tự động."""
    k = m.key
    if k == "sdxl-1.0":
        return SDXLLocal()
    if m.route == "local":
        return DiffusersLocal(k, m.family, m.min_vram_gb or 16)
    if k in ("flux2-pro", "flux2-klein-4b-api", "flux3-image"):
        return BFLProvider(m, k)
    if k == "gemini-nano-banana":
        return GeminiProvider(m)
    if k == "seedream-4-5":
        return SeedreamProvider(m)
    if k == "gpt-image":
        return OpenAIImageProvider(m)
    raise ProviderError("not_configured", f"model {k} không có provider tự động (route={m.route})")
