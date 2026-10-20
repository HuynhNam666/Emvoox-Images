from functools import lru_cache

import re

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_NO_STRIP = {"postgres_password", "database_url"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("*", mode="before")
    @classmethod
    def _drop_inline_comment(cls, v, info):
        """A copied .env may carry 'value   # comment'; never let the comment become the value."""
        if isinstance(v, str) and info.field_name not in _NO_STRIP:
            v = re.split(r"\s+#", v, maxsplit=1)[0].strip()
            if v.startswith("#"):
                v = ""
            if v == "" and info.field_name in {"sdxl_revision", "hf_token"}:
                return None
        return v

    database_url: str = "postgresql+psycopg://emvoox:change-me@localhost:5432/emvoox_images"
    # In Compose the URL is built from parts so any password character is escaped correctly.
    postgres_password: str | None = None
    db_host: str = "postgres"
    db_user: str = "emvoox"
    db_name: str = "emvoox_images"
    redis_url: str = "redis://localhost:6379/0"

    # postgres (bytes in asset_blobs) is the default and the only backend implemented in v1.
    asset_storage_backend: str = "postgres"
    data_dir: str = "/data"  # persistent volume: datasets, runs, LoRA artifacts, HF cache

    # sdxl = real Diffusers pipeline; mock = labelled placeholder images for CI only.
    image_provider: str = "sdxl"
    sdxl_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    sdxl_revision: str | None = None  # pin a commit hash in .env for production
    hf_token: str | None = None
    low_vram_offload: bool = True  # enable_model_cpu_offload; needed on ~8 GB cards

    trainer_script: str = "/opt/trainer/train_dreambooth_lora_sdxl.py"
    trainer_revision: str = "diffusers v0.30.3"

    cors_origins: str = "http://localhost:5173,http://localhost:8080"

    # upload limits
    max_upload_bytes: int = 25 * 1024 * 1024
    max_image_pixels: int = 40_000_000
    min_image_side: int = 512
    max_zip_files: int = 3000
    max_zip_uncompressed_bytes: int = 1024 * 1024 * 1024
    near_duplicate_hamming: int = 6

    gpu_lock_ttl_seconds: int = 90
    job_stale_seconds: int = 300


def resolved_database_url() -> str:
    from sqlalchemy.engine import URL

    s = get_settings()
    if s.postgres_password is None:
        return s.database_url
    return URL.create("postgresql+psycopg", username=s.db_user, password=s.postgres_password,
                      host=s.db_host, port=5432, database=s.db_name).render_as_string(hide_password=False)


@lru_cache
def get_settings() -> Settings:
    return Settings()
