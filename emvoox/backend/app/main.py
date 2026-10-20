from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import assets, characters, datasets, generation, system, training
from app.config import get_settings
from app.db import SessionLocal
from app.models import ModelVersion


def seed_registry() -> None:
    """Seed exactly one integrated base model. It stays 'experimental' until the GPU smoke test passes."""
    s = get_settings()
    db = SessionLocal()
    try:
        exists = db.query(ModelVersion).filter_by(kind="base", hf_id=s.sdxl_model_id, revision=s.sdxl_revision).first()
        if exists is None:
            db.add(ModelVersion(
                kind="base", name="SDXL 1.0 base", family="sdxl", hf_id=s.sdxl_model_id, revision=s.sdxl_revision,
                license="CreativeML OpenRAIL++-M (kiểm tra lại trên model card)", status="experimental",
                review_status="approved",
                capabilities={"trainable_lora": True, "negative_prompt": True, "reference_images": False,
                              "aspect_ratios": ["9:16", "1:1", "16:9"], "lora_families": ["sdxl"]},
            ))
            db.commit()
    finally:
        db.close()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        seed_registry()
    except Exception as e:  # noqa: BLE001  tables may not exist before migration
        print(f"[startup] registry seed skipped: {e}")
    yield


app = FastAPI(title="EMVOOX Character Studio", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=[o.strip() for o in get_settings().cors_origins.split(",") if o.strip()],
    allow_methods=["*"], allow_headers=["*"], expose_headers=["Content-Disposition"],
)
for r in (system.router, generation.router, assets.router, characters.router, datasets.router, training.router):
    app.include_router(r)


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": {"code": "internal_error", "message": "Lỗi máy chủ."}})
