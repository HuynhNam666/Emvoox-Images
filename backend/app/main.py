import hashlib
import io

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.api_characters import router as characters_router, seed_sample_characters
from app.api_datasets import router as datasets_router
from app.api_jobs import router as jobs_router
from app.api_manage import router as manage_router
from app.api_reports import router as reports_router
from app.api_visual import router as visual_router
from app.auth import TokenAuthMiddleware, router as auth_router
from app.api_video import router as video_router
from app.api_training import router as training_router, seed_base_model, seed_ip_adapter
from app.core.config import settings
from app.db.migrate import sync_columns
from app.db.session import Base, SessionLocal, engine, get_db
from app.visual.registry import seed_registry
from app.visual.seed import seed_pilot_library
from app.models import Asset, AssetBlob

app = FastAPI(title="EMVOOX Character Studio")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs_router)
app.include_router(characters_router)
app.include_router(datasets_router)
app.include_router(training_router)
app.include_router(video_router)
app.include_router(visual_router)
app.include_router(reports_router)
app.include_router(manage_router)
app.include_router(auth_router)
app.add_middleware(TokenAuthMiddleware)

ALLOWED_MIME ={"image/jpeg", "image/png", "image/webp"}


@app.on_event("startup")
def _create_tables() -> None:
    # Khung ban đầu: dùng create_all. Chuyển sang Alembic trước khi triển khai thật.
    Base.metadata.create_all(engine)
    sync_columns(engine)   # thêm cột mới vào bảng cũ (chưa dùng Alembic)
    seed_sample_characters()
    seed_base_model()
    seed_ip_adapter()
    with SessionLocal() as db:
        seed_registry(db)
        seed_pilot_library(db)


@app.get("/api/v1/health/live")
def live():
    return {"status": "ok"}


@app.get("/api/v1/health/ready")
def ready(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"status": "ready"}


@app.get("/api/v1/assets/{asset_id}/thumbnail")
def asset_thumbnail(asset_id: str, db: Session = Depends(get_db)):
    blob = db.get(AssetBlob, asset_id)
    if not blob:
        raise HTTPException(404, "asset không tồn tại")
    if not blob.data[:16].startswith((b"\x89PNG", b"\xff\xd8", b"RIFF")):
        raise HTTPException(415, "asset không phải ảnh (video dùng /content)")
    img = Image.open(io.BytesIO(blob.data)).convert("RGB")
    img.thumbnail((320, 320))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return Response(buf.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.post("/api/v1/assets", status_code=201)
async def upload_asset(file: UploadFile = File(...), kind: str = "dataset", db: Session = Depends(get_db)):
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "file quá lớn")
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data))
        mime = Image.MIME.get(img.format or "")
    except Exception:
        raise HTTPException(422, "không phải ảnh hợp lệ")
    if mime not in ALLOWED_MIME:
        raise HTTPException(422, "chỉ nhận JPG/PNG/WebP")
    asset = Asset(
        kind=kind, mime_type=mime, width=img.width, height=img.height,
        byte_size=len(data), sha256=hashlib.sha256(data).hexdigest(),
        storage_backend=settings.asset_storage_backend,
    )
    db.add(asset)
    db.flush()
    db.add(AssetBlob(asset_id=asset.id, data=data))
    db.commit()
    return {"id": asset.id, "sha256": asset.sha256, "width": asset.width, "height": asset.height}


@app.get("/api/v1/assets")
def list_assets(limit: int = 50, offset: int = 0, db: Session = Depends(get_db)):
    rows = db.scalars(select(Asset).order_by(Asset.created_at.desc()).limit(limit).offset(offset))
    return [{"id": a.id, "kind": a.kind, "mime_type": a.mime_type, "sha256": a.sha256} for a in rows]


@app.get("/api/v1/assets/{asset_id}/content")
def asset_content(asset_id: str, db: Session = Depends(get_db), range: str | None = Header(None)):
    asset = db.get(Asset, asset_id)
    blob = db.get(AssetBlob, asset_id)
    if not asset or not blob:
        raise HTTPException(404, "asset không tồn tại")
    data, mime = blob.data, asset.mime_type
    if not mime.startswith("video/"):
        return Response(data, media_type=mime)
    # Video: hỗ trợ Range để trình phát tua được.
    size = len(data)
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "max-age=3600"}
    if range and range.startswith("bytes=") and "," not in range:
        a, _, b = range[6:].partition("-")
        try:
            start = size - int(b) if not a else int(a)
            end = min(int(b), size - 1) if (a and b) else size - 1
        except ValueError:
            raise HTTPException(416, "Range không hợp lệ")
        if start < 0 or start >= size or end < start:
            raise HTTPException(416, "Range ngoài phạm vi", headers={"Content-Range": f"bytes */{size}"})
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        return Response(data[start:end + 1], status_code=206, media_type=mime, headers=headers)
    return Response(data, media_type=mime, headers=headers)
