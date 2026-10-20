import uuid

from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from app.api.common import err, get_or_404
from app.db import get_db
from app.models import Asset, AssetBlob
from app.services.imaging import export_fill

router = APIRouter(prefix="/api/v1/assets", tags=["assets"])


def _meta(a: Asset) -> dict:
    return {"id": a.id, "kind": a.kind, "mime_type": a.mime_type, "width": a.width, "height": a.height,
            "byte_size": a.byte_size, "sha256": a.sha256, "storage_backend": a.storage_backend,
            "provenance": a.provenance, "created_at": a.created_at}


@router.get("")
def list_assets(kind: str | None = None, limit: int = 60, offset: int = 0, db: Session = Depends(get_db)) -> dict:
    q = db.query(Asset)  # metadata only: AssetBlob is never joined here
    if kind:
        q = q.filter(Asset.kind == kind)
    total = q.count()
    rows = q.order_by(Asset.created_at.desc()).limit(min(limit, 200)).offset(offset).all()
    return {"total": total, "items": [_meta(a) for a in rows]}


@router.get("/{asset_id}")
def asset_meta(asset_id: uuid.UUID, db: Session = Depends(get_db)) -> dict:
    return _meta(get_or_404(db, Asset, asset_id, "asset"))


def _blob(db: Session, asset_id: uuid.UUID) -> tuple[Asset, AssetBlob]:
    a = get_or_404(db, Asset, asset_id, "asset")
    b = db.get(AssetBlob, asset_id)
    if b is None:
        raise err(404, "blob_missing", "Asset không có bytes trong database.")
    return a, b


@router.get("/{asset_id}/content")
def content(asset_id: uuid.UUID, db: Session = Depends(get_db)) -> Response:
    a, b = _blob(db, asset_id)
    return Response(b.data, media_type=a.mime_type, headers={"Cache-Control": "private, max-age=86400", "ETag": a.sha256})


@router.get("/{asset_id}/thumbnail")
def thumbnail(asset_id: uuid.UUID, db: Session = Depends(get_db)) -> Response:
    a, b = _blob(db, asset_id)
    return Response(b.thumbnail or b.data, media_type="image/jpeg" if b.thumbnail else a.mime_type,
                    headers={"Cache-Control": "private, max-age=86400"})


@router.get("/{asset_id}/export")
def export(asset_id: uuid.UUID, width: int = 1080, height: int = 1920, db: Session = Depends(get_db)) -> Response:
    """Aspect-preserving resize + center crop (Lanczos). Not AI upscaling; derived on the fly."""
    if not (64 <= width <= 4096 and 64 <= height <= 4096):
        raise err(422, "bad_size", "Kích thước phải trong 64..4096.")
    _, b = _blob(db, asset_id)
    return Response(export_fill(b.data, width, height), media_type="image/png",
                    headers={"Content-Disposition": f'attachment; filename="{asset_id}_{width}x{height}.png"'})
