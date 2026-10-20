import uuid

from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Asset, AssetBlob
from app.services.imaging import inspect_image, make_thumbnail


def store_asset(
    db: Session, *, project_id: uuid.UUID, data: bytes, kind: str, provenance: dict | None = None,
    enforce_min_side: bool = False,
) -> Asset:
    """Insert metadata + BYTEA bytes. Caller commits."""
    if get_settings().asset_storage_backend != "postgres":
        raise NotImplementedError("Only ASSET_STORAGE_BACKEND=postgres is implemented in v1.")
    info = inspect_image(data, enforce_min_side=enforce_min_side)
    asset = Asset(
        id=uuid.uuid4(), project_id=project_id, kind=kind, mime_type=info.mime_type, width=info.width,
        height=info.height, byte_size=len(data), sha256=info.sha256, storage_backend="postgres",
        provenance=provenance or {},
    )
    db.add(asset)
    db.flush()
    db.add(AssetBlob(asset_id=asset.id, data=data, thumbnail=make_thumbnail(data)))
    db.flush()
    return asset
