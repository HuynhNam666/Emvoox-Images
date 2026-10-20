import uuid

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.common import default_project, err, get_or_404
from app.config import get_settings
from app.db import get_db
from app.models import Asset, Dataset, DatasetItem, DatasetVersion, DatasetVersionItem
from app.services import datasets as svc

router = APIRouter(prefix="/api/v1", tags=["datasets"])


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    character_id: uuid.UUID | None = None
    objective: str = Field("character", pattern="^(character|style)$")
    trigger_token: str | None = Field(None, max_length=64)


class ItemPatch(BaseModel):
    caption: str | None = None
    status: str | None = Field(None, pattern="^(pending|approved|rejected)$")
    split: str | None = Field(None, pattern="^(train|val|test)$")
    tags: list[str] | None = None
    license: str | None = None
    source: str | None = None


class BulkPatch(BaseModel):
    item_ids: list[uuid.UUID]
    patch: ItemPatch
    caption_prefix: str | None = None  # e.g. trigger token prepended to every caption


def _item(i: DatasetItem) -> dict:
    return {"id": i.id, "asset_id": i.asset_id, "caption": i.caption, "tags": i.tags, "status": i.status,
            "split": i.split, "sha256": i.sha256, "source": i.source, "license": i.license,
            "synthetic": i.synthetic, "quality_flags": i.quality_flags}


@router.post("/datasets", status_code=201)
def create_dataset(body: DatasetCreate, db: Session = Depends(get_db)) -> dict:
    p = default_project(db)
    ds = Dataset(project_id=p.id, name=body.name, character_id=body.character_id, objective=body.objective,
                 trigger_token=body.trigger_token)
    db.add(ds)
    db.commit()
    return {"id": ds.id, "name": ds.name, "objective": ds.objective, "trigger_token": ds.trigger_token}


@router.get("/datasets")
def list_datasets(db: Session = Depends(get_db)) -> list[dict]:
    out = []
    for ds in db.query(Dataset).order_by(Dataset.created_at.desc()):
        out.append({
            "id": ds.id, "name": ds.name, "objective": ds.objective, "trigger_token": ds.trigger_token,
            "character_id": ds.character_id,
            "item_count": db.query(DatasetItem).filter_by(dataset_id=ds.id).count(),
            "approved_count": db.query(DatasetItem).filter_by(dataset_id=ds.id, status="approved").count(),
            "versions": [{"id": v.id, "version": v.version, "item_count": v.item_count, "warnings": v.warnings}
                         for v in db.query(DatasetVersion).filter_by(dataset_id=ds.id).order_by(DatasetVersion.version)],
        })
    return out


@router.post("/datasets/{dataset_id}/uploads")
async def upload(dataset_id: uuid.UUID, files: list[UploadFile] = File(...), source: str | None = Form(None),
                 license: str | None = Form(None), db: Session = Depends(get_db)) -> dict:
    ds = get_or_404(db, Dataset, dataset_id, "dataset")
    if not license:
        raise err(422, "license_required", "Cần khai báo nguồn/license/quyền sử dụng của dữ liệu.")
    cap = get_settings().max_zip_uncompressed_bytes
    payload: list[tuple[str, bytes]] = []
    total = 0
    for f in files:
        data = await f.read(cap + 1)
        total += len(data)
        if total > cap:
            raise err(413, "payload_too_large", "Tổng dung lượng upload vượt giới hạn.")
        payload.append((f.filename or "file", data))
    report = svc.import_files(db, ds, payload, source, license)
    db.commit()
    return {"added": report.added, "duplicates": report.duplicates, "rejected": report.rejected}


@router.get("/datasets/{dataset_id}/items")
def list_items(dataset_id: uuid.UUID, status: str | None = None, limit: int = 100, offset: int = 0,
               db: Session = Depends(get_db)) -> dict:
    get_or_404(db, Dataset, dataset_id, "dataset")
    q = db.query(DatasetItem).filter_by(dataset_id=dataset_id)
    if status:
        q = q.filter_by(status=status)
    return {"total": q.count(), "items": [_item(i) for i in q.order_by(DatasetItem.created_at).limit(min(limit, 500)).offset(offset)]}


@router.get("/dataset-items/{item_id}")
def get_item(item_id: uuid.UUID, db: Session = Depends(get_db)) -> dict:
    return _item(get_or_404(db, DatasetItem, item_id, "mẫu dữ liệu"))


@router.patch("/dataset-items/{item_id}")
def patch_item(item_id: uuid.UUID, body: ItemPatch, db: Session = Depends(get_db)) -> dict:
    item = get_or_404(db, DatasetItem, item_id, "mẫu dữ liệu")
    for k, v in body.model_dump(exclude_unset=True).items():
        setattr(item, k, v)
    db.commit()
    return _item(item)


@router.post("/datasets/{dataset_id}/items/bulk")
def bulk_patch(dataset_id: uuid.UUID, body: BulkPatch, db: Session = Depends(get_db)) -> dict:
    get_or_404(db, Dataset, dataset_id, "dataset")
    items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id, DatasetItem.id.in_(body.item_ids)).all()
    changes = body.patch.model_dump(exclude_unset=True)
    for it in items:
        for k, v in changes.items():
            setattr(it, k, v)
        if body.caption_prefix and not it.caption.startswith(body.caption_prefix):
            it.caption = f"{body.caption_prefix}, {it.caption}".rstrip(", ")
    db.commit()
    return {"updated": len(items)}


@router.post("/datasets/{dataset_id}/versions", status_code=201)
def make_version(dataset_id: uuid.UUID, db: Session = Depends(get_db)) -> dict:
    ds = get_or_404(db, Dataset, dataset_id, "dataset")
    try:
        v = svc.create_version(db, ds)
    except svc.VersionError as e:
        db.rollback()
        raise err(409, "cannot_version", str(e))
    db.commit()
    return {"id": v.id, "version": v.version, "item_count": v.item_count, "manifest_sha256": v.manifest_sha256,
            "warnings": v.warnings}


@router.get("/dataset-versions/{version_id}/manifest")
def manifest(version_id: uuid.UUID, db: Session = Depends(get_db)) -> dict:
    v = get_or_404(db, DatasetVersion, version_id, "dataset version")
    rows = db.query(DatasetVersionItem).filter_by(dataset_version_id=v.id).order_by(DatasetVersionItem.sha256)
    return {"version": v.version, "manifest_sha256": v.manifest_sha256,
            "items": [{"asset_id": r.asset_id, "sha256": r.sha256, "caption": r.caption, "split": r.split} for r in rows]}
