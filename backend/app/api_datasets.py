import hashlib
import json
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.models import (
    Asset, AssetBlob, Character, Dataset, DatasetItem, DatasetVersion, DatasetVersionItem,
)
from app.services.imaging import ImportError_, hamming, inspect_image, read_zip

router = APIRouter(prefix="/api/v1")

Group = Literal["identity", "body", "expression", "pose", "outfit", "appearance", "style", "scene"]
Status = Literal["pending", "approved", "rejected"]
Split = Literal["train", "val", "test"]


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    group: Group
    character_id: str | None = None
    source: str | None = None
    license: str | None = None


class ItemPatch(BaseModel):
    caption: str | None = Field(None, max_length=4000)
    status: Status | None = None
    split: Split | None = None
    synthetic: bool | None = None


class BulkPatch(BaseModel):
    item_ids: list[str] = Field(min_length=1, max_length=1000)
    status: Status | None = None
    split: Split | None = None
    caption_prepend: str | None = Field(None, max_length=200)  # vd trigger token
    caption_append: str | None = Field(None, max_length=200)


class AssetCaption(BaseModel):
    asset_id: str
    caption: str = Field("", max_length=4000)


class FromAssets(BaseModel):
    items: list[AssetCaption] = Field(min_length=1, max_length=200)


class VersionCreate(BaseModel):
    harmonize_splits: bool = False


def _ds(db: Session, dataset_id: str) -> Dataset:
    d = db.get(Dataset, dataset_id)
    if not d:
        raise HTTPException(404, "dataset không tồn tại")
    return d


def _item_dict(i: DatasetItem, a: Asset | None = None) -> dict:
    return {"id": i.id, "asset_id": i.asset_id, "filename": i.filename, "caption": i.caption,
            "status": i.status, "split": i.split, "synthetic": i.synthetic, "flags": i.flags,
            "width": a.width if a else None, "height": a.height if a else None}


def _dataset_dict(db: Session, d: Dataset) -> dict:
    counts = dict(db.execute(select(DatasetItem.status, func.count()).where(DatasetItem.dataset_id == d.id)
                             .group_by(DatasetItem.status)).all())
    last = db.scalar(select(func.max(DatasetVersion.version)).where(DatasetVersion.dataset_id == d.id)) or 0
    return {"id": d.id, "name": d.name, "group": d.group, "character_id": d.character_id,
            "source": d.source, "license": d.license, "counts": counts, "latest_version": last}


@router.post("/datasets", status_code=201)
def create_dataset(body: DatasetCreate, db: Session = Depends(get_db)):
    if body.character_id and not db.get(Character, body.character_id):
        raise HTTPException(422, "character_id không tồn tại")
    d = Dataset(**body.model_dump())
    db.add(d)
    db.commit()
    return _dataset_dict(db, d)


@router.get("/datasets")
def list_datasets(db: Session = Depends(get_db)):
    return [_dataset_dict(db, d) for d in db.scalars(select(Dataset).order_by(Dataset.created_at.desc()))]


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    return _dataset_dict(db, _ds(db, dataset_id))


@router.post("/datasets/{dataset_id}/uploads")
async def upload(dataset_id: str, files: list[UploadFile] = File(...), db: Session = Depends(get_db)):
    """Nhận ảnh rời và/hoặc ZIP (ảnh + .txt cùng tên làm caption). Trả báo cáo import."""
    ds = _ds(db, dataset_id)
    report = {"imported": 0, "duplicates": [], "near_duplicates": [], "low_resolution": [], "rejected": []}

    entries: list[tuple[str, bytes, str | None]] = []
    for f in files:
        data = await f.read()
        name = f.filename or "upload"
        if name.lower().endswith(".zip"):
            try:
                entries.extend(read_zip(data))
            except ImportError_ as e:
                report["rejected"].append({"file": name, "reason": str(e)})
        else:
            entries.append((name, data, None))

    existing = list(db.scalars(select(DatasetItem).where(DatasetItem.dataset_id == ds.id)))
    seen_sha = {a.sha256 for a in db.scalars(select(Asset).where(Asset.id.in_([i.asset_id for i in existing])))} if existing else set()
    hashes = [(i.id, i.phash) for i in existing if i.phash]

    for name, data, caption in entries:
        try:
            info = inspect_image(data)
        except ImportError_ as e:
            report["rejected"].append({"file": name, "reason": str(e)})
            continue
        sha = hashlib.sha256(data).hexdigest()
        if sha in seen_sha:
            report["duplicates"].append(name)
            continue
        seen_sha.add(sha)

        asset = db.scalar(select(Asset).where(Asset.sha256 == sha, Asset.storage_backend == settings.asset_storage_backend))
        if not asset:
            asset = Asset(kind="dataset", mime_type=info.mime, width=info.width, height=info.height,
                          byte_size=len(data), sha256=sha, storage_backend=settings.asset_storage_backend,
                          provenance={"source": "upload", "dataset_id": ds.id, "filename": name})
            db.add(asset)
            db.flush()
            db.add(AssetBlob(asset_id=asset.id, data=data))

        flags: list[str] = []
        status = "pending"
        if min(info.width, info.height) < settings.min_image_side:
            flags.append("low_resolution")
            status = "rejected"  # tự loại; người dùng có thể duyệt lại thủ công
            report["low_resolution"].append(name)
        item = DatasetItem(dataset_id=ds.id, asset_id=asset.id, caption=caption or "", status=status,
                           filename=name, phash=info.phash, flags=flags)
        db.add(item)
        db.flush()
        for other_id, ph in hashes:
            if hamming(ph, info.phash) <= settings.near_dup_hamming:
                item.flags = [*item.flags, f"near_duplicate:{other_id}"]
                report["near_duplicates"].append({"file": name, "similar_to": other_id})
                break
        hashes.append((item.id, info.phash))
        report["imported"] += 1
    db.commit()
    return report


@router.post("/datasets/{dataset_id}/items/from-assets")
def add_from_assets(dataset_id: str, body: FromAssets, db: Session = Depends(get_db)):
    """Thêm ảnh AI đã có trong kho asset vào dataset (synthetic + source_asset_id, trạng thái chờ duyệt).
    Không tự đưa mọi output vào train: người dùng chọn từng ảnh rồi vẫn phải duyệt."""
    ds = _ds(db, dataset_id)
    existing = {i.asset_id: i for i in db.scalars(select(DatasetItem).where(DatasetItem.dataset_id == ds.id))}
    hashes = [(i.id, i.phash) for i in existing.values() if i.phash]
    report = {"added": 0, "already_in_dataset": [], "missing": [], "near_duplicates": []}
    for ac in body.items:
        asset, blob = db.get(Asset, ac.asset_id), db.get(AssetBlob, ac.asset_id)
        if not asset or not blob:
            report["missing"].append(ac.asset_id)
            continue
        if ac.asset_id in existing:
            report["already_in_dataset"].append(ac.asset_id)
            continue
        info = inspect_image(blob.data)
        flags = ["low_resolution"] if min(info.width, info.height) < settings.min_image_side else []
        item = DatasetItem(dataset_id=ds.id, asset_id=asset.id, caption=ac.caption, status="rejected" if flags else "pending",
                           filename=f"ai-{asset.id[:8]}.png", phash=info.phash, flags=flags, synthetic=True,
                           source_asset_id=asset.id)
        db.add(item)
        db.flush()
        for other_id, ph in hashes:
            if hamming(ph, info.phash) <= settings.near_dup_hamming:
                item.flags = [*item.flags, f"near_duplicate:{other_id}"]
                report["near_duplicates"].append({"asset_id": asset.id, "similar_to": other_id})
                break
        hashes.append((item.id, info.phash))
        report["added"] += 1
    db.commit()
    return report


@router.get("/datasets/{dataset_id}/items")
def list_items(dataset_id: str, status: Status | None = None, limit: int = Query(60, le=200),
               offset: int = 0, db: Session = Depends(get_db)):
    _ds(db, dataset_id)
    q = select(DatasetItem, Asset).join(Asset, Asset.id == DatasetItem.asset_id).where(DatasetItem.dataset_id == dataset_id)
    if status:
        q = q.where(DatasetItem.status == status)
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    rows = db.execute(q.order_by(DatasetItem.created_at, DatasetItem.id).limit(limit).offset(offset)).all()
    return {"total": total, "items": [_item_dict(i, a) for i, a in rows]}


@router.patch("/dataset-items/{item_id}")
def patch_item(item_id: str, body: ItemPatch, db: Session = Depends(get_db)):
    i = db.get(DatasetItem, item_id)
    if not i:
        raise HTTPException(404, "item không tồn tại")
    for k, v in body.model_dump(exclude_none=True).items():
        setattr(i, k, v)
    db.commit()
    return _item_dict(i)


@router.post("/datasets/{dataset_id}/items/bulk")
def bulk_patch(dataset_id: str, body: BulkPatch, db: Session = Depends(get_db)):
    _ds(db, dataset_id)
    items = list(db.scalars(select(DatasetItem).where(DatasetItem.dataset_id == dataset_id,
                                                      DatasetItem.id.in_(body.item_ids))))
    for i in items:
        if body.status:
            i.status = body.status
        if body.split:
            i.split = body.split
        if body.caption_prepend and not i.caption.startswith(body.caption_prepend):
            i.caption = f"{body.caption_prepend}, {i.caption}".rstrip(", ")
        if body.caption_append and not i.caption.endswith(body.caption_append):
            i.caption = f"{i.caption}, {body.caption_append}".lstrip(", ")
    db.commit()
    return {"updated": len(items)}


def _groups(items: list[DatasetItem]) -> list[list[DatasetItem]]:
    """Gom ảnh gần trùng (dHash) bằng union-find."""
    parent = list(range(len(items)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            if items[a].phash and items[b].phash and hamming(items[a].phash, items[b].phash) <= settings.near_dup_hamming:
                parent[find(a)] = find(b)
    out: dict[int, list[DatasetItem]] = {}
    for idx, it in enumerate(items):
        out.setdefault(find(idx), []).append(it)
    return list(out.values())


@router.post("/datasets/{dataset_id}/versions", status_code=201)
def create_version(dataset_id: str, body: VersionCreate, db: Session = Depends(get_db)):
    ds = _ds(db, dataset_id)
    items = list(db.scalars(select(DatasetItem).where(DatasetItem.dataset_id == ds.id, DatasetItem.status == "approved")
                            .order_by(DatasetItem.created_at, DatasetItem.id)))
    if not items:
        raise HTTPException(422, "chưa có ảnh nào được duyệt")

    conflicts = []
    for g in _groups(items):
        splits = {i.split for i in g}
        if len(splits) > 1:
            if body.harmonize_splits:
                target = "train" if "train" in splits else sorted(splits)[0]
                for i in g:
                    i.split = target
            else:
                conflicts.append([i.id for i in g])
    if conflicts:
        raise HTTPException(409, detail={"code": "near_duplicate_split_conflict",
                                         "message": "Ảnh gần trùng đang ở các split khác nhau; sửa split, loại bớt hoặc dùng harmonize_splits.",
                                         "groups": conflicts})

    assets = {a.id: a for a in db.scalars(select(Asset).where(Asset.id.in_([i.asset_id for i in items])))}
    rows = sorted(({"sha256": assets[i.asset_id].sha256, "caption": i.caption, "split": i.split} for i in items),
                  key=lambda r: (r["sha256"], r["split"], r["caption"]))
    manifest_hash = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    split_counts: dict[str, int] = {}
    for i in items:
        split_counts[i.split] = split_counts.get(i.split, 0) + 1
    warnings = []
    if not split_counts.get("train"):
        raise HTTPException(422, "cần ít nhất một ảnh ở split train")
    if not split_counts.get("val"):
        warnings.append("Không có tập validation: không thể chọn checkpoint độc lập với dữ liệu train.")
    if not split_counts.get("test"):
        warnings.append("Không có tập test: không thể đánh giá cuối trên ảnh chưa thấy.")
    if len(items) < 30:
        warnings.append(f"Chỉ {len(items)} ảnh được duyệt (<30): chỉ phù hợp thử nghiệm.")
    empty = sum(1 for i in items if not i.caption.strip())
    if empty:
        warnings.append(f"{empty} ảnh chưa có caption.")

    last = db.scalar(select(func.max(DatasetVersion.version)).where(DatasetVersion.dataset_id == ds.id)) or 0
    v = DatasetVersion(dataset_id=ds.id, version=last + 1, manifest_hash=manifest_hash,
                       item_count=len(items), split_counts=split_counts, warnings=warnings)
    db.add(v)
    db.flush()
    for i in items:
        db.add(DatasetVersionItem(version_id=v.id, asset_id=i.asset_id, sha256=assets[i.asset_id].sha256,
                                  caption=i.caption, split=i.split, synthetic=i.synthetic))
    db.commit()
    return _version_dict(v)


def _version_dict(v: DatasetVersion) -> dict:
    return {"id": v.id, "version": v.version, "manifest_hash": v.manifest_hash, "item_count": v.item_count,
            "split_counts": v.split_counts, "warnings": v.warnings,
            "created_at": v.created_at.isoformat() if v.created_at else None}


@router.get("/datasets/{dataset_id}/versions")
def list_versions(dataset_id: str, db: Session = Depends(get_db)):
    _ds(db, dataset_id)
    return [_version_dict(v) for v in db.scalars(select(DatasetVersion).where(DatasetVersion.dataset_id == dataset_id)
                                                 .order_by(DatasetVersion.version.desc()))]


@router.get("/dataset-versions/{version_id}/manifest")
def manifest(version_id: str, db: Session = Depends(get_db)):
    v = db.get(DatasetVersion, version_id)
    if not v:
        raise HTTPException(404, "version không tồn tại")
    rows = db.scalars(select(DatasetVersionItem).where(DatasetVersionItem.version_id == v.id)
                      .order_by(DatasetVersionItem.sha256))
    lines = [json.dumps({"asset_id": r.asset_id, "sha256": r.sha256, "caption": r.caption,
                         "split": r.split, "synthetic": r.synthetic}, ensure_ascii=False) for r in rows]
    return Response("\n".join(lines) + "\n", media_type="application/x-ndjson",
                    headers={"Content-Disposition": f'attachment; filename="dataset-v{v.version}.jsonl"',
                             "X-Manifest-Hash": v.manifest_hash})
