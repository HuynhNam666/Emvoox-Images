"""Sửa và xóa (CRUD còn thiếu): nhân vật, trang phục, bối cảnh, dataset, ảnh dataset, job ảnh/video/benchmark, bản xuất.

Nguyên tắc (xem app/services/cleanup.py): kiểm tra tham chiếu trước, còn bị dùng => 409 kèm lý do (không xóa dở); job đang chạy không xóa được;
ảnh/video chỉ bị xóa khi không còn nơi nào dùng. Version nhân vật/dataset/bối cảnh là bất biến nên "sửa" luôn tạo version mới ở API cũ;
ở đây chỉ sửa các trường hồ sơ (tên, mã, license...).
"""
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import (
    BenchmarkRating, BenchmarkRun, Character, CharacterReference, CharacterVersion, Costume, Dataset, DatasetItem, DatasetVersion,
    DatasetVersionItem, ExportRecord, GenerationJob, GenerationOutput, ModelVersion, Scene, SceneVersion, TrainingRun, VideoAttempt,
    VideoJob, VisualAttempt, VisualCandidate, VisualJob,
)
from app.services.cleanup import purge_assets

router = APIRouter(prefix="/api/v1")
RUNNING = ("queued", "running")


def _conflict(msg: str, refs: list[str]):
    raise HTTPException(409, f"{msg}: đang được dùng bởi {', '.join(refs)}")


# ------------------------------------------------------------------ Nhân vật
class CharacterPatch(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=200)
    code: str | None = Field(None, min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_-]+$")


@router.patch("/characters/{character_id}")
def patch_character(character_id: str, body: CharacterPatch, db: Session = Depends(get_db)):
    c = db.get(Character, character_id)
    if not c:
        raise HTTPException(404, "nhân vật không tồn tại")
    if body.code and body.code != c.code:
        if db.scalar(select(Character).where(Character.code == body.code)):
            raise HTTPException(409, "mã nhân vật đã tồn tại")
        c.code = body.code
    if body.name:
        c.name = body.name
    db.commit()
    return {"id": c.id, "code": c.code, "name": c.name}


def character_refs(db: Session, c: Character) -> list[str]:
    vids = [v.id for v in db.scalars(select(CharacterVersion).where(CharacterVersion.character_id == c.id))]
    refs = []
    if db.scalar(select(Dataset.id).where(Dataset.character_id == c.id).limit(1)):
        refs.append("dataset")
    if db.scalar(select(ModelVersion.id).where(ModelVersion.character_id == c.id).limit(1)):
        refs.append("LoRA đã đăng ký")
    if vids and db.scalar(select(GenerationJob.id).where(GenerationJob.character_version_id.in_(vids)).limit(1)):
        refs.append("job tạo ảnh (SDXL)")
    if vids and db.scalar(select(VideoJob.id).where(VideoJob.character_version_id.in_(vids)).limit(1)):
        refs.append("job video")
    if any(set(j.character_version_ids or []) & set(vids) for j in db.scalars(select(VisualJob))):
        refs.append("job ảnh theo shot")
    return refs


@router.delete("/characters/{character_id}")
def delete_character(character_id: str, db: Session = Depends(get_db)):
    c = db.get(Character, character_id)
    if not c:
        raise HTTPException(404, "nhân vật không tồn tại")
    refs = character_refs(db, c)
    if refs:
        _conflict(f"không xóa được {c.code}; xóa các mục này trước", refs)
    vids = [v.id for v in db.scalars(select(CharacterVersion).where(CharacterVersion.character_id == c.id))]
    ref_assets = [r.asset_id for r in db.scalars(select(CharacterReference).where(CharacterReference.character_version_id.in_(vids)))] if vids else []
    cost_assets = [co.reference_asset_id for co in db.scalars(select(Costume).where(Costume.character_id == c.id))]
    db.execute(delete(Costume).where(Costume.character_id == c.id))
    if vids:
        db.execute(delete(CharacterReference).where(CharacterReference.character_version_id.in_(vids)))
    c.current_version_id = None
    db.flush()
    db.execute(delete(CharacterVersion).where(CharacterVersion.character_id == c.id))
    db.delete(c)
    db.flush()
    purged = purge_assets(db, ref_assets + cost_assets)
    db.commit()
    return {"deleted": c.code, "assets_purged": purged}


# ------------------------------------------------------------------ Trang phục
class CostumePatch(BaseModel):
    code: str | None = Field(None, min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_-]+$")
    description_en: str | None = Field(None, min_length=1, max_length=500)
    reference_asset_id: str | None = None


@router.patch("/costumes/{costume_id}")
def patch_costume(costume_id: str, body: CostumePatch, db: Session = Depends(get_db)):
    co = db.get(Costume, costume_id)
    if not co:
        raise HTTPException(404, "trang phục không tồn tại")
    if body.code and body.code != co.code:
        if db.scalar(select(Costume).where(Costume.character_id == co.character_id, Costume.code == body.code)):
            raise HTTPException(409, "mã trang phục đã tồn tại cho nhân vật này")
        co.code = body.code
    if body.description_en:
        co.description_en = body.description_en
    if body.reference_asset_id:
        from app.models import Asset
        if not db.get(Asset, body.reference_asset_id):
            raise HTTPException(422, "reference_asset_id không tồn tại")
        co.reference_asset_id = body.reference_asset_id
    db.commit()
    return {"id": co.id, "code": co.code}


@router.delete("/costumes/{costume_id}")
def delete_costume(costume_id: str, db: Session = Depends(get_db)):
    co = db.get(Costume, costume_id)
    if not co:
        raise HTTPException(404, "trang phục không tồn tại")
    if any(costume_id in (j.costume_ids or []) for j in db.scalars(select(VisualJob))):
        _conflict(f"không xóa được trang phục {co.code}", ["job ảnh theo shot"])
    ref = co.reference_asset_id
    db.delete(co)
    db.flush()
    purge_assets(db, [ref])
    db.commit()
    return {"deleted": co.code}


# ------------------------------------------------------------------ Bối cảnh
class ScenePatch(BaseModel):
    name: str = Field(min_length=1, max_length=200)


@router.patch("/scenes/{code}")
def patch_scene(code: str, body: ScenePatch, db: Session = Depends(get_db)):
    s = db.scalar(select(Scene).where(Scene.code == code))
    if not s:
        raise HTTPException(404, "bối cảnh không tồn tại")
    s.name = body.name
    db.commit()
    return {"code": s.code, "name": s.name}


@router.delete("/scenes/{code}")
def delete_scene(code: str, db: Session = Depends(get_db)):
    s = db.scalar(select(Scene).where(Scene.code == code))
    if not s:
        raise HTTPException(404, "bối cảnh không tồn tại")
    svs = list(db.scalars(select(SceneVersion).where(SceneVersion.scene_id == s.id)))
    ids = {v.id for v in svs}
    if any(j.scene_version_id in ids for j in db.scalars(select(VisualJob).where(VisualJob.scene_version_id.is_not(None)))):
        _conflict(f"không xóa được bối cảnh {code}", ["job ảnh theo shot"])
    assets = [a for v in svs for a in (v.reference_asset_ids or [])]
    s.current_version_id = None
    db.flush()
    db.execute(delete(SceneVersion).where(SceneVersion.scene_id == s.id))
    db.delete(s)
    db.flush()
    purge_assets(db, assets)
    db.commit()
    return {"deleted": code}


# ------------------------------------------------------------------ Dataset
class DatasetPatch(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=200)
    group: Literal["identity", "body", "expression", "pose", "outfit", "appearance", "style", "scene"] | None = None
    character_id: str | None = None
    detach_character: bool = False
    source: str | None = None
    license: str | None = None


@router.patch("/datasets/{dataset_id}")
def patch_dataset(dataset_id: str, body: DatasetPatch, db: Session = Depends(get_db)):
    d = db.get(Dataset, dataset_id)
    if not d:
        raise HTTPException(404, "dataset không tồn tại")
    has_versions = db.scalar(select(DatasetVersion.id).where(DatasetVersion.dataset_id == d.id).limit(1))
    if body.group and body.group != d.group:
        if has_versions:
            raise HTTPException(409, "dataset đã có version khóa: không đổi nhóm (tạo dataset mới)")
        d.group = body.group
    if body.character_id:
        if not db.get(Character, body.character_id):
            raise HTTPException(422, "character_id không tồn tại")
        d.character_id = body.character_id
    if body.detach_character:
        d.character_id = None
    for k in ("name", "source", "license"):
        v = getattr(body, k)
        if v is not None:
            setattr(d, k, v)
    db.commit()
    return {"id": d.id, "name": d.name, "group": d.group, "license": d.license, "character_id": d.character_id}


def dataset_used_by_training(db: Session, version_ids: list[str]) -> bool:
    vs = set(version_ids)
    for r in db.scalars(select(TrainingRun)):
        if r.dataset_version_id in vs or vs & set((r.config or {}).get("extra_dataset_version_ids", [])):
            return True
    return False


@router.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    d = db.get(Dataset, dataset_id)
    if not d:
        raise HTTPException(404, "dataset không tồn tại")
    vers = list(db.scalars(select(DatasetVersion).where(DatasetVersion.dataset_id == d.id)))
    if dataset_used_by_training(db, [v.id for v in vers]):
        _conflict(f"không xóa được dataset '{d.name}' (version bất biến phục vụ truy vết)", ["lượt huấn luyện LoRA"])
    assets = [i.asset_id for i in db.scalars(select(DatasetItem).where(DatasetItem.dataset_id == d.id))]
    vids = [v.id for v in vers]
    if vids:
        assets += [vi.asset_id for vi in db.scalars(select(DatasetVersionItem).where(DatasetVersionItem.version_id.in_(vids)))]
        db.execute(delete(DatasetVersionItem).where(DatasetVersionItem.version_id.in_(vids)))
        db.execute(delete(DatasetVersion).where(DatasetVersion.dataset_id == d.id))
    db.execute(delete(DatasetItem).where(DatasetItem.dataset_id == d.id))
    db.delete(d)
    db.flush()
    purged = purge_assets(db, assets)
    db.commit()
    return {"deleted": d.name, "versions": len(vers), "assets_purged": purged}


class ItemIds(BaseModel):
    item_ids: list[str] = Field(min_length=1, max_length=500)


def _delete_items(db: Session, items: list[DatasetItem]) -> int:
    assets = [i.asset_id for i in items]
    for i in items:
        db.delete(i)
    db.flush()
    return purge_assets(db, assets)


@router.delete("/dataset-items/{item_id}")
def delete_item(item_id: str, db: Session = Depends(get_db)):
    i = db.get(DatasetItem, item_id)
    if not i:
        raise HTTPException(404, "ảnh không tồn tại")
    purged = _delete_items(db, [i])
    db.commit()
    return {"deleted": 1, "assets_purged": purged, "note": "version đã khóa vẫn giữ nguyên ảnh này (bất biến)"}


@router.post("/datasets/{dataset_id}/items/delete")
def delete_items(dataset_id: str, body: ItemIds, db: Session = Depends(get_db)):
    items = list(db.scalars(select(DatasetItem).where(DatasetItem.dataset_id == dataset_id, DatasetItem.id.in_(body.item_ids))))
    purged = _delete_items(db, items)
    db.commit()
    return {"deleted": len(items), "assets_purged": purged}


# ------------------------------------------------------------------ Job
def _delete_visual(db: Session, j: VisualJob) -> int:
    cands = list(db.scalars(select(VisualCandidate).where(VisualCandidate.job_id == j.id)))
    cand_assets = {c.asset_id for c in cands}
    if db.scalar(select(VideoJob.id).where(VideoJob.source_keyframe_asset_id.in_(cand_assets)).limit(1)) if cand_assets else None:
        _conflict("không xóa được job ảnh", ["job video dùng ảnh này làm khung đầu (xóa video trước)"])
    ex = list(db.scalars(select(ExportRecord).where(ExportRecord.job_id == j.id)))
    assets = list(cand_assets) + [e.export_asset_id for e in ex if e.export_asset_id]
    for e in ex:
        db.delete(e)
    db.execute(delete(BenchmarkRating).where(BenchmarkRating.candidate_id.in_([c.id for c in cands]))) if cands else None
    for c in cands:
        db.delete(c)
    db.flush()
    db.execute(delete(VisualAttempt).where(VisualAttempt.job_id == j.id))
    db.delete(j)
    db.flush()
    return purge_assets(db, assets)


@router.delete("/visual/jobs/{job_id}")
def delete_visual_job(job_id: str, db: Session = Depends(get_db)):
    j = db.get(VisualJob, job_id)
    if not j:
        raise HTTPException(404, "visual job không tồn tại")
    if j.state in RUNNING:
        raise HTTPException(409, "job đang chạy/đang chờ: không xóa được")
    purged = _delete_visual(db, j)
    db.commit()
    return {"deleted": job_id, "assets_purged": purged}


@router.delete("/video/jobs/{job_id}")
def delete_video_job(job_id: str, db: Session = Depends(get_db)):
    j = db.get(VideoJob, job_id)
    if not j:
        raise HTTPException(404, "video job không tồn tại")
    if j.state in RUNNING:
        raise HTTPException(409, "job đang chạy/đang chờ: không xóa được")
    atts = list(db.scalars(select(VideoAttempt).where(VideoAttempt.job_id == j.id)))
    ex = list(db.scalars(select(ExportRecord).where(ExportRecord.job_id == j.id)))
    assets = [a.output_asset_id for a in atts if a.output_asset_id] + [e.export_asset_id for e in ex if e.export_asset_id]
    assets += [j.output_asset_id] if j.output_asset_id else []
    for e in ex:
        db.delete(e)
    for a in atts:
        db.delete(a)
    db.flush()
    src = j.source_keyframe_asset_id
    db.delete(j)
    db.flush()
    purged = purge_assets(db, assets + [src])
    db.commit()
    return {"deleted": job_id, "assets_purged": purged}


@router.delete("/generation-jobs/{job_id}")
def delete_generation_job(job_id: str, db: Session = Depends(get_db)):
    j = db.get(GenerationJob, job_id)
    if not j:
        raise HTTPException(404, "job không tồn tại")
    if j.state in RUNNING:
        raise HTTPException(409, "job đang chạy/đang chờ: không xóa được")
    outs = list(db.scalars(select(GenerationOutput).where(GenerationOutput.job_id == j.id)))
    oa = {o.asset_id for o in outs}
    if oa and db.scalar(select(VideoJob.id).where(VideoJob.source_keyframe_asset_id.in_(oa)).limit(1)):
        _conflict("không xóa được job ảnh", ["job video dùng ảnh này làm khung đầu (xóa video trước)"])
    for o in outs:
        db.delete(o)
    db.flush()
    db.delete(j)
    db.flush()
    purged = purge_assets(db, list(oa))
    db.commit()
    return {"deleted": job_id, "assets_purged": purged}


@router.delete("/benchmark/runs/{run_id}")
def delete_benchmark(run_id: str, db: Session = Depends(get_db)):
    run = db.get(BenchmarkRun, run_id)
    if not run:
        raise HTTPException(404, "lượt benchmark không tồn tại")
    jobs = list(db.scalars(select(VisualJob).where(VisualJob.benchmark_run_id == run_id)))
    if any(j.state in RUNNING for j in jobs):
        raise HTTPException(409, "còn job đang chạy/đang chờ: đợi xong rồi xóa")
    db.execute(delete(BenchmarkRating).where(BenchmarkRating.run_id == run_id))
    purged = 0
    for j in jobs:
        purged += _delete_visual(db, j)
    db.delete(run)
    db.commit()
    return {"deleted": run.name, "jobs": len(jobs), "assets_purged": purged}


@router.delete("/exports/{export_id}")
def delete_export(export_id: str, db: Session = Depends(get_db)):
    e = db.get(ExportRecord, export_id)
    if not e:
        raise HTTPException(404, "bản xuất không tồn tại")
    a = e.export_asset_id
    db.delete(e)
    db.flush()
    purge_assets(db, [a] if a else [])
    db.commit()
    return {"deleted": export_id}
