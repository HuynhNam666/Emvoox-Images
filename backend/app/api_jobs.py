import hashlib
import json
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import Asset, Character, CharacterReference, CharacterVersion, GenerationJob, GenerationOutput, ModelVersion
from app.workers.celery_app import celery_app

router = APIRouter(prefix="/api/v1")


class GenerationRequest(BaseModel):
    prompt_original: str = Field(min_length=1, max_length=4000)
    prompt_compiled: str | None = None
    negative_prompt: str = ""
    character_version_id: str | None = None
    aspect_ratio: Literal["9:16", "1:1", "16:9", "1:1-small"] = "9:16"
    num_images: int = Field(1, ge=1, le=4)
    seed: int = Field(42, ge=0, lt=2**31)
    steps: int = Field(28, ge=1, le=100)
    guidance_scale: float = Field(6.0, ge=0, le=20)
    lora_version_id: str | None = None
    lora_scale: float = Field(1.0, ge=0, le=2)
    evaluation: bool = False  # cho phép LoRA chưa duyệt (awaiting_review) để đánh giá
    use_references: bool = False  # IP-Adapter: dùng ảnh chuẩn của nhân vật làm tham chiếu diện mạo
    reference_asset_ids: list[str] | None = Field(None, max_length=4)  # ghi đè ảnh chuẩn (tối đa 4)
    reference_scale: float = Field(0.6, ge=0, le=1)


@router.post("/generation-jobs", status_code=202)
def create_job(
    req: GenerationRequest,
    db: Session = Depends(get_db),
    idempotency_key: str | None = Header(None),
):
    if req.character_version_id and not db.get(CharacterVersion, req.character_version_id):
        raise HTTPException(422, "character_version_id không tồn tại")
    if req.lora_version_id:
        mv = db.get(ModelVersion, req.lora_version_id)
        if not mv or mv.kind not in ("character_lora", "style_lora"):
            raise HTTPException(422, "lora_version_id không hợp lệ")
        if mv.status != "approved" and not (req.evaluation and mv.status == "awaiting_review"):
            raise HTTPException(422, f"LoRA đang ở trạng thái '{mv.status}'; chỉ LoRA đã duyệt dùng cho sản xuất "
                                     "(đặt evaluation=true để thử LoRA chờ duyệt)")
        if req.character_version_id and mv.character_id:
            cv = db.get(CharacterVersion, req.character_version_id)
            if cv.character_id != mv.character_id:
                raise HTTPException(422, "LoRA thuộc nhân vật khác; không dùng chéo giữa các nhân vật")
    if req.use_references:
        adapter = db.scalar(select(ModelVersion).where(ModelVersion.kind == "ip_adapter", ModelVersion.family == "sdxl"))
        if not adapter or adapter.status == "unavailable":
            raise HTTPException(503, "IP-Adapter chưa khả dụng trong registry")
        ids = req.reference_asset_ids
        if ids is None:
            if not req.character_version_id:
                raise HTTPException(422, "use_references cần character_version_id hoặc reference_asset_ids")
            ids = [r.asset_id for r in db.scalars(select(CharacterReference).where(
                CharacterReference.character_version_id == req.character_version_id))][:4]
        if not ids:
            raise HTTPException(422, "nhân vật chưa có ảnh chuẩn để làm tham chiếu")
        missing = [i for i in ids if not db.get(Asset, i)]
        if missing:
            raise HTTPException(422, f"ảnh tham chiếu không tồn tại: {missing[0]}")
    payload = req.model_dump_json()
    req_hash = hashlib.sha256(payload.encode()).hexdigest()
    if idempotency_key:
        old = db.scalar(select(GenerationJob).where(GenerationJob.idempotency_key == idempotency_key))
        if old:
            if old.request_hash != req_hash:
                raise HTTPException(409, "Idempotency-Key đã dùng với payload khác")
            return {"job_id": old.id, "state": old.state}
    cfg = json.loads(payload)
    job = GenerationJob(
        idempotency_key=idempotency_key, request_hash=req_hash,
        character_version_id=req.character_version_id,
        prompt_original=req.prompt_original, prompt_compiled=req.prompt_compiled, config=cfg,
    )
    db.add(job)
    db.commit()
    try:
        celery_app.send_task("app.workers.tasks.generate", args=[job.id], queue="generation")
    except Exception as e:
        job.state, job.error = "failed", f"enqueue_failed: {e}"[:500]
        db.commit()
        raise HTTPException(503, "không đưa được job vào hàng đợi")
    return {"job_id": job.id, "state": job.state}


@router.get("/generation-jobs/{job_id}")
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.get(GenerationJob, job_id)
    if not job:
        raise HTTPException(404, "job không tồn tại")
    outs = db.scalars(
        select(GenerationOutput).where(GenerationOutput.job_id == job_id).order_by(GenerationOutput.output_index)
    )
    return {
        "id": job.id, "state": job.state, "error": job.error, "config": job.config,
        "outputs": [{"asset_id": o.asset_id, "seed": o.seed, "index": o.output_index} for o in outs],
    }


@router.get("/generation-jobs")
def list_jobs(
    limit: int = Query(24, ge=1, le=100),
    offset: int = Query(0, ge=0),
    state: str | None = None,
    character_id: str | None = None,
    db: Session = Depends(get_db),
):
    """Lịch sử tạo ảnh (mới nhất trước) kèm ảnh đầu ra và tham số đã dùng. Không tải bytes ảnh."""
    q = select(GenerationJob)
    if state:
        q = q.where(GenerationJob.state == state)
    if character_id:
        q = q.join(CharacterVersion, CharacterVersion.id == GenerationJob.character_version_id).where(
            CharacterVersion.character_id == character_id)
    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    jobs = list(db.scalars(q.order_by(GenerationJob.created_at.desc(), GenerationJob.id).limit(limit).offset(offset)))
    ids = [j.id for j in jobs]
    outs: dict[str, list] = {}
    if ids:
        for o in db.scalars(select(GenerationOutput).where(GenerationOutput.job_id.in_(ids)).order_by(
                GenerationOutput.output_index)):
            outs.setdefault(o.job_id, []).append({"asset_id": o.asset_id, "seed": o.seed, "index": o.output_index})
    chars: dict[str, dict] = {}
    for cv_id in {j.character_version_id for j in jobs if j.character_version_id}:
        cv = db.get(CharacterVersion, cv_id)
        c = db.get(Character, cv.character_id) if cv else None
        if cv and c:
            chars[cv_id] = {"id": c.id, "code": c.code, "name": c.name, "version": cv.version}
    items = [{
        "id": j.id, "state": j.state, "error": j.error, "created_at": j.created_at.isoformat() if j.created_at else None,
        "prompt_original": j.prompt_original, "prompt_compiled": j.prompt_compiled, "config": j.config,
        "character": chars.get(j.character_version_id), "outputs": outs.get(j.id, []),
    } for j in jobs]
    return {"total": total, "items": items}


class JobReview(BaseModel):
    decision: Literal["approved", "rejected"]
    note: str = Field("", max_length=2000)


@router.post("/generation-jobs/{job_id}/review")
def review_job(job_id: str, req: JobReview, db: Session = Depends(get_db)):
    """Người duyệt ảnh. Chỉ job `approved` mới cho ảnh làm khung đầu video."""
    job = db.get(GenerationJob, job_id)
    if not job:
        raise HTTPException(404, "job không tồn tại")
    if job.state not in ("review_required", "approved", "rejected"):
        raise HTTPException(409, f"job đang '{job.state}'; chỉ duyệt được khi ảnh đã tạo xong")
    job.state = req.decision
    if req.note:
        job.config = {**job.config, "review_note": req.note}
    db.commit()
    return {"id": job.id, "state": job.state}
