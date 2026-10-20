import uuid

from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.common import default_project, err, get_or_404
from app.db import get_db
from app.ml.providers import ASPECT_SIZES
from app.models import GenerationJob, ModelVersion, Review
from app.services.jobs import dispatch_job, request_hash, transition
from app.services.prompt_compiler import COMPILER_VERSION, DEFAULT_NEGATIVE, compile_prompt

router = APIRouter(prefix="/api/v1", tags=["generation"])


class GenParams(BaseModel):
    steps: int = Field(28, ge=1, le=100)
    guidance_scale: float = Field(6.0, ge=0, le=20)
    lora_scale: float = Field(1.0, ge=0, le=2)


class GenerationRequest(BaseModel):
    project_id: uuid.UUID | None = None
    character_version_id: uuid.UUID | None = None
    prompt_original: str = Field(min_length=1, max_length=4000)
    prompt_compiled: str | None = None
    negative_prompt: str | None = None
    model_version_id: uuid.UUID
    lora_version_id: uuid.UUID | None = None
    reference_asset_ids: list[uuid.UUID] = []
    aspect_ratio: str = "9:16"
    num_images: int = Field(1, ge=1, le=4)
    seed: int = Field(42, ge=0, le=2**31 - 1)
    parameters: GenParams = GenParams()


class CompileRequest(BaseModel):
    fields: dict = {}
    description_en: str | None = None
    trigger_token: str | None = None


def serialize(job: GenerationJob) -> dict:
    return {
        "id": job.id, "state": job.state, "purpose": job.purpose, "prompt_original": job.prompt_original,
        "prompt_compiled": job.prompt_compiled, "config": job.config, "progress_step": job.progress_step,
        "progress_total": job.progress_total, "error_code": job.error_code, "error_message": job.error_message,
        "model_version_id": job.model_version_id, "lora_version_id": job.lora_version_id,
        "character_version_id": job.character_version_id, "created_at": job.created_at,
        "outputs": [{"asset_id": o.asset_id, "output_index": o.output_index, "seed": o.seed} for o in job.outputs],
    }


@router.post("/prompt/compile")
def compile_endpoint(body: CompileRequest) -> dict:
    c = compile_prompt(body.fields, trigger_token=body.trigger_token, description_en=body.description_en)
    return {"prompt": c.prompt, "negative_prompt": c.negative_prompt, "warnings": c.warnings, "version": c.version}


@router.post("/generation-jobs", status_code=202)
def create_job(body: GenerationRequest, idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
               db: Session = Depends(get_db)) -> dict:
    project = db.get(type(default_project(db)), body.project_id) if body.project_id else default_project(db)
    if project is None:
        raise err(404, "not_found", "Không tìm thấy project.")
    if body.aspect_ratio not in ASPECT_SIZES:
        raise err(422, "bad_aspect", f"Tỉ lệ hỗ trợ: {', '.join(ASPECT_SIZES)}.")
    base = get_or_404(db, ModelVersion, body.model_version_id, "model")
    if base.kind != "base" or base.status == "unavailable":
        raise err(422, "bad_model", "Model nền không khả dụng.")
    if body.reference_asset_ids:
        raise err(422, "unsupported", "Ảnh tham chiếu (IP-Adapter) chưa được hỗ trợ ở bản này.")
    if body.lora_version_id:
        lora = get_or_404(db, ModelVersion, body.lora_version_id, "LoRA")
        if lora.parent_base_id != base.id or lora.family != base.family:
            raise err(422, "incompatible_lora", "LoRA không tương thích với model nền đã chọn.")
        if lora.review_status != "approved":
            raise err(422, "lora_not_approved", "LoRA chưa được duyệt; dùng tạo ảnh đánh giá trong Training Center.")
    w, h = ASPECT_SIZES[body.aspect_ratio]
    compiled = body.prompt_compiled or body.prompt_original
    cfg = {
        "width": w, "height": h, "steps": body.parameters.steps, "guidance_scale": body.parameters.guidance_scale,
        "seed": body.seed, "num_images": body.num_images, "lora_scale": body.parameters.lora_scale,
        "aspect_ratio": body.aspect_ratio, "compiler_version": COMPILER_VERSION,
    }
    payload = body.model_dump(mode="json")
    payload["project_id"] = str(project.id)
    rhash = request_hash(payload)
    if idempotency_key:
        existing = db.query(GenerationJob).filter_by(project_id=project.id, idempotency_key=idempotency_key).first()
        if existing:
            if existing.request_hash != rhash:
                raise err(409, "idempotency_conflict", "Idempotency-Key đã dùng với payload khác.")
            return serialize(existing)
    job = GenerationJob(
        project_id=project.id, character_version_id=body.character_version_id, model_version_id=base.id,
        lora_version_id=body.lora_version_id, idempotency_key=idempotency_key, request_hash=rhash,
        prompt_original=body.prompt_original, prompt_compiled=compiled,
        negative_prompt=body.negative_prompt if body.negative_prompt is not None else DEFAULT_NEGATIVE, config=cfg,
    )
    db.add(job)
    try:
        db.commit()  # outbox: row is durable before enqueue
    except IntegrityError:
        db.rollback()
        existing = db.query(GenerationJob).filter_by(project_id=project.id, idempotency_key=idempotency_key).first()
        if existing and existing.request_hash == rhash:
            return serialize(existing)
        raise err(409, "idempotency_conflict", "Idempotency-Key xung đột.")
    try:
        dispatch_job(db, job.id)
    except Exception:  # broker down: stays queued with enqueued_at NULL, sweeper retries
        db.rollback()
    db.refresh(job)
    return serialize(job)


@router.get("/generation-jobs")
def list_jobs(limit: int = 50, offset: int = 0, db: Session = Depends(get_db)) -> list[dict]:
    rows = (db.query(GenerationJob).filter(GenerationJob.purpose == "production")
            .order_by(GenerationJob.created_at.desc()).limit(min(limit, 200)).offset(offset).all())
    return [serialize(r) for r in rows]


@router.get("/generation-jobs/{job_id}")
def get_job(job_id: uuid.UUID, db: Session = Depends(get_db)) -> dict:
    return serialize(get_or_404(db, GenerationJob, job_id, "job"))


@router.post("/generation-jobs/{job_id}/cancel")
def cancel_job(job_id: uuid.UUID, db: Session = Depends(get_db)) -> dict:
    job = get_or_404(db, GenerationJob, job_id, "job")
    if job.state in {"review_required", "approved", "rejected", "failed", "cancelled"}:
        raise err(409, "terminal_state", f"Job đã ở trạng thái {job.state}.")
    job.cancel_requested = True
    db.commit()
    transition(db, GenerationJob, job_id, {"queued"}, "cancelled")  # running jobs notice the flag at next step
    db.refresh(job)
    return serialize(job)


class ReviewBody(BaseModel):
    decision: str = Field(pattern="^(approved|rejected)$")
    reason: str | None = None


@router.post("/generation-jobs/{job_id}/review")
def review_job(job_id: uuid.UUID, body: ReviewBody, db: Session = Depends(get_db)) -> dict:
    get_or_404(db, GenerationJob, job_id, "job")
    if not transition(db, GenerationJob, job_id, {"review_required"}, body.decision):
        raise err(409, "bad_state", "Chỉ job ở trạng thái review_required mới duyệt được.")
    db.add(Review(subject_type="generation_job", subject_id=job_id, decision=body.decision, reason=body.reason))
    db.commit()
    return serialize(db.get(GenerationJob, job_id))
