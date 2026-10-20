"""API video (đề xuất /video/jobs trong hướng dẫn workflow): ảnh đã duyệt -> clip qua ComfyUI + LTX-2.3 + PromptRelay.

Quy tắc nghiệp vụ:
- Chặn khi ảnh nguồn chưa được duyệt (ảnh đầu ra của job `approved` hoặc ảnh chuẩn của nhân vật).
- Tối đa `video_max_repairs` lượt sửa tự động và trần ngân sách; vượt thì `needs_human`.
- Render thành công => `review_required`, không có nghĩa đã duyệt. Người duyệt quyết định và chọn lý do loại,
  lý do quyết định công đoạn phải làm lại (ảnh nguồn hay tuyến video).
"""
import hashlib
import statistics
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.ml import comfy_client as C
from app.models import (
    Asset, CharacterReference, CharacterVersion, GenerationJob, GenerationOutput, VideoAttempt, VideoJob, VisualCandidate, VisualJob,
)
from app.visual import export as X
from app.workers.celery_app import celery_app

router = APIRouter(prefix="/api/v1/video")

# Lý do loại -> hướng xử lý (hướng dẫn mục 5)
REJECT_ROUTES = {
    "keyframe_identity": "image",      # sai nhân vật ở ảnh nguồn: sửa tuyến ảnh
    "motion_identity": "video",        # ảnh nguồn đạt nhưng mặt/áo biến đổi khi chuyển động
    "wrong_action": "video",
    "interaction_error": "video",      # sai tương tác: tách shot / giảm số hành động
    "visual_artifact": "video",
    "audio_issue": "video",
    "other": "video",
}
RejectReason = Literal["keyframe_identity", "motion_identity", "wrong_action", "interaction_error",
                       "visual_artifact", "audio_issue", "other"]


class Segment(BaseModel):
    start: float = Field(ge=0, le=100)
    end: float = Field(gt=0, le=100)
    prompt: str = Field(min_length=1, max_length=600)


class VideoRequest(BaseModel):
    source_keyframe_asset_id: str
    character_version_id: str | None = None
    scene_version: str | None = Field(None, max_length=64)
    global_prompt: str = Field(min_length=1, max_length=1500)
    motion_segments: list[Segment]
    continuity_constraints: list[str] = Field(default_factory=list, max_length=12)
    width: int = Field(512, ge=256, le=1280)
    height: int = Field(896, ge=256, le=1280)
    frame_count: int = Field(97, ge=9, le=C.MAX_FRAMES)
    fps: int = Field(25, ge=8, le=60)
    seed: int = Field(42, ge=0, lt=2**31)
    audio_mode: Literal["generated", "none"] = "generated"  # Image+Audio→Video chưa kiểm thử => không nhận
    budget_limit: float = Field(0.0, ge=0)

    @model_validator(mode="after")
    def _check(self):
        C.validate_video_params(self.width, self.height, self.frame_count, self.fps)
        self.motion_segments = [Segment(**s) for s in C.validate_segments([s.model_dump() for s in self.motion_segments])]
        return self


class RetryRequest(BaseModel):
    seed: int | None = Field(None, ge=0, lt=2**31)
    global_prompt: str | None = Field(None, min_length=1, max_length=1500)
    motion_segments: list[Segment] | None = None
    frame_count: int | None = Field(None, ge=9, le=C.MAX_FRAMES)
    width: int | None = Field(None, ge=256, le=1280)
    height: int | None = Field(None, ge=256, le=1280)


class ReviewRequest(BaseModel):
    decision: Literal["approve", "reject"]
    reason: RejectReason | None = None
    note: str = Field("", max_length=2000)
    scores: dict[str, int] = Field(default_factory=dict)  # vd identity/action/artifacts: 1–5

    @model_validator(mode="after")
    def _need_reason(self):
        if self.decision == "reject" and not self.reason:
            raise ValueError("loại clip phải kèm reason")
        if any(not 1 <= v <= 5 for v in self.scores.values()):
            raise ValueError("scores phải trong 1–5")
        return self


def keyframe_is_approved(db: Session, asset_id: str) -> tuple[bool, str | None, str | None]:
    """Ảnh được duyệt khi: (1) là ứng viên ĐƯỢC CHỌN của VisualJob đã `approved`; (2) đầu ra của job ảnh cũ `approved`;
    (3) là ảnh chuẩn trong thư viện nhân vật. Trả (ok, id job nguồn, character_version_id nếu suy ra được)."""
    asset = db.get(Asset, asset_id)
    if not asset or not asset.mime_type.startswith("image/"):
        return False, None, None
    cand = db.scalar(select(VisualCandidate).where(VisualCandidate.asset_id == asset_id))
    if cand:
        vj = db.get(VisualJob, cand.job_id)
        ok = vj is not None and vj.state == "approved" and vj.selected_candidate_id == cand.id
        cv = vj.character_version_ids[0] if vj and len(vj.character_version_ids or []) == 1 else None
        return ok, (vj.id if vj else None), cv
    out = db.scalar(select(GenerationOutput).where(GenerationOutput.asset_id == asset_id))
    if out:
        job = db.get(GenerationJob, out.job_id)
        return (job is not None and job.state == "approved"), out.job_id, (job.character_version_id if job else None)
    if db.scalar(select(CharacterReference).where(CharacterReference.asset_id == asset_id)):
        return True, None, None
    return False, None, None


def _attempt_json(a: VideoAttempt) -> dict:
    return {
        "id": a.id, "attempt_no": a.attempt_no, "kind": a.kind, "state": a.state, "prompt_id": a.prompt_id,
        "output_asset_id": a.output_asset_id, "error": a.error, "gpu_seconds": a.gpu_seconds, "total_seconds": a.total_seconds,
        "cost": a.cost, "params": a.params, "review": a.review,
        "created_at": a.created_at.isoformat() if a.created_at else None,
    }


def _job_json(j: VideoJob, attempts: list[VideoAttempt] | None = None) -> dict:
    d = {
        "id": j.id, "state": j.state, "error": j.error, "source_keyframe_asset_id": j.source_keyframe_asset_id,
        "source_generation_job_id": j.source_generation_job_id, "character_version_id": j.character_version_id,
        "scene_version": j.scene_version, "global_prompt": j.global_prompt, "motion_segments": j.motion_segments,
        "continuity_constraints": j.continuity_constraints, "config": j.config, "workflow_version": j.workflow_version,
        "model_version": j.model_version, "audio_mode": j.audio_mode, "budget_limit": j.budget_limit,
        "cost_spent": j.cost_spent, "repair_count": j.repair_count, "max_repairs": settings.video_max_repairs,
        "output_asset_id": j.output_asset_id, "review_reason": j.review_reason, "review_note": j.review_note,
        "created_at": j.created_at.isoformat() if j.created_at else None,
    }
    if attempts is not None:
        d["attempts"] = [_attempt_json(a) for a in attempts]
    return d


def _attempts(db: Session, job_id: str) -> list[VideoAttempt]:
    return list(db.scalars(select(VideoAttempt).where(VideoAttempt.job_id == job_id).order_by(VideoAttempt.attempt_no)))


def _enqueue(db: Session, job: VideoJob, attempt: VideoAttempt) -> None:
    try:
        celery_app.send_task("app.workers.tasks.generate_video", args=[job.id, attempt.id], queue="video")
    except Exception as e:
        attempt.state, attempt.error = "failed", f"enqueue_failed: {e}"[:500]
        job.state, job.error = "failed", attempt.error
        if attempt.kind == "repair":  # chưa render gì: không tính là đã dùng lượt sửa
            job.repair_count = max(0, job.repair_count - 1)
        db.commit()
        raise HTTPException(503, "không đưa được job vào hàng đợi")


@router.post("/jobs", status_code=202)
def create_video_job(req: VideoRequest, db: Session = Depends(get_db), idempotency_key: str | None = Header(None)):
    payload = req.model_dump_json()
    req_hash = hashlib.sha256(payload.encode()).hexdigest()
    if idempotency_key:  # phát lại cùng khóa phải trả job cũ dù ảnh nguồn/workflow đã đổi sau đó
        old = db.scalar(select(VideoJob).where(VideoJob.idempotency_key == idempotency_key))
        if old:
            if old.request_hash != req_hash:
                raise HTTPException(409, "Idempotency-Key đã dùng với payload khác")
            return {"job_id": old.id, "state": old.state}
    if not db.get(Asset, req.source_keyframe_asset_id):
        raise HTTPException(422, "source_keyframe_asset_id không tồn tại")
    ok, src_job, src_char_ver = keyframe_is_approved(db, req.source_keyframe_asset_id)
    if not ok:
        raise HTTPException(409, "ảnh nguồn chưa được duyệt: duyệt job ảnh (review) hoặc đặt làm ảnh chuẩn trước khi tạo video")
    char_ver = req.character_version_id
    if char_ver and not db.get(CharacterVersion, char_ver):
        raise HTTPException(422, "character_version_id không tồn tại")
    if not char_ver:  # ghim đúng version nhân vật đã dùng cho ảnh nguồn
        char_ver = src_char_ver
    try:
        wf = C.load_workflow()
    except C.ComfyError as e:
        raise HTTPException(503, f"{e.code}: {e}")

    cfg = {"width": req.width, "height": req.height, "frame_count": req.frame_count, "fps": req.fps, "seed": req.seed,
           "duration_s": round(req.frame_count / req.fps, 2)}
    job = VideoJob(
        idempotency_key=idempotency_key, request_hash=req_hash, source_keyframe_asset_id=req.source_keyframe_asset_id,
        source_generation_job_id=src_job, character_version_id=char_ver, scene_version=req.scene_version,
        global_prompt=req.global_prompt, motion_segments=[s.model_dump() for s in req.motion_segments],
        continuity_constraints=req.continuity_constraints, config=cfg, workflow_version=wf["version"],
        model_version=wf.get("model_version"), audio_mode=req.audio_mode, budget_limit=req.budget_limit,
    )
    db.add(job)
    try:
        db.flush()
    except IntegrityError:  # hai request cùng khóa đua nhau: người thua trả job của người thắng
        db.rollback()
        old = db.scalar(select(VideoJob).where(VideoJob.idempotency_key == idempotency_key))
        if old and old.request_hash == req_hash:
            return {"job_id": old.id, "state": old.state}
        raise HTTPException(409, "Idempotency-Key đã dùng với payload khác")
    attempt = VideoAttempt(job_id=job.id, attempt_no=1, kind="initial", params=_resolve_params(job))
    db.add(attempt)
    db.commit()
    _enqueue(db, job, attempt)
    return {"job_id": job.id, "state": job.state, "attempt_id": attempt.id, "workflow_verified": wf["verified"]}


def _resolve_params(job: VideoJob) -> dict:
    return {**job.config, "global_prompt": job.global_prompt, "motion_segments": job.motion_segments,
            "audio_mode": job.audio_mode, "workflow_version": job.workflow_version}


@router.get("/jobs/{job_id}")
def get_video_job(job_id: str, db: Session = Depends(get_db)):
    job = db.get(VideoJob, job_id)
    if not job:
        raise HTTPException(404, "video job không tồn tại")
    return _job_json(job, _attempts(db, job_id))


@router.get("/jobs")
def list_video_jobs(limit: int = 24, offset: int = 0, state: str | None = None, db: Session = Depends(get_db)):
    limit = max(1, min(limit, 100))
    q = select(VideoJob)
    if state:
        q = q.where(VideoJob.state == state)
    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    rows = db.scalars(q.order_by(VideoJob.created_at.desc(), VideoJob.id).limit(limit).offset(max(0, offset)))
    return {"total": total, "items": [_job_json(j) for j in rows]}


@router.post("/jobs/{job_id}/retry", status_code=202)
def retry_video_job(job_id: str, req: RetryRequest, db: Session = Depends(get_db)):
    """Lượt sửa tự động: tạo attempt mới trên cùng job. Mỗi lần retry (kể cả sau lỗi thực thi) tính là một lượt sửa."""
    job = db.get(VideoJob, job_id)
    if not job:
        raise HTTPException(404, "video job không tồn tại")
    if job.state not in ("rejected", "failed"):
        raise HTTPException(409, f"job đang '{job.state}'; chỉ sửa được job bị loại hoặc thất bại")
    if job.review_reason and REJECT_ROUTES.get(job.review_reason) == "image":
        raise HTTPException(409, "lỗi nằm ở ảnh nguồn: quay về công đoạn tạo ảnh, tạo video mới khi có ảnh được duyệt")
    if job.budget_limit and job.cost_spent >= job.budget_limit:
        job.state, job.error = "needs_human", "budget_exhausted: đã hết ngân sách, chuyển người xử lý"
        db.commit()
        raise HTTPException(409, job.error)
    if job.repair_count >= settings.video_max_repairs:
        job.state, job.error = "needs_human", f"repair_limit: đã dùng {job.repair_count} lượt sửa, chuyển người xử lý"
        db.commit()
        raise HTTPException(409, job.error)

    cfg = dict(job.config)
    for k in ("seed", "frame_count", "width", "height"):
        if getattr(req, k) is not None:
            cfg[k] = getattr(req, k)
    segs = job.motion_segments
    try:
        C.validate_video_params(cfg["width"], cfg["height"], cfg["frame_count"], cfg["fps"])
        if req.motion_segments is not None:
            segs = C.validate_segments([s.model_dump() for s in req.motion_segments])
        wf = C.load_workflow()  # lượt sửa dùng workflow hiện hành và ghi lại đúng phiên bản đó
    except ValueError as e:
        raise HTTPException(422, str(e))
    except C.ComfyError as e:
        raise HTTPException(503, f"{e.code}: {e}")
    cfg["duration_s"] = round(cfg["frame_count"] / cfg["fps"], 2)

    # Chiếm job bằng UPDATE có điều kiện: hai retry đồng thời chỉ một cái thắng.
    won = db.execute(update(VideoJob).where(VideoJob.id == job_id, VideoJob.state == job.state,
                                            VideoJob.repair_count == job.repair_count).values(
        state="queued", error=None, review_reason=None, repair_count=job.repair_count + 1, config=cfg,
        motion_segments=segs, global_prompt=req.global_prompt or job.global_prompt,
        workflow_version=wf["version"], model_version=wf.get("model_version")))
    if won.rowcount == 0:
        db.rollback()
        raise HTTPException(409, "job vừa được xử lý bởi yêu cầu khác")
    db.expire(job)
    n = (db.scalar(select(func.max(VideoAttempt.attempt_no)).where(VideoAttempt.job_id == job_id)) or 0) + 1
    attempt = VideoAttempt(job_id=job_id, attempt_no=n, kind="repair", params=_resolve_params(job))
    db.add(attempt)
    db.commit()
    _enqueue(db, job, attempt)
    return {"job_id": job.id, "state": job.state, "attempt_id": attempt.id, "repair_count": job.repair_count}


@router.post("/jobs/{job_id}/review")
def review_video_job(job_id: str, req: ReviewRequest, db: Session = Depends(get_db)):
    job = db.get(VideoJob, job_id)
    if not job:
        raise HTTPException(404, "video job không tồn tại")
    latest = next((a for a in reversed(_attempts(db, job_id)) if a.state == "rendered"), None)
    if job.state != "review_required" or not latest:
        raise HTTPException(409, f"job đang '{job.state}'; chỉ duyệt được khi 'review_required' và đã có clip")
    if req.decision == "approve":
        new_state, error = "approved", None
    elif REJECT_ROUTES[req.reason] == "image":
        new_state, error = "returned_to_image", "keyframe_identity: ảnh nguồn sai nhân vật, quay về công đoạn ảnh"
    else:
        new_state, error = "rejected", None
    # Chuyển trạng thái có điều kiện: hai lần duyệt đồng thời chỉ một cái được ghi nhận.
    won = db.execute(update(VideoJob).where(VideoJob.id == job_id, VideoJob.state == "review_required").values(
        state=new_state, error=error, review_note=req.note or None,
        review_reason=None if req.decision == "approve" else req.reason,
        output_asset_id=latest.output_asset_id if req.decision == "approve" else None))
    if won.rowcount == 0:
        db.rollback()
        raise HTTPException(409, "job vừa được duyệt bởi yêu cầu khác")
    latest.review = {"decision": req.decision, "reason": req.reason, "note": req.note, "scores": req.scores}
    if req.decision == "approve":
        asset = db.get(Asset, latest.output_asset_id)
        if asset:
            asset.provenance = {**asset.provenance, "review": "approved"}
    db.commit()
    db.refresh(job)
    return _job_json(job, _attempts(db, job_id))


@router.get("/metrics")
def video_metrics(db: Session = Depends(get_db)):
    """Số liệu pilot (hướng dẫn mục 6): tỷ lệ đạt, độ trễ p50/p95, chi phí / clip đạt, lý do loại.
    Chỉ tính attempt đã render; attempt thất bại được đếm riêng nhưng chi phí của chúng vẫn nằm trong tổng."""
    atts = list(db.scalars(select(VideoAttempt)))
    rendered = [a for a in atts if a.state == "rendered"]
    reviewed = [a for a in rendered if a.review]
    approved = [a for a in reviewed if a.review.get("decision") == "approve"]
    lat = sorted(a.total_seconds for a in rendered if a.total_seconds is not None)

    def pct(p: float):
        if not lat:
            return None
        return lat[min(len(lat) - 1, int(round(p * (len(lat) - 1))))]

    reasons: dict[str, int] = {}
    for a in reviewed:
        if a.review.get("decision") == "reject":
            reasons[a.review.get("reason") or "other"] = reasons.get(a.review.get("reason") or "other", 0) + 1
    total_cost = sum(a.cost or 0 for a in atts)
    return {
        "attempts_total": len(atts), "attempts_rendered": len(rendered), "attempts_failed": sum(a.state == "failed" for a in atts),
        "reviewed": len(reviewed), "approved": len(approved),
        "approval_rate": (len(approved) / len(reviewed)) if reviewed else None,
        "latency_s": {"p50": statistics.median(lat) if lat else None, "p95": pct(0.95), "n": len(lat)},
        "total_cost": total_cost, "cost_per_approved": (total_cost / len(approved)) if approved else None,
        "reject_reasons": reasons, "pilot_pass_threshold": {"approval_rate_min": 0.8},
        "note": "Chi phí gồm cả attempt thất bại/sửa. Đơn vị = giây GPU × video_cost_per_gpu_second (mặc định 1).",
    }


@router.get("/health")
def video_health():
    """Trạng thái cầu nối tới ComfyUI và workflow đang ghim. Không giả lập: lỗi kết nối được báo thẳng."""
    out: dict = {"comfyui_url": settings.comfyui_url}
    try:
        wf = C.load_workflow()
        out.update(workflow_version=wf["version"], workflow_verified=wf["verified"])
    except C.ComfyError as e:
        return {**out, "ok": False, "error": f"{e.code}: {e}"}
    try:
        oi = C._json("/object_info", timeout=120)
        miss = sorted({n["class_type"] for n in wf["graph"].values() if n["class_type"] not in oi})
        files = C.missing_model_files(wf, oi)
        errs = C.structure_errors(wf, oi)
        out.update(ok=not errs and not files and wf["verified"], missing_nodes=miss, missing_model_files=files, structure_errors=errs,
                   workflow_status=wf["status"])
        if files:
            out["warning"] = "chưa có file model trong ComfyUI: chạy `docker compose --profile init run --rm models-init`"
        elif wf["status"] != "verified":
            out["warning"] = ("workflow đã khớp schema ComfyUI nhưng CHƯA có clip nào render thật: chạy scripts/smoke_video.py rồi đặt _meta.status=verified"
                              if wf["verified"] else "workflow là bản nháp chưa kiểm chứng")
    except C.ComfyError as e:
        out.update(ok=False, error=f"{e.code}: {e}")
    return out


class VideoExportBody(BaseModel):
    target: Literal["auto", "1080x1920", "1920x1080", "native"] = "auto"
    label: bool = True


@router.post("/jobs/{job_id}/export", status_code=201)
def export_video(job_id: str, body: VideoExportBody, db: Session = Depends(get_db)):
    """Xuất bản clip đã DUYỆT: đưa về khung đích, chèn nhãn AI hiển thị, ký C2PA, đọc lại xác minh (mục 3.6 báo cáo)."""
    from app.api_visual import _export_json
    from app.models import AssetBlob, ExportRecord
    job = db.get(VideoJob, job_id)
    if not job:
        raise HTTPException(404, "video job không tồn tại")
    if job.state != "approved" or not job.output_asset_id:
        raise HTTPException(409, "chỉ xuất bản clip đã được người phụ trách duyệt")
    if settings.require_label and not body.label:
        raise HTTPException(422, "cấu hình hiện bắt buộc nhãn AI hiển thị (REQUIRE_LABEL=true)")
    src, blob = db.get(Asset, job.output_asset_id), db.get(AssetBlob, job.output_asset_id)
    w, h = job.config["width"], job.config["height"]
    target = body.target if body.target != "auto" else ("1080x1920" if h > w else "1920x1080" if w > h else "native")
    edited = False
    try:
        data, info = X.render_video(blob.data, (w, h), target, settings.ai_label_text_generated if body.label else None)
    except X.ExportError as e:
        raise HTTPException(422, str(e))
    prov = {"asset_id": src.id, "source": "ai", "lineage": "generated", "video_job_id": job.id, "model": job.model_version,
            "workflow_version": job.workflow_version, "seed": job.config.get("seed"), "fps": job.config.get("fps"),
            "frames": job.config.get("frame_count"), "source_keyframe_asset_id": job.source_keyframe_asset_id,
            "source_generation_job_id": job.source_generation_job_id, "character_version_id": job.character_version_id,
            "audio_mode": job.audio_mode, "source_sha256": src.sha256, "reviewed_note": job.review_note}
    signed, c2 = X.sign_and_verify(data, "video/mp4", ".mp4", X.manifest_json(f"video/{job.id[:8]}", edited, prov))
    final = signed or data
    blocked = bool(settings.require_c2pa and not (c2.get("signed") and c2.get("verified")))
    rec = ExportRecord(job_id=job.id, kind="video", source_asset_id=src.id, target=target, label=info.get("label") or {}, c2pa=c2,
                       status="blocked" if blocked else "released", provenance={**prov, "render": info})
    if not blocked:
        a = Asset(kind="export", mime_type="video/mp4", width=info["size"][0], height=info["size"][1], byte_size=len(final),
                  sha256=hashlib.sha256(final).hexdigest(), storage_backend=settings.asset_storage_backend,
                  provenance={"source": "ai", "export_of": src.id, "c2pa_signed": c2.get("signed"), "label": bool(info.get("label"))})
        db.add(a)
        db.flush()
        db.add(AssetBlob(asset_id=a.id, data=final))
        rec.export_asset_id, rec.sha256 = a.id, a.sha256
    db.add(rec)
    db.commit()
    return _export_json(rec)
