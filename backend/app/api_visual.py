"""API pipeline ảnh theo hợp đồng báo cáo EMVOOX1 mục 3.3:
POST /visual/jobs · GET /visual/jobs/{id} · POST /visual/jobs/{id}/review — cộng Model Registry, Scene/Costume Library, nhập thủ công
(Midjourney), xuất bản cuối (nhãn AI + C2PA).
Trạng thái: queued, running, review_required, approved, failed (+ rejected, needs_human).
"""
import hashlib
import io
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.models import (
    Asset, AssetBlob, Character, Costume, ExportRecord, ProviderModel, Scene, SceneVersion, VisualAttempt,
    VisualCandidate, VisualJob,
)
from app.visual import export as X
from app.visual import router as R
from app.visual.service import ServiceError, VisualRequest, continuity_warnings, create_visual_job, worker_status
from app.workers.celery_app import celery_app

router = APIRouter(prefix="/api/v1")


def _enqueue(db: Session, job: VisualJob) -> None:
    try:
        celery_app.send_task("app.workers.tasks.visual_run", args=[job.id], queue="visual")
    except Exception as e:  # noqa: BLE001
        job.state, job.error = "failed", f"enqueue_failed: {e}"[:500]
        db.commit()
        raise HTTPException(503, "không đưa được job vào hàng đợi")


# ------------------------------------------------------------------ Model Registry
@router.get("/visual/models")
def list_models(db: Session = Depends(get_db)):
    worker = worker_status(db)
    out = []
    for m in db.scalars(select(ProviderModel).order_by(ProviderModel.route, ProviderModel.key)):
        ok, why = R.availability(m, worker)
        out.append({"key": m.key, "name": m.name, "vendor": m.vendor, "family": m.family, "route": m.route,
                    "capabilities": m.capabilities, "license": m.license, "commercial_ok": m.commercial_ok, "pricing": m.pricing,
                    "min_vram_gb": m.min_vram_gb, "tasks": m.tasks, "verification": m.verification, "checked_at": m.checked_at,
                    "notes": m.notes, "enabled": m.enabled, "benchmark": m.benchmark, "available": ok, "availability": why})
    return {"worker": worker, "models": out}


class ModelPatch(BaseModel):
    enabled: bool


@router.patch("/visual/models/{key}")
def patch_model(key: str, body: ModelPatch, db: Session = Depends(get_db)):
    m = db.get(ProviderModel, key)
    if not m:
        raise HTTPException(404, "model không có trong registry")
    m.enabled = body.enabled
    db.commit()
    return {"key": key, "enabled": m.enabled}


# ------------------------------------------------------------------ Jobs
def _job_json(db: Session, j: VisualJob, full: bool = True) -> dict:
    d = {
        "id": j.id, "project_id": j.project_id, "episode_id": j.episode_id, "shot_id": j.shot_id, "state": j.state, "error": j.error,
        "aspect_ratio": j.aspect_ratio, "num_variants": j.num_variants, "seed": j.seed, "pinned_model_key": j.pinned_model_key,
        "budget_limit": j.budget_limit, "cost_spent": round(j.cost_spent or 0, 6), "repair_count": j.repair_count,
        "max_repairs": settings.visual_max_repairs, "auto_repair": j.auto_repair, "selected_candidate_id": j.selected_candidate_id,
        "reviewer": j.reviewer, "reviewed_at": j.reviewed_at.isoformat() if j.reviewed_at else None, "review_note": j.review_note,
        "created_at": j.created_at.isoformat() if j.created_at else None, "benchmark_run_id": j.benchmark_run_id,
    }
    if not full:
        return d
    atts = db.scalars(select(VisualAttempt).where(VisualAttempt.job_id == j.id).order_by(VisualAttempt.attempt_no)).all()
    cands = db.scalars(select(VisualCandidate).where(VisualCandidate.job_id == j.id).order_by(VisualCandidate.created_at)).all()
    res = j.scene_spec.get("_resolved", {})
    d.update({
        "prompt": j.prompt, "negative_prompt": j.negative_prompt, "route_plan": j.route_plan, "task": j.scene_spec.get("_task"),
        "pins": {"character_version_ids": j.character_version_ids, "scene_version_id": j.scene_version_id, "costume_ids": j.costume_ids,
                 "reference_asset_ids": j.reference_asset_ids},
        "characters": [{"code": c["code"], "version": c["version"], "costume": c.get("costume_en")} for c in res.get("characters", [])],
        "scene": {k: res["scene"][k] for k in ("code", "name", "version")} if res.get("scene") else None,
        "continuity_warnings": continuity_warnings(db, j), "review_history": j.review_history,
        "attempts": [{"id": a.id, "no": a.attempt_no, "kind": a.kind, "model_key": a.model_key, "route": a.route, "state": a.state,
                      "error": a.error, "instruction": a.instruction, "cost_api": a.cost_api, "cost_gpu": a.cost_gpu, "metrics": a.metrics,
                      "remote_id": a.remote_id} for a in atts],
        "candidates": [{"id": c.id, "attempt_id": c.attempt_id, "asset_id": c.asset_id, "seed": c.seed, "qc_verdict": c.qc_verdict,
                        "qc_score": c.qc_score, "qc": c.qc} for c in cands],
    })
    return d


@router.post("/visual/jobs", status_code=202)
def create_job(req: VisualRequest, db: Session = Depends(get_db), idempotency_key: str | None = Header(None)):
    try:
        existing = db.scalar(select(VisualJob.id).where(VisualJob.idempotency_key == idempotency_key)) if idempotency_key else None
        job = create_visual_job(db, req, idempotency_key)
    except ServiceError as e:
        raise HTTPException(e.status, e.msg)
    if existing:
        return {"job_id": job.id, "state": job.state, "replayed": True}
    _enqueue(db, job)
    return {"job_id": job.id, "state": job.state, "task": job.scene_spec.get("_task"), "route_plan": [
        {k: p[k] for k in ("model_key", "route", "est_cost", "preferred", "verification", "reason")} for p in job.route_plan],
        "excluded": getattr(job, "_rejected", [])}


@router.get("/visual/jobs/{job_id}")
def get_job(job_id: str, db: Session = Depends(get_db)):
    j = db.get(VisualJob, job_id)
    if not j:
        raise HTTPException(404, "visual job không tồn tại")
    return _job_json(db, j)


@router.get("/visual/jobs")
def list_jobs(limit: int = Query(24, ge=1, le=200), offset: int = Query(0, ge=0), state: str | None = None, project_id: str | None = None,
              shot_id: str | None = None, db: Session = Depends(get_db)):
    q = select(VisualJob).where(VisualJob.benchmark_run_id.is_(None))
    if state:
        q = q.where(VisualJob.state == state)
    if project_id:
        q = q.where(VisualJob.project_id == project_id)
    if shot_id:
        q = q.where(VisualJob.shot_id == shot_id)
    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    rows = db.scalars(q.order_by(VisualJob.created_at.desc(), VisualJob.id).limit(limit).offset(offset)).all()
    items = []
    for j in rows:
        d = _job_json(db, j, full=False)
        sel = db.get(VisualCandidate, j.selected_candidate_id) if j.selected_candidate_id else None
        first = sel or db.scalar(select(VisualCandidate).where(VisualCandidate.job_id == j.id).order_by(VisualCandidate.created_at))
        d["thumb_asset_id"] = first.asset_id if first else None
        d["model_key"] = db.scalar(select(VisualAttempt.model_key).where(VisualAttempt.job_id == j.id).order_by(VisualAttempt.attempt_no.desc()))
        items.append(d)
    return {"total": total, "items": items}


class ReviewBody(BaseModel):
    decision: Literal["approve", "reject", "repair"]
    candidate_id: str | None = None
    reviewer: str | None = Field(None, max_length=100)
    note: str = Field("", max_length=2000)
    reason: str | None = Field(None, max_length=200)           # bắt buộc khi reject
    instruction: str | None = Field(None, max_length=600)      # bắt buộc khi repair
    model_key: str | None = None                                # repair: đổi tuyến
    override_qc: bool = False                                   # approve ảnh QC fail: phải xác nhận có chủ đích
    budget_increase: float = Field(0.0, ge=0)


@router.post("/visual/jobs/{job_id}/review")
def review_job(job_id: str, body: ReviewBody, db: Session = Depends(get_db), x_reviewer: str | None = Header(None)):
    j = db.get(VisualJob, job_id)
    if not j:
        raise HTTPException(404, "visual job không tồn tại")
    reviewer = (body.reviewer or x_reviewer or "").strip()
    if not reviewer:
        raise HTTPException(422, "cần reviewer (tên người duyệt) để lưu lịch sử duyệt")
    if j.state not in ("review_required", "needs_human"):
        raise HTTPException(409, f"job đang '{j.state}'; chỉ duyệt được khi review_required/needs_human")
    when = datetime.now(timezone.utc)
    entry = {"at": when.isoformat(), "reviewer": reviewer, "decision": body.decision, "note": body.note}
    hist = list(j.review_history or [])
    from_state = j.state

    if body.decision == "approve":
        cid = body.candidate_id or j.selected_candidate_id
        c = db.get(VisualCandidate, cid) if cid else None
        if not c or c.job_id != j.id:
            raise HTTPException(422, "cần candidate_id thuộc job này")
        if c.qc_verdict == "fail" and not body.override_qc:
            raise HTTPException(409, "ảnh này lỗi QC tự động; đặt override_qc=true nếu người duyệt vẫn chấp nhận (sẽ được ghi lại)")
        entry.update(candidate_id=cid, qc_verdict=c.qc_verdict, override_qc=bool(body.override_qc and c.qc_verdict == "fail"))
        won = db.execute(update(VisualJob).where(VisualJob.id == j.id, VisualJob.state == from_state).values(
            state="approved", selected_candidate_id=cid, reviewer=reviewer, reviewed_at=when, review_note=body.note or None,
            review_history=hist + [entry], error=None))
    elif body.decision == "reject":
        if not body.reason:
            raise HTTPException(422, "loại ảnh phải kèm reason")
        entry["reason"] = body.reason
        won = db.execute(update(VisualJob).where(VisualJob.id == j.id, VisualJob.state == from_state).values(
            state="rejected", reviewer=reviewer, reviewed_at=when, review_note=body.note or None, review_history=hist + [entry]))
    else:  # repair do người yêu cầu (không tính vào 2 lượt sửa tự động; vẫn chịu ngân sách)
        if not body.instruction:
            raise HTTPException(422, "sửa ảnh phải kèm instruction")
        last = db.scalar(select(VisualAttempt).where(VisualAttempt.job_id == j.id).order_by(VisualAttempt.attempt_no.desc()))
        key = body.model_key or last.model_key
        m = db.get(ProviderModel, key)
        if not m or m.route == "manual":
            raise HTTPException(422, "model_key không hợp lệ cho sửa tự động")
        ok, why = R.availability(m, worker_status(db))
        if not ok:
            raise HTTPException(422, f"model {key} không khả dụng: {why}")
        entry.update(instruction=body.instruction, model_key=key)
        new_budget = (j.budget_limit + body.budget_increase) if j.budget_limit else 0.0
        won = db.execute(update(VisualJob).where(VisualJob.id == j.id, VisualJob.state == from_state).values(
            state="queued", reviewer=reviewer, reviewed_at=when, review_history=hist + [entry], budget_limit=new_budget, error=None))
        if won.rowcount:
            n = (last.attempt_no if last else 0) + 1
            db.add(VisualAttempt(job_id=j.id, attempt_no=n, kind="human_repair", model_key=key, route=m.route, state="queued",
                                 instruction=body.instruction))
    if won.rowcount == 0:
        db.rollback()
        raise HTTPException(409, "job vừa được xử lý bởi yêu cầu khác")
    db.commit()
    db.refresh(j)
    if body.decision == "repair":
        _enqueue(db, j)
    return _job_json(db, j)


@router.post("/visual/jobs/{job_id}/manual-import", status_code=202)
async def manual_import(job_id: str, files: list[UploadFile] = File(...), model_key: str = Form("midjourney"), cost: float = Form(0.0),
                        note: str = Form(""), db: Session = Depends(get_db)):
    """Nhập ảnh tạo NGOÀI hệ thống (Midjourney không có API). Ảnh đi qua QC như mọi ứng viên; chi phí tự khai được cộng vào job."""
    j = db.get(VisualJob, job_id)
    m = db.get(ProviderModel, model_key)
    if not j or not m or m.route != "manual":
        raise HTTPException(404, "job hoặc model thủ công không tồn tại")
    if j.state not in ("review_required", "needs_human", "rejected"):
        raise HTTPException(409, f"job đang '{j.state}'")
    if not 1 <= len(files) <= 8:
        raise HTTPException(422, "nhập 1–8 ảnh")
    n = (db.scalar(select(func.max(VisualAttempt.attempt_no)).where(VisualAttempt.job_id == j.id)) or 0) + 1
    att = VisualAttempt(job_id=j.id, attempt_no=n, kind="manual", model_key=m.key, route="manual", state="done", cost_api=cost,
                        instruction=note or None, metrics={"manual": True, "n_images": len(files)})
    db.add(att)
    db.flush()
    for f in files:
        data = await f.read()
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, "file quá lớn")
        try:
            im = Image.open(io.BytesIO(data))
            im.verify()
            im = Image.open(io.BytesIO(data))
            png = io.BytesIO()
            im.convert("RGB").save(png, "PNG")
            data, w, h = png.getvalue(), im.width, im.height
        except Exception:
            raise HTTPException(422, f"{f.filename}: không phải ảnh hợp lệ")
        a = Asset(kind="generated", mime_type="image/png", width=w, height=h, byte_size=len(data), sha256=hashlib.sha256(data).hexdigest(),
                  storage_backend=settings.asset_storage_backend,
                  provenance={"source": "ai", "model_key": m.key, "model_name": m.name, "route": "manual", "visual_job_id": j.id,
                              "attempt_id": att.id, "imported_filename": f.filename, "note": "tạo ngoài hệ thống, người dùng nhập"})
        db.add(a)
        db.flush()
        db.add(AssetBlob(asset_id=a.id, data=data))
        db.add(VisualCandidate(job_id=j.id, attempt_id=att.id, asset_id=a.id))
    j.cost_spent = (j.cost_spent or 0) + cost
    j.state = "queued"
    db.commit()
    _enqueue(db, j)
    return {"job_id": j.id, "attempt_id": att.id, "state": j.state}


# ------------------------------------------------------------------ Scene / Costume Library
class SceneSpecBody(BaseModel):
    layout: str = ""
    openings: str = ""                 # vị trí cửa, đồ vật cố định
    time_of_day: str = ""
    light_direction: str = ""
    palette: str = ""
    props: list[str] = Field(default_factory=list)


class SceneCreate(BaseModel):
    code: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=200)
    spec: SceneSpecBody = SceneSpecBody()
    reference_asset_ids: list[str] = Field(default_factory=list, max_length=4)


def _scene_json(db: Session, s: Scene) -> dict:
    vs = db.scalars(select(SceneVersion).where(SceneVersion.scene_id == s.id).order_by(SceneVersion.version.desc())).all()
    return {"code": s.code, "name": s.name, "current_version": next((v.version for v in vs if v.id == s.current_version_id), None),
            "versions": [{"id": v.id, "version": v.version, "spec": v.spec, "reference_asset_ids": v.reference_asset_ids} for v in vs]}


def _check_assets(db: Session, ids: list[str]) -> None:
    for a in ids:
        if not db.get(Asset, a):
            raise HTTPException(422, f"asset không tồn tại: {a}")


@router.post("/scenes", status_code=201)
def create_scene(body: SceneCreate, db: Session = Depends(get_db)):
    if db.scalar(select(Scene).where(Scene.code == body.code)):
        raise HTTPException(409, "mã bối cảnh đã tồn tại")
    _check_assets(db, body.reference_asset_ids)
    s = Scene(code=body.code, name=body.name)
    db.add(s)
    db.flush()
    v = SceneVersion(scene_id=s.id, version=1, spec=body.spec.model_dump(), reference_asset_ids=body.reference_asset_ids)
    db.add(v)
    db.flush()
    s.current_version_id = v.id
    db.commit()
    return _scene_json(db, s)


@router.get("/scenes")
def list_scenes(db: Session = Depends(get_db)):
    return [_scene_json(db, s) for s in db.scalars(select(Scene).order_by(Scene.code))]


@router.post("/scenes/{code}/versions", status_code=201)
def new_scene_version(code: str, body: SceneCreate, db: Session = Depends(get_db)):
    """Đổi bối cảnh => version mới (job cũ vẫn trỏ về version đã dùng)."""
    s = db.scalar(select(Scene).where(Scene.code == code))
    if not s:
        raise HTTPException(404, "bối cảnh không tồn tại")
    _check_assets(db, body.reference_asset_ids)
    last = db.scalar(select(func.max(SceneVersion.version)).where(SceneVersion.scene_id == s.id)) or 0
    v = SceneVersion(scene_id=s.id, version=last + 1, spec=body.spec.model_dump(), reference_asset_ids=body.reference_asset_ids)
    db.add(v)
    db.flush()
    s.current_version_id = v.id
    s.name = body.name
    db.commit()
    return _scene_json(db, s)


class CostumeCreate(BaseModel):
    code: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_-]+$")
    description_en: str = Field(min_length=1, max_length=500)
    reference_asset_id: str | None = None


@router.post("/characters/{character_id}/costumes", status_code=201)
def create_costume(character_id: str, body: CostumeCreate, db: Session = Depends(get_db)):
    if not db.get(Character, character_id):
        raise HTTPException(404, "nhân vật không tồn tại")
    if db.scalar(select(Costume).where(Costume.character_id == character_id, Costume.code == body.code)):
        raise HTTPException(409, "mã trang phục đã tồn tại cho nhân vật này")
    if body.reference_asset_id:
        _check_assets(db, [body.reference_asset_id])
    c = Costume(character_id=character_id, code=body.code, description_en=body.description_en, reference_asset_id=body.reference_asset_id)
    db.add(c)
    db.commit()
    return {"id": c.id, "code": c.code}


@router.get("/characters/{character_id}/costumes")
def list_costumes(character_id: str, db: Session = Depends(get_db)):
    return [{"id": c.id, "code": c.code, "description_en": c.description_en, "reference_asset_id": c.reference_asset_id}
            for c in db.scalars(select(Costume).where(Costume.character_id == character_id).order_by(Costume.code))]


# ------------------------------------------------------------------ Xuất bản cuối
class ExportBody(BaseModel):
    target: Literal["auto", "1080x1920", "1920x1080", "native"] = "auto"
    label: bool = True


@router.post("/visual/jobs/{job_id}/export", status_code=201)
def export_job(job_id: str, body: ExportBody, db: Session = Depends(get_db)):
    j = db.get(VisualJob, job_id)
    if not j:
        raise HTTPException(404, "visual job không tồn tại")
    if j.state != "approved" or not j.selected_candidate_id:
        raise HTTPException(409, "chỉ xuất bản ảnh đã được người phụ trách duyệt")
    if settings.require_label and not body.label:
        raise HTTPException(422, "cấu hình hiện bắt buộc nhãn AI hiển thị (REQUIRE_LABEL=true)")
    cand = db.get(VisualCandidate, j.selected_candidate_id)
    src = db.get(Asset, cand.asset_id)
    blob = db.get(AssetBlob, cand.asset_id)
    target = body.target if body.target != "auto" else {"9:16": "1080x1920", "16:9": "1920x1080"}.get(j.aspect_ratio, "native")
    res = j.scene_spec.get("_resolved", {})
    edited = bool(res.get("extra_refs"))
    text = settings.ai_label_text_edited if edited else settings.ai_label_text_generated
    faces = []
    try:
        from app.visual import qc as QC
        eng = QC.face_engine()
        if eng:
            img = Image.open(io.BytesIO(blob.data)).convert("RGB")
            boxes, _ = eng.mtcnn.detect(img)
            if boxes is not None:
                sx = 1.0
                faces = [tuple(float(v) * sx for v in b) for b in boxes]
    except Exception:  # noqa: BLE001
        faces = []
    try:
        png, info = X.render_image(blob.data, target, text if body.label else None, _scale_boxes(faces, blob.data, target))
    except X.ExportError as e:
        raise HTTPException(422, str(e))
    prov = {
        "asset_id": cand.asset_id, "source": "ai", "lineage": "edited" if edited else "generated", "job_id": j.id,
        "project_id": j.project_id, "shot_id": j.shot_id, "model": src.provenance.get("model_name"), "model_key": src.provenance.get("model_key"),
        "model_id": src.provenance.get("model_id"), "seed": src.provenance.get("seed"), "generated_at": src.created_at.isoformat() if src.created_at else None,
        "character_version_ids": j.character_version_ids, "scene_version_id": j.scene_version_id,
        "prompt_sha256": src.provenance.get("prompt_sha256"), "reviewer": j.reviewer,
        "reviewed_at": j.reviewed_at.isoformat() if j.reviewed_at else None, "source_sha256": src.sha256,
    }
    manifest = X.manifest_json(f"{j.project_id}/{j.shot_id}", edited, {k: v for k, v in prov.items() if k not in ("reviewer",)})
    signed, c2 = X.sign_and_verify(png, "image/png", ".png", manifest)
    final = signed or png
    blocked = bool(settings.require_c2pa and not (c2.get("signed") and c2.get("verified")))
    rec = ExportRecord(job_id=j.id, kind="image", source_asset_id=cand.asset_id, target=target, label=info.get("label") or {},
                       c2pa=c2, invisible_watermark="provider-native-only" if src.provenance.get("synthid") else "none",
                       status="blocked" if blocked else "released", provenance={**prov, "render": info})
    if not blocked:
        a = Asset(kind="export", mime_type="image/png", width=info["size"][0], height=info["size"][1], byte_size=len(final),
                  sha256=hashlib.sha256(final).hexdigest(), storage_backend=settings.asset_storage_backend,
                  provenance={"source": "ai", "export_of": cand.asset_id, "c2pa_signed": c2.get("signed"), "label": bool(info.get("label"))})
        db.add(a)
        db.flush()
        db.add(AssetBlob(asset_id=a.id, data=final))
        rec.export_asset_id, rec.sha256 = a.id, a.sha256
    db.add(rec)
    db.commit()
    return _export_json(rec)


def _scale_boxes(faces, data: bytes, target: str):
    """Quy đổi hộp mặt từ ảnh nguồn sang toạ độ ảnh đích (cover + crop giữa)."""
    if not faces or target == "native" or target not in X.TARGETS:
        return faces
    w, h = Image.open(io.BytesIO(data)).size
    tw, th = X.TARGETS[target]
    s = max(tw / w, th / h)
    ox, oy = (w * s - tw) / 2, (h * s - th) / 2
    return [(b[0] * s - ox, b[1] * s - oy, b[2] * s - ox, b[3] * s - oy) for b in faces]


def _export_json(r: ExportRecord) -> dict:
    return {"id": r.id, "job_id": r.job_id, "kind": r.kind, "status": r.status, "target": r.target, "export_asset_id": r.export_asset_id,
            "label": r.label, "c2pa": r.c2pa, "invisible_watermark": r.invisible_watermark, "sha256": r.sha256,
            "provenance": r.provenance, "created_at": r.created_at.isoformat() if r.created_at else None}


@router.get("/exports/{export_id}")
def get_export(export_id: str, db: Session = Depends(get_db)):
    r = db.get(ExportRecord, export_id)
    if not r:
        raise HTTPException(404, "bản xuất không tồn tại")
    return _export_json(r)


@router.get("/exports/{export_id}/file")
def export_file(export_id: str, db: Session = Depends(get_db)):
    r = db.get(ExportRecord, export_id)
    if not r:
        raise HTTPException(404, "bản xuất không tồn tại")
    if r.status != "released" or not r.export_asset_id:
        raise HTTPException(409, "bản xuất bị chặn phát hành: thiếu dấu bắt buộc (xem c2pa.error)")
    a, blob = db.get(Asset, r.export_asset_id), db.get(AssetBlob, r.export_asset_id)
    ext = "mp4" if a.mime_type == "video/mp4" else "png"
    return Response(blob.data, media_type=a.mime_type, headers={"Content-Disposition": f'attachment; filename="emvoox-{r.id[:8]}.{ext}"'})


@router.get("/exports")
def list_exports(job_id: str | None = None, db: Session = Depends(get_db)):
    q = select(ExportRecord).order_by(ExportRecord.created_at.desc()).limit(100)
    if job_id:
        q = q.where(ExportRecord.job_id == job_id)
    return [_export_json(r) for r in db.scalars(q)]

