import shutil
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.dataset_rights import rights_status
from app.db.session import SessionLocal, get_db
from app.ml import training as T
from app.models import (
    Dataset, DatasetItem, DatasetVersion, DatasetVersionItem, GenerationJob, ModelVersion, TrainingMetric, TrainingRun,
)
from app.workers.celery_app import celery_app

router = APIRouter(prefix="/api/v1")


class TrainConfig(BaseModel):
    resolution: Literal[512, 768, 1024] = 512
    rank: int = Field(16, ge=4, le=64)
    learning_rate: float = Field(1e-4, ge=1e-6, le=1e-3)
    max_train_steps: int | None = Field(None, ge=1, le=5000)
    num_train_epochs: int | None = Field(None, ge=1, le=200)
    train_batch_size: int = Field(1, ge=1, le=4)
    gradient_accumulation_steps: int = Field(4, ge=1, le=16)
    checkpointing_steps: int = Field(100, ge=5, le=2000)
    mixed_precision: Literal["fp16", "bf16", "no"] = "fp16"
    gradient_checkpointing: bool = True
    use_vae_fp16_fix: bool = True
    seed: int = Field(42, ge=0, lt=2**31)

    @model_validator(mode="after")
    def _one_of(self):
        if bool(self.max_train_steps) == bool(self.num_train_epochs):
            raise ValueError("chỉ định đúng một trong max_train_steps hoặc num_train_epochs")
        return self


class RunRequest(BaseModel):
    dataset_version_id: str
    extra_dataset_version_ids: list[str] = Field(default_factory=list, max_length=16)  # gộp thêm nhóm dataset
    base_model_version_id: str
    trigger_token: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_]+$")
    objective: Literal["character", "style"] = "character"
    config: TrainConfig


class ReviewBody(BaseModel):
    decision: Literal["approved", "rejected"]
    note: str | None = Field(None, max_length=1000)


def _gpu(timeout: int = 20) -> dict:
    """GPU do worker sở hữu: hỏi worker thật; không trả lời = worker offline."""
    try:
        return celery_app.send_task("app.workers.tasks.gpu_info", queue="training").get(timeout=timeout)
    except Exception as e:
        return {"cuda": False, "reason": f"worker không phản hồi ({type(e).__name__})", "offline": True}


def _checks(db: Session, req: RunRequest) -> dict:
    checks: list[dict] = []

    def add(name: str, status: str, msg: str):
        checks.append({"name": name, "status": status, "message": msg})

    dsv = db.get(DatasetVersion, req.dataset_version_id)
    base = db.get(ModelVersion, req.base_model_version_id)
    n_train = 0
    all_ids = list(dict.fromkeys([req.dataset_version_id, *req.extra_dataset_version_ids]))
    missing = [i for i in all_ids if not db.get(DatasetVersion, i)]
    if not dsv or missing:
        add("dataset", "fail", "dataset version không tồn tại")
    else:
        n_train = db.scalar(select(func.count()).select_from(DatasetVersionItem).where(
            DatasetVersionItem.version_id.in_(all_ids), DatasetVersionItem.split == "train")) or 0
        n_all = db.scalar(select(func.count()).select_from(DatasetVersionItem).where(DatasetVersionItem.version_id.in_(all_ids))) or 0
        add("dataset", "ok" if n_train else "fail", f"{n_train} ảnh train ({n_all} tổng) từ {len(all_ids)} dataset version")
        dup = db.scalar(select(func.count()).select_from(
            select(DatasetVersionItem.sha256).where(DatasetVersionItem.version_id.in_(all_ids))
            .group_by(DatasetVersionItem.sha256).having(func.count() > 1).subquery())) or 0
        if dup:
            add("dataset_overlap", "warn", f"{dup} ảnh xuất hiện ở nhiều version được gộp (sẽ bị lặp khi train)")
        ds = db.get(Dataset, dsv.dataset_id)
        if req.objective == "character" and ds.group not in ("identity", "body", "expression", "pose", "outfit", "appearance"):
            add("objective", "warn", f"dataset nhóm '{ds.group}' không phải dữ liệu nhận dạng nhân vật")
        if req.objective == "character" and not ds.character_id:
            add("character", "warn", "dataset chưa gắn nhân vật: LoRA sẽ không liên kết hồ sơ nhân vật")
        # Quyền sử dụng của MỌI dataset được gộp (COMMERCIAL_USE=true => chặn phi thương mại / chưa rõ)
        for did in dict.fromkeys(db.get(DatasetVersion, v).dataset_id for v in all_ids):
            d2 = db.get(Dataset, did)
            n_it = db.scalar(select(func.count()).select_from(DatasetItem).where(DatasetItem.dataset_id == did, DatasetItem.status == "approved")) or 0
            n_syn = db.scalar(select(func.count()).select_from(DatasetItem).where(
                DatasetItem.dataset_id == did, DatasetItem.status == "approved", DatasetItem.synthetic.is_(True))) or 0
            st, why = rights_status(d2.license, d2.source, n_it, n_syn)
            if st == "ok":
                add("rights", "ok", f"{d2.name}: {why}")
            else:
                add("rights", "fail" if settings.commercial_use else "warn",
                    f"{d2.name}: {why}" + (" — COMMERCIAL_USE=true nên bị chặn" if settings.commercial_use else " — chỉ dùng nội bộ/nghiên cứu"))
        if n_train and n_train < 10:
            add("data_size", "warn", f"chỉ {n_train} ảnh train; kết quả thăm dò, dễ overfit/không học đủ")
    if not base or base.kind != "base":
        add("base_model", "fail", "base model không hợp lệ")
    elif not base.trainable:
        add("base_model", "fail", f"'{base.name}' chưa được đánh dấu trainable")
    else:
        rev = T.snapshot_revision(base.model_id)
        add("base_model", "ok" if rev else "warn",
            f"{base.model_id} @ {rev[:10]}" if rev else "chưa có trong cache cục bộ; sẽ tải ~7GB khi bắt đầu")

    cfg = req.config.model_dump()
    steps = T.resolve_steps(n_train, cfg) if n_train else 0
    add("steps", "ok" if steps else "fail", f"{steps} bước tối ưu (kích thước hiệu dụng "
        f"{cfg['train_batch_size'] * cfg['gradient_accumulation_steps']})")

    g = _gpu()
    if not g.get("cuda"):
        add("gpu", "fail", f"không có GPU khả dụng: {g.get('reason')}")
    else:
        need = T.HEURISTIC_MIN_FREE_GB[cfg["resolution"]]
        if g["total_gb"] < 6:
            add("gpu", "fail", f"{g['name']}: {g['total_gb']}GB tổng < 6GB")
        elif g["free_gb"] < need:
            add("gpu", "warn", f"{g['name']}: trống {g['free_gb']}GB / {g['total_gb']}GB; ngưỡng gợi ý "
                f"{need}GB ở {cfg['resolution']}px (heuristic, chưa benchmark). Có thể OOM — thử giảm resolution.")
        else:
            add("gpu", "ok", f"{g['name']}: trống {g['free_gb']}GB / {g['total_gb']}GB (ngưỡng gợi ý {need}GB, heuristic)")

    if not T.TRAINER_SCRIPT.exists():
        add("trainer", "fail", f"thiếu {T.TRAINER_SCRIPT.name}")
    else:
        add("trainer", "ok", T.TRAINER_REVISION)
    Path(settings.dataset_dir).mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(settings.dataset_dir).free / 2**30
    add("disk", "ok" if free_gb >= 15 else "fail", f"trống {free_gb:.0f}GB (cần ≥15GB)")

    busy = db.scalar(select(func.count()).select_from(TrainingRun).where(
        TrainingRun.state.in_(("preparing", "running", "validating")))) or 0
    busy += db.scalar(select(func.count()).select_from(GenerationJob).where(GenerationJob.state == "running")) or 0
    if busy:
        add("gpu_busy", "warn", f"đang có {busy} job nặng; job mới xếp hàng sau (một GPU một job)")

    overall = "fail" if any(c["status"] == "fail" for c in checks) else "warn" if any(c["status"] == "warn" for c in checks) else "ok"
    return {"overall": overall, "checks": checks, "resolved_steps": steps, "n_train": n_train}


@router.post("/training-runs/preflight")
def preflight(req: RunRequest, db: Session = Depends(get_db)):
    return _checks(db, req)


def _run_dict(r: TrainingRun) -> dict:
    return {"id": r.id, "state": r.state, "phase": r.phase, "step": r.step, "max_steps": r.max_steps,
            "trigger_token": r.trigger_token, "objective": r.objective, "config": r.config, "error": r.error,
            "dataset_version_id": r.dataset_version_id, "base_model_version_id": r.base_model_version_id,
            "trainer_revision": r.trainer_revision, "base_revision": r.base_revision, "manifest_hash": r.manifest_hash,
            "output_model_version_id": r.output_model_version_id, "validation_asset_ids": r.validation_asset_ids,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "finished_at": r.finished_at.isoformat() if r.finished_at else None,
            "heartbeat_at": r.heartbeat_at.isoformat() if r.heartbeat_at else None}


@router.post("/training-runs", status_code=202)
def create_run(req: RunRequest, db: Session = Depends(get_db)):
    pf = _checks(db, req)
    if pf["overall"] == "fail":
        raise HTTPException(422, detail={"code": "preflight_failed", "preflight": pf})
    run = TrainingRun(dataset_version_id=req.dataset_version_id, base_model_version_id=req.base_model_version_id,
                      objective=req.objective, trigger_token=req.trigger_token,
                      config={**req.config.model_dump(), "extra_dataset_version_ids": req.extra_dataset_version_ids},
                      max_steps=pf["resolved_steps"], state="queued")
    db.add(run)
    db.flush()
    run.work_dir = str((Path(settings.dataset_dir).resolve() / "runs" / run.id))
    db.commit()
    try:
        celery_app.send_task("app.workers.tasks.train", args=[run.id], queue="training")
    except Exception as e:
        run.state, run.error = "failed", f"enqueue_failed: {e}"[:500]
        db.commit()
        raise HTTPException(503, "không đưa được job vào hàng đợi")
    return {**_run_dict(run), "preflight": pf}


@router.get("/training-runs")
def list_runs(db: Session = Depends(get_db)):
    return [_run_dict(r) for r in db.scalars(select(TrainingRun).order_by(TrainingRun.created_at.desc()).limit(50))]


def _run(db: Session, run_id: str) -> TrainingRun:
    r = db.get(TrainingRun, run_id)
    if not r:
        raise HTTPException(404, "training run không tồn tại")
    return r


@router.get("/training-runs/{run_id}")
def get_run(run_id: str, tail: int = 40, db: Session = Depends(get_db)):
    r = _run(db, run_id)
    log = ""
    if r.work_dir and (Path(r.work_dir) / "train.log").exists():
        lines = (Path(r.work_dir) / "train.log").read_text(encoding="utf-8", errors="replace").splitlines()
        log = "\n".join(lines[-max(1, min(tail, 200)):])
    return {**_run_dict(r), "log_tail": log}


@router.get("/training-runs/{run_id}/metrics")
def metrics(run_id: str, db: Session = Depends(get_db)):
    _run(db, run_id)
    rows = db.scalars(select(TrainingMetric).where(TrainingMetric.run_id == run_id).order_by(TrainingMetric.step))
    return [{"step": m.step, "loss": m.loss, "lr": m.lr, "ts": m.ts.isoformat() if m.ts else None} for m in rows]


@router.post("/training-runs/{run_id}/cancel")
def cancel(run_id: str, db: Session = Depends(get_db)):
    r = _run(db, run_id)
    if r.state in ("awaiting_review", "approved", "rejected", "failed", "cancelled"):
        raise HTTPException(409, f"không thể hủy ở trạng thái {r.state}")
    r.cancel_requested = True
    if r.state == "queued":
        r.state = "cancelled"  # chưa có worker nào nhận
    db.commit()
    return _run_dict(r)


@router.post("/training-runs/{run_id}/resume", status_code=202)
def resume(run_id: str, db: Session = Depends(get_db)):
    r = _run(db, run_id)
    if r.state not in ("failed", "cancelled"):
        raise HTTPException(409, f"chỉ resume được run failed/cancelled (hiện: {r.state})")
    ckpts = list((Path(r.work_dir) / "out").glob("checkpoint-*")) if r.work_dir else []
    if not ckpts:
        raise HTTPException(409, "chưa có checkpoint để resume")
    celery_app.send_task("app.workers.tasks.train", args=[r.id, True], queue="training")
    return {"id": r.id, "resuming_from": sorted(c.name for c in ckpts)[-1]}


@router.post("/training-runs/{run_id}/validate", status_code=202)
def revalidate(run_id: str, db: Session = Depends(get_db)):
    """Chạy lại tạo ảnh validation (vd khi bước này lỗi sau khi train xong)."""
    r = _run(db, run_id)
    if r.state != "awaiting_review" or not r.output_model_version_id:
        raise HTTPException(409, "chỉ chạy lại validation cho run đã train xong và đang chờ duyệt")
    celery_app.send_task("app.workers.tasks.validate_run", args=[r.id], queue="training")
    return {"id": r.id, "state": "validating"}


# ---------------- Registry ----------------
def _mv_dict(m: ModelVersion) -> dict:
    return {"id": m.id, "kind": m.kind, "name": m.name, "family": m.family, "model_id": m.model_id,
            "revision": m.revision, "license": m.license, "status": m.status, "trainable": m.trainable,
            "sha256": m.sha256, "base_model_version_id": m.base_model_version_id, "training_run_id": m.training_run_id,
            "character_id": m.character_id, "trigger_token": m.trigger_token, "metrics": m.metrics,
            "review_note": m.review_note, "created_at": m.created_at.isoformat() if m.created_at else None}


@router.get("/models")
def list_models(kind: str | None = None, db: Session = Depends(get_db)):
    q = select(ModelVersion).order_by(ModelVersion.created_at.desc())
    if kind:
        q = q.where(ModelVersion.kind == kind)
    return [_mv_dict(m) for m in db.scalars(q)]


@router.post("/model-versions/{mv_id}/review")
def review(mv_id: str, body: ReviewBody, db: Session = Depends(get_db)):
    m = db.get(ModelVersion, mv_id)
    if not m or m.kind == "base":
        raise HTTPException(404, "model version không tồn tại")
    if m.status not in ("awaiting_review", "approved", "rejected"):
        raise HTTPException(409, f"không duyệt được ở trạng thái {m.status}")
    m.status, m.review_note = body.decision, body.note
    run = db.get(TrainingRun, m.training_run_id) if m.training_run_id else None
    if run and run.state in ("awaiting_review", "approved", "rejected"):
        run.state = body.decision
    db.commit()
    return _mv_dict(m)


def seed_ip_adapter() -> None:
    """Adapter tham chiếu diện mạo (không phải model nền/LoRA). Weights tải từ HF khi dùng lần đầu."""
    from app.ml import sdxl
    with SessionLocal() as db:
        ex = db.scalar(select(ModelVersion).where(ModelVersion.kind == "ip_adapter"))
        if ex:
            if not ex.revision:  # weights tải sau lần seed đầu -> cập nhật revision thật
                ex.revision = T.snapshot_revision(sdxl.IP_ADAPTER_REPO)
                db.commit()
            return
        db.add(ModelVersion(
            kind="ip_adapter", name="IP-Adapter Plus Face (SDXL, ViT-H)", family="sdxl", model_id=sdxl.IP_ADAPTER_REPO,
            revision=T.snapshot_revision(sdxl.IP_ADAPTER_REPO), status="experimental", trainable=False,
            license="Apache-2.0 theo model card h94/IP-Adapter (xác minh 17/10/2026; biến thể FaceID có license khác)",
            metrics={"weight": sdxl.IP_ADAPTER_WEIGHT, "encoder": f"{sdxl.IP_ADAPTER_REPO}/{sdxl.IP_ENCODER_SUBFOLDER}",
                     "reference_support": True, "max_references": sdxl.MAX_REFERENCES}))
        db.commit()


def seed_base_model() -> None:
    """Chỉ seed base model thực sự tích hợp (SDXL 1.0). Trạng thái 'experimental' cho training đến khi có run thành công."""
    with SessionLocal() as db:
        if db.scalar(select(ModelVersion).where(ModelVersion.kind == "base", ModelVersion.model_id == settings.sdxl_model_id)):
            return
        db.add(ModelVersion(
            kind="base", name="SDXL 1.0 base", family="sdxl", model_id=settings.sdxl_model_id,
            revision=T.snapshot_revision(settings.sdxl_model_id), status="experimental", trainable=True,
            license="CreativeML Open RAIL++-M (kiểm tra lại model card trước khi phát hành)",
            metrics={"inference": "đã chạy thật trên RTX 4060 8GB", "training": "plumbing chạy thật (30 bước, 512px, 8GB); chất lượng nhận dạng chưa xác minh"}))
        db.commit()
