import shutil
import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.common import default_project, err, get_or_404
from app.config import get_settings
from app.db import get_db
from app.ml import trainer
from app.models import DatasetVersion, DatasetVersionItem, ModelVersion, Review, TrainingMetric, TrainingRun
from app.services.jobs import dispatch_training, transition

router = APIRouter(prefix="/api/v1", tags=["training"])


class TrainRequest(BaseModel):
    dataset_version_id: uuid.UUID
    base_model_id: uuid.UUID
    trigger_token: str = Field(min_length=2, max_length=64, pattern=r"^[A-Za-z0-9_]+$")
    objective: str = Field("character", pattern="^(character|style)$")
    preset: str = "sdxl-8gb-explore"
    overrides: dict = {}


def _resolve(body: TrainRequest) -> dict:
    if body.preset not in trainer.PRESETS:
        raise err(422, "bad_preset", f"Preset hợp lệ: {', '.join(trainer.PRESETS)}.")
    cfg = {**trainer.PRESETS[body.preset], "preset": body.preset}
    allowed = set(trainer.VALUE_FLAGS) | trainer.BOOL_FLAGS
    for k, v in body.overrides.items():
        if k not in allowed:
            raise err(422, "bad_override", f"Không cho phép ghi đè '{k}'.")
        cfg[k] = v
    if not 1 <= int(cfg["max_train_steps"]) <= 20000:
        raise err(422, "bad_steps", "max_train_steps phải trong 1..20000.")
    if int(cfg["rank"]) not in range(1, 257):
        raise err(422, "bad_rank", "rank phải trong 1..256.")
    return cfg


def _preflight(db: Session, body: TrainRequest) -> dict:
    cfg = _resolve(body)
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    ver = db.get(DatasetVersion, body.dataset_version_id)
    add("dataset_version", ver is not None, "" if ver else "Không tìm thấy dataset version.")
    if ver:
        n_train = db.query(DatasetVersionItem).filter_by(dataset_version_id=ver.id, split="train").count()
        add("train_images", n_train >= 5, f"{n_train} ảnh train (khuyến nghị ≥15; 30–60 là khởi điểm thực nghiệm).")
    base = db.get(ModelVersion, body.base_model_id)
    add("base_model", bool(base and base.kind == "base" and base.status != "unavailable"),
        "" if base else "Không tìm thấy model nền.")
    if base:
        add("base_family_trainable", bool(base.capabilities.get("trainable_lora")), f"family={base.family}")
    cuda_ok, vram, msg = False, None, "torch chưa cài trong process API; kiểm tra thực hiện ở worker GPU."
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            vram = torch.cuda.get_device_properties(0).total_memory / 2**30
            msg = f"{torch.cuda.get_device_name(0)} {vram:.1f} GiB"
        else:
            msg = "CUDA không khả dụng."
    except ImportError:
        pass
    add("cuda", cuda_ok, msg)
    if vram is not None and cfg["resolution"] >= 1024 and vram < 20:
        add("vram_for_resolution", False, f"{cfg['resolution']}px SDXL thường cần >~20 GiB; có {vram:.1f} GiB. Dùng preset 8gb.")
    free_gb = shutil.disk_usage(get_settings().data_dir if _exists(get_settings().data_dir) else ".").free / 2**30
    add("disk", free_gb > 20, f"{free_gb:.0f} GiB trống tại data_dir.")
    missing = trainer.check_script_flags(cfg)
    add("trainer_flags", not missing, "OK" if not missing else f"Thiếu/không khớp: {missing}")
    return {"ok": all(c["ok"] for c in checks), "checks": checks, "resolved_config": cfg}


def _exists(p: str) -> bool:
    import os
    return os.path.isdir(p)


def _run_out(r: TrainingRun) -> dict:
    return {"id": r.id, "state": r.state, "objective": r.objective, "trigger_token": r.trigger_token,
            "current_step": r.current_step, "total_steps": r.total_steps, "error_code": r.error_code,
            "error_message": r.error_message, "config": r.config, "dataset_version_id": r.dataset_version_id,
            "base_model_id": r.base_model_id, "output_model_version_id": r.output_model_version_id,
            "trainer_revision": r.trainer_revision, "created_at": r.created_at}


@router.post("/training-runs/preflight")
def preflight(body: TrainRequest, db: Session = Depends(get_db)) -> dict:
    return _preflight(db, body)


@router.post("/training-runs", status_code=202)
def create_run(body: TrainRequest, db: Session = Depends(get_db)) -> dict:
    pf = _preflight(db, body)
    failed = [c for c in pf["checks"] if not c["ok"] and c["name"] not in {"cuda", "trainer_flags"}]
    # cuda/trainer_flags are evaluated by the GPU worker image; API container may lack both.
    if failed:
        raise err(422, "preflight_failed", "; ".join(f"{c['name']}: {c['detail']}" for c in failed))
    p = default_project(db)
    run = TrainingRun(project_id=p.id, dataset_version_id=body.dataset_version_id, base_model_id=body.base_model_id,
                      objective=body.objective, trigger_token=body.trigger_token, config=pf["resolved_config"],
                      total_steps=pf["resolved_config"]["max_train_steps"])
    db.add(run)
    db.commit()
    try:
        dispatch_training(db, run.id)
    except Exception:
        db.rollback()
    db.refresh(run)
    return _run_out(run)


@router.get("/training-runs")
def list_runs(db: Session = Depends(get_db)) -> list[dict]:
    return [_run_out(r) for r in db.query(TrainingRun).order_by(TrainingRun.created_at.desc()).limit(100)]


@router.get("/training-runs/{run_id}")
def get_run(run_id: uuid.UUID, db: Session = Depends(get_db)) -> dict:
    return _run_out(get_or_404(db, TrainingRun, run_id, "training run"))


@router.get("/training-runs/{run_id}/metrics")
def metrics(run_id: uuid.UUID, db: Session = Depends(get_db)) -> list[dict]:
    get_or_404(db, TrainingRun, run_id, "training run")
    rows = db.query(TrainingMetric).filter_by(run_id=run_id).order_by(TrainingMetric.step).limit(5000)
    return [{"step": m.step, "loss": m.loss, "lr": m.lr, "recorded_at": m.recorded_at} for m in rows]


@router.get("/training-runs/{run_id}/log")
def log_tail(run_id: uuid.UUID, lines: int = 200, db: Session = Depends(get_db)) -> dict:
    get_or_404(db, TrainingRun, run_id, "training run")
    p = trainer.run_dir(run_id) / "train.log"
    if not p.exists():
        return {"lines": []}
    return {"lines": p.read_text(encoding="utf-8", errors="replace").splitlines()[-min(lines, 1000):]}


@router.post("/training-runs/{run_id}/cancel")
def cancel_run(run_id: uuid.UUID, db: Session = Depends(get_db)) -> dict:
    run = get_or_404(db, TrainingRun, run_id, "training run")
    if run.state in {"awaiting_review", "approved", "rejected", "failed", "cancelled"}:
        raise err(409, "terminal_state", f"Run đã ở trạng thái {run.state}.")
    run.cancel_requested = True
    db.commit()
    transition(db, TrainingRun, run_id, {"queued"}, "cancelled")
    db.refresh(run)
    return _run_out(run)


@router.post("/training-runs/{run_id}/resume")
def resume_run(run_id: uuid.UUID, db: Session = Depends(get_db)) -> dict:
    """Not implemented: the pinned script's checkpoint resume has not been verified end to end."""
    get_or_404(db, TrainingRun, run_id, "training run")
    raise err(501, "not_implemented", "Resume chưa được kiểm chứng; hãy tạo run mới.")


class ModelReview(BaseModel):
    decision: str = Field(pattern="^(approved|rejected)$")
    reason: str | None = None


@router.post("/model-versions/{mv_id}/review")
def review_model(mv_id: uuid.UUID, body: ModelReview, db: Session = Depends(get_db)) -> dict:
    mv = get_or_404(db, ModelVersion, mv_id, "model version")
    if mv.review_status != "awaiting_review":
        raise err(409, "bad_state", "Model không ở trạng thái awaiting_review.")
    mv.review_status = body.decision
    if mv.training_run_id:
        transition(db, TrainingRun, mv.training_run_id, {"awaiting_review"}, body.decision)
    db.add(Review(subject_type="model_version", subject_id=mv.id, decision=body.decision, reason=body.reason))
    db.commit()
    return {"id": mv.id, "review_status": mv.review_status}
