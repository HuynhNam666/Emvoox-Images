from datetime import datetime, timedelta, timezone

import redis
from fastapi import APIRouter, Depends, Response
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.models import Character, Dataset, GenerationJob, ModelVersion, TrainingRun

router = APIRouter(prefix="/api/v1", tags=["system"])


@router.get("/health/live")
def live() -> dict:
    return {"status": "ok"}


def _gpu_worker_online() -> bool:
    """True if any worker refreshed its presence key in the last ~30 s (works while it is busy)."""
    try:
        r = redis.Redis.from_url(get_settings().redis_url, socket_connect_timeout=1)
        return next(r.scan_iter(match="emvoox:worker:*", count=100), None) is not None
    except Exception:  # noqa: BLE001
        return False


@router.get("/health/ready")
def ready(response: Response, db: Session = Depends(get_db)) -> dict:
    out = {"database": False, "redis": False, "gpu_worker": False}
    try:
        db.execute(text("select 1"))
        out["database"] = True
    except Exception:  # noqa: BLE001
        pass
    try:
        redis.Redis.from_url(get_settings().redis_url, socket_connect_timeout=1).ping()
        out["redis"] = True
    except Exception:  # noqa: BLE001
        pass
    # Reported separately: the API is ready without a worker. A job heartbeat in the last 60 s also proves a busy worker.
    busy = False
    if out["database"]:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=60)
        busy = any(db.query(m.id).filter(m.heartbeat_at > cutoff).first() for m in (TrainingRun, GenerationJob))
    out["gpu_worker"] = busy or (out["redis"] and _gpu_worker_online())
    if not (out["database"] and out["redis"]):
        response.status_code = 503
    return {**out, "asset_storage_backend": get_settings().asset_storage_backend,
            "image_provider": get_settings().image_provider}


@router.get("/dashboard")
def dashboard(db: Session = Depends(get_db)) -> dict:
    return {
        "characters": db.query(Character).count(),
        "datasets": db.query(Dataset).count(),
        "model_versions": db.query(ModelVersion).filter(ModelVersion.kind != "base").count(),
        "running_jobs": db.query(GenerationJob).filter(GenerationJob.state.in_(["queued", "running"])).count()
        + db.query(TrainingRun).filter(TrainingRun.state.in_(["queued", "preparing", "running", "validating"])).count(),
        "recent_jobs": [
            {"id": j.id, "state": j.state, "created_at": j.created_at, "prompt": j.prompt_original[:80]}
            for j in db.query(GenerationJob).filter_by(purpose="production").order_by(GenerationJob.created_at.desc()).limit(8)
        ],
    }


@router.get("/models")
def list_models(kind: str | None = None, db: Session = Depends(get_db)) -> list[dict]:
    q = db.query(ModelVersion)
    if kind:
        q = q.filter_by(kind=kind)
    return [{
        "id": m.id, "kind": m.kind, "name": m.name, "family": m.family, "hf_id": m.hf_id, "revision": m.revision,
        "license": m.license, "status": m.status, "review_status": m.review_status, "capabilities": m.capabilities,
        "parent_base_id": m.parent_base_id, "trigger_token": m.trigger_token, "artifact_sha256": m.artifact_sha256,
        "training_run_id": m.training_run_id, "created_at": m.created_at,
    } for m in q.order_by(ModelVersion.created_at.desc())]
