"""Conditional state transitions, idempotency and outbox-style enqueue for generation/training."""
import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import or_, update
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import GenerationJob, TrainingRun

GEN_TERMINAL = {"approved", "rejected", "failed", "cancelled"}
TRAIN_TERMINAL = {"approved", "rejected", "failed", "cancelled"}


def request_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def transition(db: Session, model, row_id: uuid.UUID, from_states: set[str], to_state: str, **fields) -> bool:
    """Atomic compare-and-set; True if this caller won the transition."""
    res = db.execute(
        update(model).where(model.id == row_id, model.state.in_(from_states)).values(state=to_state, **fields)
    )
    db.commit()
    return res.rowcount == 1


def heartbeat(db: Session, model, row_id: uuid.UUID, **fields) -> None:
    db.execute(update(model).where(model.id == row_id).values(heartbeat_at=datetime.now(timezone.utc), **fields))
    db.commit()


def dispatch_job(db: Session, job_id: uuid.UUID) -> None:
    """Enqueue after commit; enqueued_at stays NULL if the broker is down so the sweeper retries."""
    from app.workers.celery_app import celery_app

    celery_app.send_task("app.workers.tasks.run_generation", args=[str(job_id)], queue="generate")
    db.execute(update(GenerationJob).where(GenerationJob.id == job_id).values(enqueued_at=datetime.now(timezone.utc)))
    db.commit()


def dispatch_training(db: Session, run_id: uuid.UUID) -> None:
    from app.workers.celery_app import celery_app

    celery_app.send_task("app.workers.tasks.run_training", args=[str(run_id)], queue="train")
    db.execute(update(TrainingRun).where(TrainingRun.id == run_id).values(enqueued_at=datetime.now(timezone.utc)))
    db.commit()


def find_stale(db: Session) -> tuple[list[uuid.UUID], list[uuid.UUID]]:
    """Queued-but-never-enqueued rows and running rows whose heartbeat has gone silent."""
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=get_settings().job_stale_seconds)
    out = []
    for model in (GenerationJob, TrainingRun):
        out.append(
            [r.id for r in db.query(model).filter(
                or_(
                    (model.state == "queued") & (model.enqueued_at.is_(None)),
                    (model.state.in_(["running", "preparing", "validating"])) & (model.heartbeat_at < cutoff),
                )
            )]
        )
    return out[0], out[1]
