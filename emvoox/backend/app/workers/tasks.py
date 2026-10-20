import hashlib
import json
import logging
import os
import signal
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import SessionLocal
from app.ml import trainer
from app.ml.gpu_lock import GpuBusy, GpuLease
from app.ml.providers import ASPECT_SIZES, Cancelled, GenRequest, ProviderUnavailable, get_provider
from app.models import GenerationJob, GenerationOutput, JobAttempt, ModelVersion, TrainingMetric, TrainingRun
from app.services.assets import store_asset
from app.services.jobs import dispatch_job, dispatch_training, find_stale, heartbeat, transition
from app.workers.celery_app import celery_app

log = logging.getLogger(__name__)


# ---------- generation ----------

def _fail(db: Session, job_id, code: str, msg: str) -> None:
    db.rollback()
    db.execute(update(GenerationJob).where(GenerationJob.id == job_id, GenerationJob.state.notin_(
        ["approved", "rejected", "cancelled"])).values(state="failed", error_code=code, error_message=msg[:2000]))
    db.commit()


def execute_generation(job_id: uuid.UUID, db: Session | None = None) -> str:
    """Idempotent body of a generation job. Returns the final state."""
    own = db is None
    db = db or SessionLocal()
    s = get_settings()
    try:
        job = db.get(GenerationJob, job_id)
        if job is None:
            return "missing"
        if job.state in {"review_required", "approved", "rejected", "failed", "cancelled"}:
            return job.state  # redelivered task: nothing to do
        if job.cancel_requested:
            transition(db, GenerationJob, job_id, {"queued", "running"}, "cancelled")
            return "cancelled"
        # running -> running allowed so a redelivered task after a crash can resume
        if not transition(db, GenerationJob, job_id, {"queued", "running"}, "running"):
            return "lost_race"
        db.add(JobAttempt(job_id=job_id, attempt=(db.query(JobAttempt).filter_by(job_id=job_id).count() + 1),
                          worker=os.uname().nodename if hasattr(os, "uname") else "worker"))
        db.commit()
        attempt = db.query(JobAttempt).filter_by(job_id=job_id).order_by(JobAttempt.attempt.desc()).first()

        cfg = job.config
        lora_path = None
        if job.lora_version_id:
            lora = db.get(ModelVersion, job.lora_version_id)
            lora_path = Path(s.data_dir) / lora.artifact_key
            if not lora_path.exists():
                raise FileNotFoundError(f"LoRA artifact missing: {lora_path}")
        req = GenRequest(
            prompt=job.prompt_compiled, negative_prompt=job.negative_prompt, width=cfg["width"], height=cfg["height"],
            steps=cfg["steps"], guidance_scale=cfg["guidance_scale"], seed=cfg["seed"], num_images=cfg["num_images"],
            lora_path=lora_path, lora_scale=cfg.get("lora_scale", 1.0),
        )
        last_hb = 0.0

        def on_step(step: int, total: int) -> None:
            nonlocal last_hb
            now = time.monotonic()
            if now - last_hb < 2 and step < total:
                return
            last_hb = now
            heartbeat(db, GenerationJob, job_id, progress_step=step, progress_total=total)
            db.refresh(job)
            if job.cancel_requested:
                raise Cancelled()

        with GpuLease(f"gen:{job_id}"):
            images = get_provider().generate(req, on_step)

        for idx, img in enumerate(images):
            asset = store_asset(db, project_id=job.project_id, data=img.data, kind="generated", provenance={
                "ai_generated": True, "provider": get_provider().name, "job_id": str(job_id),
                "model_version_id": str(job.model_version_id), "lora_version_id": str(job.lora_version_id) if job.lora_version_id else None,
                "seed": img.seed, "prompt": job.prompt_compiled, **img.meta,
            })
            stmt = pg_insert(GenerationOutput).values(
                id=uuid.uuid4(), job_id=job_id, asset_id=asset.id, output_index=idx, seed=img.seed,
            ).on_conflict_do_nothing(constraint="uq_output_index")
            if db.execute(stmt).rowcount == 0:  # redelivery already stored this output; drop the duplicate asset
                db.delete(asset)
            db.commit()
        final = "review_required"
        transition(db, GenerationJob, job_id, {"running"}, final, error_code=None, error_message=None)
        attempt.finished_at, attempt.outcome = datetime.now(timezone.utc), "ok"
        db.commit()
        return final
    except Cancelled:
        transition(db, GenerationJob, job_id, {"running"}, "cancelled")
        return "cancelled"
    except GpuBusy as e:
        db.rollback()
        db.execute(update(GenerationJob).where(GenerationJob.id == job_id).values(state="queued", enqueued_at=None))
        db.commit()
        log.warning("GPU busy for %s: %s", job_id, e)
        return "requeued"
    except ProviderUnavailable as e:
        _fail(db, job_id, "provider_unavailable", str(e))
        return "failed"
    except Exception as e:  # noqa: BLE001
        code = "cuda_oom" if "out of memory" in str(e).lower() else "internal_error"
        _fail(db, job_id, code, f"{type(e).__name__}: {e}")
        return "failed"
    finally:
        if own:
            db.close()


@celery_app.task(name="app.workers.tasks.run_generation")
def run_generation(job_id: str) -> str:
    return execute_generation(uuid.UUID(job_id))


# ---------- training ----------

def _train_fail(db: Session, run_id, code: str, msg: str) -> None:
    db.rollback()
    db.execute(update(TrainingRun).where(TrainingRun.id == run_id, TrainingRun.state.notin_(
        ["approved", "rejected", "cancelled"])).values(state="failed", error_code=code, error_message=msg[-4000:]))
    db.commit()


def execute_training(run_id: uuid.UUID) -> str:
    db = SessionLocal()
    s = get_settings()
    proc: subprocess.Popen | None = None
    work: Path | None = None
    try:
        run = db.get(TrainingRun, run_id)
        if run is None or run.state not in {"queued", "preparing", "running"}:
            return run.state if run else "missing"
        if not transition(db, TrainingRun, run_id, {"queued"}, "preparing", trainer_revision=s.trainer_revision):
            return "lost_race"
        from app.models import DatasetVersion  # local import keeps module graph light

        base = db.get(ModelVersion, run.base_model_id)
        cfg = run.config
        work = trainer.run_dir(run_id)
        n = trainer.materialize_dataset(db, run.dataset_version_id, work / "dataset")
        if n == 0:
            raise ValueError("Dataset version không có ảnh train.")
        out_dir = work / "output"
        argv = trainer.build_argv(cfg, work / "dataset", out_dir, run.trigger_token, base.hf_id, base.revision)
        with GpuLease(f"train:{run_id}"):
            transition(db, TrainingRun, run_id, {"preparing"}, "running", total_steps=cfg["max_train_steps"])
            heartbeat(db, TrainingRun, run_id)  # visible as "worker online" before the first progress line
            log_path = work / "train.log"
            with open(log_path, "w", encoding="utf-8") as logf:
                proc = subprocess.Popen(  # argv list, never shell=True
                    argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                    env={**os.environ, "HF_TOKEN": s.hf_token or "", "PYTHONUNBUFFERED": "1"},
                    start_new_session=True,
                )
                last_db = 0.0
                for raw in proc.stdout:
                    line = raw.replace("\r", "\n")
                    logf.write(line.replace(s.hf_token, "***") if s.hf_token else line)
                    prog = trainer.parse_progress(line)
                    now = time.monotonic()
                    if prog and now - last_db > 3:
                        step, total, loss, lr = prog
                        db.add(TrainingMetric(run_id=run_id, step=step, loss=loss, lr=lr))
                        heartbeat(db, TrainingRun, run_id, current_step=step, total_steps=total)
                        db.refresh(run)
                        last_db = now
                        if run.cancel_requested:
                            os.killpg(proc.pid, signal.SIGTERM)
                            proc.wait(timeout=60)
                            transition(db, TrainingRun, run_id, {"running"}, "cancelled")
                            return "cancelled"
                code = proc.wait()
            if code != 0:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
                err = "cuda_oom" if "out of memory" in tail.lower() else "trainer_failed"
                _train_fail(db, run_id, err, tail)
                return "failed"

        weights = out_dir / "pytorch_lora_weights.safetensors"
        if not weights.exists():
            _train_fail(db, run_id, "no_artifact", f"Không thấy {weights.name} trong output.")
            return "failed"
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        mv = ModelVersion(
            kind="character_lora" if run.objective == "character" else "style_lora",
            name=f"lora-{str(run_id)[:8]}", family=base.family, parent_base_id=base.id,
            trigger_token=run.trigger_token, artifact_key=str(weights.relative_to(s.data_dir)),
            artifact_sha256=digest, training_run_id=run_id, review_status="awaiting_review", status="experimental",
            capabilities={"lora_scale": [0.0, 2.0]}, metrics={"final_step": run.current_step},
        )
        db.add(mv)
        db.flush()
        db.execute(update(TrainingRun).where(TrainingRun.id == run_id).values(output_model_version_id=mv.id))
        db.commit()
        transition(db, TrainingRun, run_id, {"running"}, "validating")
        _validation_samples(db, run, mv)
        transition(db, TrainingRun, run_id, {"validating"}, "awaiting_review")
        return "awaiting_review"
    except GpuBusy:
        db.rollback()
        db.execute(update(TrainingRun).where(TrainingRun.id == run_id).values(state="queued", enqueued_at=None))
        db.commit()
        return "requeued"
    except Exception as e:  # noqa: BLE001
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        _train_fail(db, run_id, "internal_error", f"{type(e).__name__}: {e}")
        return "failed"
    finally:
        if work is not None:
            trainer.cleanup_materialized(work / "dataset")
        db.close()


def _validation_samples(db: Session, run: TrainingRun, mv: ModelVersion) -> None:
    """Reload the produced LoRA and generate prompts that were not training captions."""
    prompts = [f"{run.trigger_token}, portrait, neutral background", f"{run.trigger_token}, full body, walking in a park"]
    # a GPU lease is not held here; training lease was released above, so execute_generation takes its own
    for i, p in enumerate(prompts):
        w, h = ASPECT_SIZES["1:1"] if run.config.get("resolution", 1024) >= 1024 else (768, 768)
        job = GenerationJob(
            project_id=run.project_id, purpose="validation", model_version_id=run.base_model_id,
            lora_version_id=mv.id, training_run_id=run.id, request_hash=f"val:{run.id}:{i}",
            idempotency_key=f"val:{run.id}:{i}", prompt_original=p, prompt_compiled=p,
            config={"width": w, "height": h, "steps": 25, "guidance_scale": 6.0, "seed": 1000 + i, "num_images": 1,
                    "lora_scale": 1.0},
        )
        db.add(job)
        db.commit()
        execute_generation(job.id)


@celery_app.task(name="app.workers.tasks.run_training")
def run_training(run_id: str) -> str:
    return execute_training(uuid.UUID(run_id))


@celery_app.task(name="app.workers.tasks.sweep")
def sweep() -> dict:
    """Outbox + liveness recovery: re-enqueue never-enqueued rows, fail silent running rows."""
    db = SessionLocal()
    try:
        gen_ids, run_ids = find_stale(db)
        for jid in gen_ids:
            job = db.get(GenerationJob, jid)
            if job.state == "queued":
                dispatch_job(db, jid)
            else:  # heartbeat went silent -> back to queued once (task is idempotent), then normal path
                db.execute(update(GenerationJob).where(GenerationJob.id == jid).values(state="queued", enqueued_at=None))
                db.commit()
        for rid in run_ids:
            run = db.get(TrainingRun, rid)
            if run.state == "queued":
                dispatch_training(db, rid)
            else:
                _train_fail(db, rid, "worker_lost", "Worker ngừng heartbeat; hãy resume/chạy lại.")
        return {"generation": len(gen_ids), "training": len(run_ids)}
    finally:
        db.close()
