import hashlib
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select, update

from app.core.config import settings
from app.ml import comfy_client as C
from app.visual.telemetry import ComfySampler
from app.db.session import SessionLocal
from app.ml import training as T
from app.ml.sdxl import IP_ADAPTER_REPO, IP_ADAPTER_WEIGHT, GenParams, ProviderUnavailable, generate_images, gpu_info
from app.visual.providers.local import release_all as release_pipeline
from app.models import (
    Asset, AssetBlob, CharacterReference, DatasetVersion, DatasetVersionItem, GenerationJob, GenerationOutput, ModelVersion, TrainingRun,
    VideoAttempt, VideoJob,
)
from app.workers.celery_app import celery_app


def _fail(job_id: str, msg: str) -> None:
    with SessionLocal() as db:
        db.execute(update(GenerationJob).where(GenerationJob.id == job_id).values(state="failed", error=msg[:2000]))
        db.commit()


def _save_outputs(db, results, meta: dict) -> list[str]:
    ids = []
    for seed, data, w, h in results:
        asset = Asset(kind="generated", mime_type="image/png", width=w, height=h, byte_size=len(data),
                      sha256=hashlib.sha256(data).hexdigest(), storage_backend=settings.asset_storage_backend,
                      provenance={"source": "ai", "model": settings.sdxl_model_id, "seed": seed, **meta})
        db.add(asset)
        db.flush()
        db.add(AssetBlob(asset_id=asset.id, data=data))
        ids.append(asset.id)
    return ids


@celery_app.task(name="app.workers.tasks.gpu_info")
def gpu_info_task():
    # Preflight cho train: giải phóng pipeline suy luận đang giữ VRAM (worker solo => không có job đang chạy).
    # Đổi lại, lần tạo ảnh kế tiếp phải nạp lại model.
    release_pipeline()
    return gpu_info()


@celery_app.task(name="app.workers.tasks.generate")
def generate(job_id: str):
    # Chuyển trạng thái có điều kiện: chỉ một worker nhận được job queued; giao lại task thì bỏ qua.
    with SessionLocal() as db:
        res = db.execute(
            update(GenerationJob)
            .where(GenerationJob.id == job_id, GenerationJob.state == "queued")
            .values(state="running")
        )
        db.commit()
        if res.rowcount == 0:
            return "skipped"
        job = db.get(GenerationJob, job_id)
        cfg = job.config
        prompt = job.prompt_compiled or job.prompt_original
        lora_path, lora_trigger = None, None
        if cfg.get("lora_version_id"):
            mv = db.get(ModelVersion, cfg["lora_version_id"])
            if not mv or not mv.artifact_path or not Path(mv.artifact_path).exists():
                _fail(job_id, "lora_missing: không tìm thấy file LoRA")
                return "failed"
            if T.sha256_file(Path(mv.artifact_path)) != mv.sha256:
                _fail(job_id, "lora_checksum_mismatch: file LoRA không khớp checksum đã đăng ký")
                return "failed"
            lora_path, lora_trigger = mv.artifact_path, mv.trigger_token
        ref_ids = []
        if cfg.get("use_references"):
            ref_ids = cfg.get("reference_asset_ids") or [
                r.asset_id for r in db.scalars(select(CharacterReference).where(
                    CharacterReference.character_version_id == job.character_version_id))][:4]
            ip_images = [db.get(AssetBlob, i).data for i in ref_ids]
        else:
            ip_images = []
    if lora_trigger and lora_trigger not in prompt:
        prompt = f"{lora_trigger}, {prompt}"

    params = GenParams(
        prompt=prompt,
        negative_prompt=cfg.get("negative_prompt", ""),
        aspect_ratio=cfg.get("aspect_ratio", "9:16"),
        steps=cfg.get("steps", 28),
        guidance_scale=cfg.get("guidance_scale", 6.0),
        seed=cfg.get("seed", 42),
        lora_path=lora_path, lora_scale=cfg.get("lora_scale", 1.0),
        ip_images=ip_images, ip_scale=cfg.get("reference_scale", 0.6),
    )
    try:
        try:
            results = generate_images(params, cfg.get("num_images", 1))
        except Exception as first:
            if "out of memory" not in str(first).lower():
                raise
            # Hết VRAM: nhả toàn bộ pipeline + cache rồi thử lại đúng 1 lần (đường nạp sạch hơn, ít phân mảnh).
            release_pipeline()  # không ném lỗi
            results = generate_images(params, cfg.get("num_images", 1))
    except ProviderUnavailable as e:
        _fail(job_id, f"provider_unavailable: {e}")
        return "failed"
    except Exception as e:  # kể cả CUDA OOM — không retry mù cùng cấu hình
        oom = "out of memory" in str(e).lower()
        kind = "cuda_oom" if oom else "generation_error"
        if oom:
            hint = (" | Gợi ý: dùng khung 768² (1:1-small), tắt IP-Adapter/LoRA nếu không cần, đóng ứng dụng khác đang dùng GPU/RAM. "
                    "Nếu là lỗi CUDA nặng, worker sẽ tự khởi động lại để dọn trạng thái.")
            _fail(job_id, f"{kind}: {str(e)[:300]}{hint}")  # ghi lỗi TRƯỚC mọi bước dọn dẹp
            release_pipeline()
            if "cuda error" in str(e).lower():  # context CUDA có thể đã hỏng: thoát, supervisor/Docker bật lại
                import os
                import threading
                threading.Timer(3, os._exit, [3]).start()
            return "failed"
        _fail(job_id, f"{kind}: {e}")
        return "failed"

    # Thành công chỉ sau khi bytes + metadata đã ghi xong.
    with SessionLocal() as db:
        for idx, (seed, data, w, h) in enumerate(results):
            if db.scalar(select(GenerationOutput).where(
                GenerationOutput.job_id == job_id, GenerationOutput.output_index == idx
            )):
                continue
            meta = {"job_id": job_id, "steps": params.steps, "guidance_scale": params.guidance_scale}
            if lora_path:
                meta.update({"lora_version_id": cfg["lora_version_id"], "lora_scale": params.lora_scale})
            if ip_images:
                meta.update({"ip_adapter": f"{IP_ADAPTER_REPO}/{IP_ADAPTER_WEIGHT}", "reference_asset_ids": ref_ids,
                             "reference_scale": params.ip_scale})
            aid = _save_outputs(db, [(seed, data, w, h)], meta)[0]
            db.add(GenerationOutput(job_id=job_id, asset_id=aid, output_index=idx, seed=seed))
        db.execute(update(GenerationJob).where(GenerationJob.id == job_id).values(state="review_required"))
        db.commit()
    return "review_required"


def _run_state(run_id: str, state: str, **extra) -> None:
    with SessionLocal() as db:
        db.execute(update(TrainingRun).where(TrainingRun.id == run_id).values(state=state, **extra))
        db.commit()


@celery_app.task(name="app.workers.tasks.train")
def train(run_id: str, resume: bool = False):
    allowed_from = ("queued", "failed", "cancelled") if resume else ("queued",)
    with SessionLocal() as db:
        res = db.execute(update(TrainingRun).where(TrainingRun.id == run_id, TrainingRun.state.in_(allowed_from))
                         .values(state="preparing", error=None, cancel_requested=False, started_at=T.now(),
                                 heartbeat_at=T.now(), phase="chuẩn bị dữ liệu"))
        db.commit()
        if res.rowcount == 0:
            return "skipped"
        run = db.get(TrainingRun, run_id)
        base = db.get(ModelVersion, run.base_model_version_id)
        snap = {"config": run.config, "trigger_token": run.trigger_token, "max_steps": run.max_steps,
                "version_ids": [run.dataset_version_id, *run.config.get("extra_dataset_version_ids", [])],
                "base_model_id": base.model_id}
        work = Path(run.work_dir)
        hashes = [db.get(DatasetVersion, v).manifest_hash for v in snap["version_ids"]]
        manifest_hash = hashes[0] if len(hashes) == 1 else hashlib.sha256(" ".join(sorted(hashes)).encode()).hexdigest()

    work.mkdir(parents=True, exist_ok=True)
    try:
        release_pipeline()  # nhường VRAM cho trainer
        n_train, val_prompts = T.materialize(run_id, snap["version_ids"], snap["trigger_token"], work)
        if n_train == 0:
            raise RuntimeError("dataset version không có ảnh split=train")
        _run_state(run_id, "running", phase="training", manifest_hash=manifest_hash,
                   trainer_revision=T.TRAINER_REVISION, base_revision=T.snapshot_revision(snap["base_model_id"]))
        argv = T.build_argv(snap, snap["base_model_id"], work, resume)
        outcome, note = T.run_trainer(run_id, argv, work, T.TRAINER_DIR)
    except Exception as e:
        _run_state(run_id, "failed", error=f"preparing_error: {e}"[:2000], finished_at=T.now())
        return "failed"

    if outcome == "cancelled":
        _run_state(run_id, "cancelled", finished_at=T.now(), phase=None)
        return "cancelled"
    if outcome == "failed":
        _run_state(run_id, "failed", error=note[:2000], finished_at=T.now(), phase=None)
        return "failed"

    _run_state(run_id, "validating", phase="đăng ký LoRA + ảnh validation")
    try:
        mv_id = T.register_lora(run_id, work)
    except Exception as e:
        _run_state(run_id, "failed", error=f"register_error: {e}"[:2000], finished_at=T.now())
        return "failed"

    _validate(run_id, mv_id, val_prompts)
    return "awaiting_review"


def _validate(run_id: str, mv_id: str, val_prompts: list[str]) -> None:
    """Tạo ảnh validation bằng chính LoRA vừa train (load lại thật). Lỗi ở đây không xóa model."""
    try:
        release_pipeline()
        with SessionLocal() as db:
            mv = db.get(ModelVersion, mv_id)
            path, trig = mv.artifact_path, mv.trigger_token
        prompts = [f"{trig}, portrait photo, soft window light"] + list(val_prompts[:1])
        asset_ids = []
        for i, pr in enumerate(prompts):
            res = generate_images(GenParams(prompt=pr, aspect_ratio="1:1-small", steps=20, seed=1000 + i,
                                            lora_path=path), 1)
            with SessionLocal() as db:
                asset_ids += _save_outputs(db, res, {"validation_for_run": run_id, "lora_version_id": mv_id})
                db.commit()
        _run_state(run_id, "awaiting_review", output_model_version_id=mv_id, validation_asset_ids=asset_ids,
                   error=None, finished_at=T.now(), phase=None)
    except Exception as e:
        _run_state(run_id, "awaiting_review", output_model_version_id=mv_id,
                   error=f"validation_failed: {e}"[:500], finished_at=T.now(), phase=None)


@celery_app.task(name="app.workers.tasks.validate_run")
def validate_run(run_id: str):
    with SessionLocal() as db:
        run = db.get(TrainingRun, run_id)
        mv_id = run.output_model_version_id
        val = [r.caption for r in db.scalars(select(DatasetVersionItem).where(
            DatasetVersionItem.version_id == run.dataset_version_id, DatasetVersionItem.split == "val"))]
    _run_state(run_id, "validating", phase="tạo lại ảnh validation")
    _validate(run_id, mv_id, val)
    return "done"


def _video_cost(elapsed: float) -> float:
    return elapsed * (settings.video_cost_per_gpu_second or 1.0)  # 0 = tính theo giây GPU


def _video_fail(job_id: str, attempt_id: str, msg: str, t0: float, metrics: dict | None = None) -> str:
    """Ghi lỗi + chi phí của lượt thất bại (chi phí thất bại vẫn được tính)."""
    elapsed = time.perf_counter() - t0
    cost = _video_cost(elapsed)
    with SessionLocal() as db:
        db.execute(update(VideoAttempt).where(VideoAttempt.id == attempt_id).values(
            state="failed", error=msg[:2000], total_seconds=elapsed, gpu_seconds=elapsed, cost=cost,
            metrics=metrics or {}, finished_at=datetime.now(timezone.utc)))
        job = db.get(VideoJob, job_id)
        job.cost_spent = (job.cost_spent or 0) + cost
        job.state, job.error = "failed", msg[:2000]
        db.commit()
    return "failed"


@celery_app.task(name="app.workers.tasks.generate_video")
def generate_video(job_id: str, attempt_id: str):
    """Ảnh khung đầu đã duyệt -> upload ComfyUI -> /prompt -> chờ /history -> lưu mp4 vào asset_blobs.

    Idempotent: attempt `queued` được nhận một lần; attempt `running` đã có prompt_id (worker chết giữa chừng)
    được nối lại bằng cách đối chiếu /history thay vì gửi trùng prompt."""
    t0 = time.perf_counter()
    with SessionLocal() as db:
        att = db.get(VideoAttempt, attempt_id)
        job = db.get(VideoJob, job_id)
        if not att or not job or att.job_id != job_id:
            return "skipped"
        resumable = att.state == "running" and bool(att.prompt_id)
        if att.state != "queued" and not resumable:
            return "skipped"
        if not resumable and job.budget_limit and job.cost_spent >= job.budget_limit:
            att.state, att.error = "failed", "budget_exhausted"
            job.state, job.error = "needs_human", "budget_exhausted: đã hết ngân sách, chuyển người xử lý"
            db.commit()
            return "needs_human"
        att.state, att.started_at = "running", datetime.now(timezone.utc)
        job.state, job.error = "running", None
        params, prompt_id = dict(att.params), att.prompt_id
        src_asset_id, audio_mode = job.source_keyframe_asset_id, job.audio_mode
        blob = db.get(AssetBlob, src_asset_id)
        src_bytes = blob.data if blob else None
        db.commit()
    if src_bytes is None:
        return _video_fail(job_id, attempt_id, "keyframe_missing: không đọc được ảnh nguồn", t0)

    cs = ComfySampler(settings.comfyui_url)
    try:
        release_pipeline()  # nhường VRAM cho ComfyUI (một GPU, một job nặng mỗi lúc)
        wf = C.load_workflow()
        if wf["version"] != params["workflow_version"]:
            # Workflow đã ghim lúc tạo/sửa job; file trên đĩa đổi giữa chừng thì không chạy ngầm bản khác.
            raise C.ComfyError("workflow_changed", f"workflow đã đổi ({params['workflow_version']} → {wf['version']}); "
                                                   "bấm sửa lại để ghim phiên bản mới")
        if not resumable:
            miss = C.missing_nodes(wf["graph"])
            if miss:
                raise C.ComfyError("comfy_nodes_missing", "ComfyUI thiếu node: " + ", ".join(miss) +
                                   " (cài ComfyUI-PromptRelay, ComfyUI-KJNodes, ComfyUI-VideoHelperSuite rồi khởi động lại)")
            name = C.upload_image(src_bytes, f"emvoox_{attempt_id}.png")
            graph = C.build_prompt(
                wf, image_name=name, width=params["width"], height=params["height"], frame_count=params["frame_count"],
                fps=params["fps"], seed=params["seed"], global_prompt=params["global_prompt"],
                segments=params["motion_segments"], audio_mode=audio_mode, filename_prefix=f"emvoox/{attempt_id}")
            prompt_id = C.queue_prompt(graph, client_id=str(uuid.uuid4()))
            with SessionLocal() as db:  # ghi prompt_id TRƯỚC khi chờ: timeout/restart có gì để đối chiếu
                db.execute(update(VideoAttempt).where(VideoAttempt.id == attempt_id).values(prompt_id=prompt_id))
                db.commit()
        t_wait = time.perf_counter()
        with cs:  # đo VRAM/RAM của ComfyUI trong lúc render (qua /system_stats)
            hist = C.wait_for(prompt_id)
        wait_s = time.perf_counter() - t_wait
        t_dl = time.perf_counter()
        f = C.find_output_file(hist, wf["mapping"]["output_node"])
        data = C.fetch_file(f, settings.video_max_upload_bytes)
        download_s = time.perf_counter() - t_dl
    except C.ComfyError as e:
        return _video_fail(job_id, attempt_id, f"{e.code}: {e}", t0, cs.summary() if cs.t0 else None)
    except Exception as e:
        return _video_fail(job_id, attempt_id, f"video_error: {e}", t0, cs.summary() if cs.t0 else None)
    finally:
        C.free_memory()

    elapsed = time.perf_counter() - t0
    cost = _video_cost(elapsed)
    with SessionLocal() as db:
        job = db.get(VideoJob, job_id)
        src = db.get(Asset, src_asset_id)
        asset = Asset(
            kind="generated_video", mime_type=C.mime_for(f["filename"]), width=params["width"], height=params["height"],
            byte_size=len(data), sha256=hashlib.sha256(data).hexdigest(), storage_backend=settings.asset_storage_backend,
            provenance={
                "source": "ai", "kind": "video", "model": job.model_version, "workflow_version": job.workflow_version,
                "workflow_verified": wf["verified"], "source_keyframe_asset_id": src_asset_id,
                "source_generation_job_id": job.source_generation_job_id, "video_job_id": job_id, "attempt_id": attempt_id,
                "character_version_id": job.character_version_id, "scene_version": job.scene_version,
                "seed": params["seed"], "fps": params["fps"], "frame_count": params["frame_count"],
                "audio_mode": audio_mode, "prompt_id": prompt_id,
                "keyframe_sha256": src.sha256 if src else None,
            })
        db.add(asset)
        db.flush()
        db.add(AssetBlob(asset_id=asset.id, data=data))
        db.execute(update(VideoAttempt).where(VideoAttempt.id == attempt_id).values(
            state="rendered", output_asset_id=asset.id, error=None, total_seconds=elapsed, gpu_seconds=elapsed, cost=cost,
            metrics={**cs.summary(), "wait_s": round(wait_s, 2), "download_s": round(download_s, 2),
                     "frames": params["frame_count"], "fps": params["fps"], "size": f"{params['width']}x{params['height']}",
                     "steps": 8, "batch": 1, "output_bytes": len(data)},
            finished_at=datetime.now(timezone.utc)))
        job.cost_spent = (job.cost_spent or 0) + cost
        job.state, job.error = "review_required", None  # output_asset_id chỉ đặt khi người duyệt chấp nhận
        db.commit()
    return "review_required"


def report_worker_status() -> dict:
    """Worker ghi năng lực thật (CUDA, VRAM...) vào DB để API/Model Router biết model tự chạy nào khả dụng."""
    import platform
    import shutil
    from app.models import SystemStatus
    info = gpu_info()
    info.update(host=platform.node(), at=datetime.now(timezone.utc).isoformat(),
                models_free_gb=round(shutil.disk_usage(settings.model_cache_dir).free / 2**30, 1) if Path(settings.model_cache_dir).exists() else None)
    try:
        import torch
        info["torch"] = torch.__version__
    except ImportError:
        pass
    from app.visual import qc as QC
    info["face_ok"] = QC.face_engine() is not None      # Quality Gate đo nhận dạng được hay không
    info["face_error"] = QC.engine_error()
    with SessionLocal() as db:
        row = db.get(SystemStatus, "worker")
        if row is None:
            db.add(SystemStatus(key="worker", value=info))
        else:
            row.value = info
        db.commit()
    return info


@celery_app.task(name="app.workers.tasks.visual_run")
def visual_run(job_id: str):
    """Render Worker cho VisualJob: tạo ảnh → QC → sửa/dự phòng. Xem app/visual/pipeline.py."""
    from app.visual.pipeline import run_visual_job
    try:
        report_worker_status()
    except Exception:  # noqa: BLE001 - báo cáo trạng thái hỏng không được chặn job
        pass
    return run_visual_job(job_id)
