"""Render Worker + Quality Gate + vòng sửa (báo cáo mục 3.3–3.5).

Luồng một VisualJob (task Celery `visual_run`):
  attempt queued → [kiểm ngân sách] → biên dịch prompt theo model → provider.generate (đo VRAM/RAM/thời gian)
  → lưu ảnh ứng viên (một giao dịch) → QC từng ảnh → quyết định:
      có ảnh đạt            → review_required (người duyệt chọn)
      chỉ ảnh cần xem       → review_required (QC tự động không đủ kết luận)
      tất cả lỗi            → sửa tự động (tối đa `visual_max_repairs`, trong ngân sách) hoặc needs_human
      lỗi nhà cung cấp      → thử tuyến dự phòng kế tiếp (không tính vào lượt sửa); hết tuyến → failed
Lỗi không bao giờ bị che bằng ảnh giả: provider không khả dụng => attempt failed có mã lỗi rõ.
Chống tính phí trùng: remote_id (BFL) được lưu trước khi chờ; task giao lại sẽ nối vào job cũ thay vì gửi lại; API đồng bộ bị
ngắt giữa chừng được coi là "có thể đã tính phí" và chuyển sang tuyến KHÁC thay vì gọi lại cùng nhà cung cấp.
"""
import hashlib
import json
import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal
from app.models import (
    Asset, AssetBlob, ModelVersion, ProviderModel, VisualAttempt, VisualCandidate, VisualJob,
)
from app.visual import qc as QC
from app.visual.compiler import Context, compile_prompt, ref_plan
from app.visual.providers import ProviderError, get_provider
from app.visual.providers.base import Hooks, ImageRequest, Ref
from app.visual.providers.local import size_for
from app.visual.registry import estimate_api_cost
from app.visual.service import MAX_REFS
from app.visual.telemetry import ResourceSampler

log = logging.getLogger("visual.pipeline")
ASPECTS = {"9:16": (9, 16), "16:9": (16, 9), "1:1": (1, 1)}
FAMILY_SIZE = {"sdxl": "sdxl", "flux": "flux", "qwen": "qwen", "z-image": "z-image"}
VERDICT_RANK = {"pass": 0, "review": 1, "fail": 2, "pending": 3}


def now():
    return datetime.now(timezone.utc)


def _utc_naive(dt: datetime) -> datetime:
    """SQLite trả datetime naive (UTC), Postgres trả aware: chuẩn hoá để trừ nhau."""
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


def _gpu_rate() -> float:
    return settings.gpu_cost_per_hour / 3600.0


# ------------------------------------------------------------------ dựng yêu cầu cho một model
def build_request(db: Session, job: VisualJob, m: ProviderModel, instruction: str, seed_shift: int) -> tuple[ImageRequest, list[str], dict]:
    ctx = Context.from_dict(job.scene_spec["_resolved"])
    caps = m.capabilities or {}
    use_refs = bool(caps.get("ref_identity") or caps.get("multi_ref"))
    refs_meta = ref_plan(ctx, MAX_REFS.get(m.key, 4)) if use_refs else []
    refs = []
    for r in refs_meta:
        blob = db.get(AssetBlob, r["asset_id"])
        if blob is None:
            raise ProviderError("error", f"ảnh tham chiếu {r['asset_id']} không còn trong kho")
        refs.append(Ref(blob.data, r["role"], r["asset_id"], r.get("label", "")))
    prompt, neg = compile_prompt(ctx, caps, refs_meta, instruction)
    size_family = FAMILY_SIZE.get(m.family, "flux")
    w, h = size_for(size_family, job.aspect_ratio)
    req = ImageRequest(prompt=prompt, negative_prompt=neg, width=w, height=h, aspect_ratio=job.aspect_ratio,
                       seed=job.seed + seed_shift, n=job.num_variants, refs=refs)
    extra = {"ref_asset_ids": [r["asset_id"] for r in refs_meta], "n": job.num_variants, "width": w, "height": h}
    # Character LoRA: chỉ cho đúng họ SDXL đã huấn luyện, và chỉ khi đã duyệt + checksum khớp (mục 3.5: không dùng chéo model)
    if m.key == "sdxl-1.0" and len(ctx.characters) == 1:
        from app.models import Character
        ch = db.scalar(select(Character).where(Character.code == ctx.characters[0]["code"]))
        cid = db.scalar(select(ModelVersion).where(ModelVersion.kind == "character_lora", ModelVersion.status == "approved",
                                                   ModelVersion.character_id == ch.id).order_by(ModelVersion.created_at.desc())) if ch else None
        if cid and cid.artifact_path and Path(cid.artifact_path).exists():
            from app.ml.training import sha256_file
            if sha256_file(Path(cid.artifact_path)) == cid.sha256:
                req.lora_path, req.lora_scale = cid.artifact_path, 0.8
                if cid.trigger_token and cid.trigger_token not in req.prompt:
                    req.prompt = f"{cid.trigger_token}, {req.prompt}"
                extra["lora_version_id"] = cid.id
    return req, [r["asset_id"] for r in refs_meta], extra


def _estimate(db: Session, m: ProviderModel, req: ImageRequest) -> float:
    if m.route == "api":
        return estimate_api_cost(m, req.width, req.height, req.n, len(req.refs))
    rows = db.scalars(select(VisualAttempt).where(VisualAttempt.model_key == m.key, VisualAttempt.state == "done").limit(30)).all()
    per = [a.cost_gpu / max(1, a.params.get("n", 1)) for a in rows if a.cost_gpu > 0]
    return statistics.median(per) * req.n if per else 0.0


# ------------------------------------------------------------------ một attempt
def _fail_attempt(att_id: str, job_id: str, err: ProviderError | str, est_cost: float, sampler: dict | None) -> None:
    code, msg = (err.code, str(err)) if isinstance(err, ProviderError) else ("error", str(err))
    uncertain = isinstance(err, ProviderError) and err.uncertain_billing
    with SessionLocal() as db:
        att = db.get(VisualAttempt, att_id)
        job = db.get(VisualJob, job_id)
        cost_api = est_cost if (uncertain and att.route == "api") else 0.0
        att.state, att.error, att.finished_at = "failed", f"{code}: {msg}"[:2000], now()
        att.cost_api = cost_api
        m = sampler or {}
        att.metrics = {**(att.metrics or {}), **m, "uncertain_billing": uncertain}
        if att.route == "local" and m.get("wall_s"):
            att.cost_gpu = m["wall_s"] * _gpu_rate()
        job.cost_spent = (job.cost_spent or 0) + att.cost_api + att.cost_gpu
        db.commit()


def _execute(job_id: str, att_id: str) -> str:
    """Trả: 'ok' | 'provider_failed' | 'budget'."""
    with SessionLocal() as db:
        job, att = db.get(VisualJob, job_id), db.get(VisualAttempt, att_id)
        m = db.get(ProviderModel, att.model_key)
        resume = json.loads(att.remote_id) if att.remote_id else []
        first = att.started_at is None
        if att.state == "running" and not resume and att.route == "api" and m.key not in ("flux2-pro", "flux2-klein-4b-api", "flux3-image"):
            # API đồng bộ bị ngắt giữa chừng: có thể đã bị tính phí; KHÔNG gọi lại cùng nhà cung cấp.
            est = estimate_api_cost(m, *size_for("flux", job.aspect_ratio), job.num_variants)
            db.commit()
            _fail_attempt(att_id, job_id, ProviderError("interrupted", "tiến trình bị ngắt khi đang gọi API đồng bộ; có thể đã tính phí",
                                                       uncertain_billing=True), est, None)
            return "provider_failed"
        instruction = att.instruction or ""
        try:
            req, ref_ids, extra = build_request(db, job, m, instruction, seed_shift=(att.attempt_no - 1) * 1000)
        except ProviderError as e:
            att.state = "running"
            db.commit()
            _fail_attempt(att_id, job_id, e, 0.0, None)
            return "provider_failed"
        est = _estimate(db, m, req)
        if job.budget_limit and job.cost_spent + est > job.budget_limit:
            att.state, att.error, att.finished_at = "failed", f"budget_exhausted: đã tiêu {job.cost_spent:.4f} + ước tính {est:.4f} > {job.budget_limit}", now()
            db.commit()
            return "budget"
        att.state, att.prompt = "running", req.prompt
        att.started_at = att.started_at or now()
        if first:
            att.metrics = {**(att.metrics or {}), "queue_s": round((_utc_naive(att.started_at) - _utc_naive(job.created_at)).total_seconds(), 3)
                           if job.created_at and att.attempt_no == 1 else 0.0}
        att.params = {**(att.params or {}), **extra, "steps": req.steps, "prompt_chars": len(req.prompt)}
        job.state = "running"
        db.commit()
        queue_s = (att.metrics or {}).get("queue_s")
        route = att.route

    prov = get_provider(m)

    def persist_remote(ids: list[str]) -> None:
        with SessionLocal() as d2:
            d2.execute(update(VisualAttempt).where(VisualAttempt.id == att_id).values(remote_id=json.dumps(ids)))
            d2.commit()

    sampler = ResourceSampler()
    try:
        ok, why = prov.available()
        if not ok:
            raise ProviderError("unavailable", why)
        with sampler:
            res = prov.generate(req, Hooks(on_remote_id=persist_remote, resume_ids=resume))
    except ProviderError as e:
        _fail_attempt(att_id, job_id, e, est, sampler.summary() if sampler.t0 else None)
        return "provider_failed"
    except Exception as e:  # noqa: BLE001 - lỗi bất ngờ vẫn phải ghi rõ, không treo job
        log.exception("attempt %s lỗi bất ngờ", att_id)
        _fail_attempt(att_id, job_id, ProviderError("error", f"{type(e).__name__}: {e}"), est, sampler.summary() if sampler.t0 else None)
        return "provider_failed"

    summary = sampler.summary()
    metrics = {**summary, **res.meta, "queue_s": queue_s, "per_image_s": round(summary["wall_s"] / max(1, len(res.images)), 3),
               "n_images": len(res.images)}
    cost_gpu = summary["wall_s"] * _gpu_rate() if route == "local" else 0.0
    with SessionLocal() as db:  # một giao dịch: ảnh + ứng viên + attempt done
        att, job = db.get(VisualAttempt, att_id), db.get(VisualJob, job_id)
        for img in res.images:
            a = Asset(kind="generated", mime_type="image/png", width=img.width, height=img.height, byte_size=len(img.data),
                      sha256=hashlib.sha256(img.data).hexdigest(), storage_backend=settings.asset_storage_backend,
                      provenance={"source": "ai", "model_key": m.key, "model_name": m.name, "vendor": m.vendor,
                                  "model_id": res.meta.get("model_id") or (m.key if m.route == "local" else None),
                                  "route": m.route, "visual_job_id": job_id, "attempt_id": att_id, "seed": img.seed,
                                  "reference_asset_ids": ref_ids, "project_id": job.project_id, "shot_id": job.shot_id,
                                  "scene_version_id": job.scene_version_id, "character_version_ids": job.character_version_ids,
                                  "prompt_sha256": hashlib.sha256(req.prompt.encode()).hexdigest(), **{k: v for k, v in img.meta.items() if k != "remote"}})
            db.add(a)
            db.flush()
            db.add(AssetBlob(asset_id=a.id, data=img.data))
            db.add(VisualCandidate(job_id=job_id, attempt_id=att_id, asset_id=a.id, seed=img.seed))
        att.state, att.error, att.finished_at = "done", None, now()
        att.cost_api, att.cost_gpu = res.cost_api, cost_gpu
        att.metrics = {**(att.metrics or {}), **metrics}
        job.cost_spent = (job.cost_spent or 0) + res.cost_api + cost_gpu
        db.commit()
    return "ok"


# ------------------------------------------------------------------ QC
def _character_refs(db: Session, ctx: Context) -> list[list]:
    out = []
    for c in ctx.characters:
        embs = []
        for r in c.get("refs", [])[:2]:
            blob = db.get(AssetBlob, r["asset_id"])
            embs.append(QC.embed_reference(blob.data) if blob else None)
        out.append(embs)
    return out


def _prior_embeds(db: Session, job: VisualJob, ctx: Context) -> list:
    """Embedding mặt của vài shot đã duyệt trước đó trong cùng project/episode có cùng nhân vật (kiểm tra chuỗi cảnh)."""
    if len(ctx.characters) != 1 or QC.face_engine() is None:
        return []
    prev = db.scalars(select(VisualJob).where(VisualJob.project_id == job.project_id, VisualJob.state == "approved",
                                              VisualJob.id != job.id).order_by(VisualJob.reviewed_at.desc()).limit(8)).all()
    out = []
    for p in prev:
        if p.character_version_ids != job.character_version_ids or not p.selected_candidate_id:
            continue
        cand = db.get(VisualCandidate, p.selected_candidate_id)
        blob = db.get(AssetBlob, cand.asset_id) if cand else None
        e = QC.embed_reference(blob.data) if blob else None
        if e is not None:
            out.append(e)
        if len(out) >= 2:
            break
    return out


def run_qc(job_id: str, att_id: str) -> None:
    import time
    t0 = time.perf_counter()
    with SessionLocal() as db:
        job, att = db.get(VisualJob, job_id), db.get(VisualAttempt, att_id)
        ctx = Context.from_dict(job.scene_spec["_resolved"])
        refs = _character_refs(db, ctx)
        prior = _prior_embeds(db, job, ctx)
        cands = db.scalars(select(VisualCandidate).where(VisualCandidate.attempt_id == att_id)).all()
        for c in cands:
            blob = db.get(AssetBlob, c.asset_id)
            q = QC.evaluate(blob.data, target_aspect=ASPECTS[job.aspect_ratio], expected_faces=len(ctx.characters),
                            ref_embeds=refs, prior_embeds=prior)
            c.qc, c.qc_verdict, c.qc_score = q, q["verdict"], q["score"]
        att.metrics = {**(att.metrics or {}), "qc_done": True, "qc_s": round(time.perf_counter() - t0, 3)}
        db.commit()


# ------------------------------------------------------------------ vòng điều phối
def _next_pending(db: Session, job_id: str) -> VisualAttempt | None:
    return db.scalar(select(VisualAttempt).where(VisualAttempt.job_id == job_id, VisualAttempt.state.in_(("queued", "running")))
                     .order_by(VisualAttempt.attempt_no))


def _new_attempt(db: Session, job: VisualJob, model_key: str, kind: str, instruction: str | None) -> VisualAttempt:
    m = db.get(ProviderModel, model_key)
    n = (db.scalar(select(VisualAttempt.attempt_no).where(VisualAttempt.job_id == job.id).order_by(VisualAttempt.attempt_no.desc())) or 0) + 1
    a = VisualAttempt(job_id=job.id, attempt_no=n, kind=kind, model_key=model_key, route=m.route, state="queued", instruction=instruction)
    db.add(a)
    db.flush()
    return a


def _set_job(job_id: str, **vals) -> None:
    with SessionLocal() as db:
        db.execute(update(VisualJob).where(VisualJob.id == job_id).values(**vals))
        db.commit()


def run_visual_job(job_id: str) -> str:
    """Điểm vào của task Celery. Idempotent: giao lại task chỉ tiếp tục phần chưa xong."""
    with SessionLocal() as db:
        job = db.get(VisualJob, job_id)
        if not job:
            return "skipped"
        if job.state not in ("queued", "running"):
            return "skipped"
        if _next_pending(db, job_id) is None and not _has_unqc(db, job_id):
            return "skipped"
        job.state = "running"
        db.commit()
    guard = settings.visual_max_repairs + 8
    while guard > 0:
        guard -= 1
        with SessionLocal() as db:
            job = db.get(VisualJob, job_id)
            unqc = _unqc_attempt(db, job_id)
            att = None if unqc else _next_pending(db, job_id)
            att_id = att.id if att else None
        if unqc:
            run_qc(job_id, unqc)
            if _decide(job_id, unqc) != "continue":
                return "done"
            continue
        if att_id is None:
            return "done"
        outcome = _execute(job_id, att_id)
        if outcome == "budget":
            _set_job(job_id, state="needs_human", error="budget_exhausted: hết ngân sách, chuyển người xử lý")
            return "needs_human"
        if outcome == "provider_failed":
            if not _failover(job_id, att_id):
                return "failed"
            continue
        run_qc(job_id, att_id)
        if _decide(job_id, att_id) != "continue":
            return "done"
    _set_job(job_id, state="needs_human", error="loop_guard: vượt số vòng xử lý cho phép")
    return "needs_human"


def _has_unqc(db: Session, job_id: str) -> bool:
    return _unqc_attempt(db, job_id) is not None


def _unqc_attempt(db: Session, job_id: str) -> str | None:
    for a in db.scalars(select(VisualAttempt).where(VisualAttempt.job_id == job_id, VisualAttempt.state == "done")
                        .order_by(VisualAttempt.attempt_no.desc())):
        return a.id if not (a.metrics or {}).get("qc_done") else None
    return None


def _failover(job_id: str, failed_att_id: str) -> bool:
    """Lỗi nhà cung cấp => thử tuyến kế tiếp trong route_plan chưa bị lỗi. True nếu đã tạo attempt mới."""
    with SessionLocal() as db:
        job = db.get(VisualJob, job_id)
        att = db.get(VisualAttempt, failed_att_id)
        tried_failed = {a.model_key for a in db.scalars(select(VisualAttempt).where(VisualAttempt.job_id == job_id,
                                                                                    VisualAttempt.state == "failed"))}
        nxt = next((p["model_key"] for p in job.route_plan if p["model_key"] not in tried_failed), None)
        if job.pinned_model_key or nxt is None:
            job.state, job.error = "failed", (att.error or "provider_failed")[:2000]
            db.commit()
            return False
        _new_attempt(db, job, nxt, att.kind, att.instruction)
        job.error = f"failover: {att.model_key} lỗi ({(att.error or '')[:160]}) → {nxt}"
        db.commit()
        return True


def _decide(job_id: str, att_id: str) -> str:
    """Sau QC: chốt job hoặc xếp lượt sửa. Trả 'continue' nếu còn attempt phải chạy."""
    with SessionLocal() as db:
        job = db.get(VisualJob, job_id)
        att = db.get(VisualAttempt, att_id)
        cands = db.scalars(select(VisualCandidate).where(VisualCandidate.attempt_id == att_id)).all()
        ranked = sorted(cands, key=lambda c: (VERDICT_RANK.get(c.qc_verdict, 3), -(c.qc_score or 0)))
        best = ranked[0] if ranked else None
        if job.benchmark_run_id:  # benchmark: không sửa tự động, QC chỉ để tham khảo
            job.state, job.selected_candidate_id, job.error = "review_required", best.id if best else None, None
            db.commit()
            return "done"
        if best and best.qc_verdict in ("pass", "review"):
            job.state, job.selected_candidate_id, job.error = "review_required", best.id, None
            db.commit()
            return "done"
        # tất cả ảnh lỗi QC
        auto_kinds = att.kind in ("initial", "repair")
        budget_left = (not job.budget_limit) or job.cost_spent < job.budget_limit
        if job.auto_repair and auto_kinds and job.repair_count < settings.visual_max_repairs and budget_left:
            # chọn chỉ dẫn từ ảnh lỗi nặng nhất; ở lượt sửa cuối, đổi sang tuyến dự phòng khác nếu có
            qc = best.qc if best else {"checks": []}
            instruction, step = QC.failure_instruction(qc)
            model_key = att.model_key
            if job.repair_count == settings.visual_max_repairs - 1 and not job.pinned_model_key:
                alt = next((p["model_key"] for p in job.route_plan if p["model_key"] != att.model_key), None)
                model_key = alt or model_key
            job.repair_count += 1
            job.error = f"qc_fail: sửa tự động lần {job.repair_count}/{settings.visual_max_repairs} ({step}): {instruction[:120]}"
            _new_attempt(db, job, model_key, "repair", instruction)
            db.commit()
            return "continue"
        job.state = "needs_human"
        job.selected_candidate_id = best.id if best else None
        job.error = ("repair_limit/budget: " if job.auto_repair else "qc_fail: ") + "ảnh chưa đạt QC, chuyển người xử lý"
        db.commit()
        return "done"
