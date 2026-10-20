"""API báo cáo, trạng thái hệ thống và benchmark chấm mù (báo cáo EMVOOX1 mục 1.2)."""
import hashlib
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import reports as REP
from app.core.config import settings
from app.db.session import get_db
from app.models import (
    BenchmarkRating, BenchmarkRun, Character, ProviderModel, VisualAttempt, VisualCandidate, VisualJob,
)
from app.visual.service import CameraIn, CharacterIn, SceneSpecIn, ServiceError, VisualRequest, create_visual_job, worker_status
from app.visual.telemetry import system_snapshot
from app.workers.celery_app import celery_app

router = APIRouter(prefix="/api/v1")
SCENES_FILE = Path(__file__).parent / "visual" / "benchmark_scenes.json"
WEIGHTS = {"consistency": 0.4, "scene_fit": 0.3, "anatomy": 0.2, "aesthetics": 0.1}


def load_scenes() -> dict:
    return json.loads(SCENES_FILE.read_text(encoding="utf8"))


# ------------------------------------------------------------------ báo cáo + hệ thống
@router.get("/reports/summary")
def report_summary(db: Session = Depends(get_db)):
    return REP.full_report(db)


@router.get("/reports/report.md", response_class=PlainTextResponse)
def report_md(db: Session = Depends(get_db)):
    return PlainTextResponse(REP.report_markdown(REP.full_report(db)), media_type="text/markdown; charset=utf-8",
                             headers={"Content-Disposition": 'attachment; filename="emvoox-bao-cao-van-hanh.md"'})


@router.get("/reports/attempts.csv", response_class=PlainTextResponse)
def report_csv(db: Session = Depends(get_db)):
    return PlainTextResponse(REP.attempts_csv(db), media_type="text/csv; charset=utf-8",
                             headers={"Content-Disposition": 'attachment; filename="emvoox-attempts.csv"'})


@router.get("/system/stats")
def system_stats(db: Session = Depends(get_db)):
    return {"api_host": system_snapshot([settings.model_cache_dir, settings.dataset_dir, settings.export_dir]),
            "worker": worker_status(db)}


# ------------------------------------------------------------------ benchmark
class RunCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    round_no: int = Field(1, ge=1, le=2)             # 1 = chỉ prompt; 2 = có tham chiếu/LoRA
    model_keys: list[str] = Field(min_length=1, max_length=8)
    scene_ids: list[str] | None = None               # None = cả 30 cảnh
    variants: int = Field(4, ge=1, le=4)
    budget_limit_total: float = Field(0.0, ge=0)


@router.get("/benchmark/scenes")
def benchmark_scenes():
    return load_scenes()


@router.post("/benchmark/runs", status_code=201)
def create_run(body: RunCreate, db: Session = Depends(get_db)):
    data = load_scenes()
    scenes = [s for s in data["scenes"] if not body.scene_ids or s["id"] in body.scene_ids]
    if not scenes:
        raise HTTPException(422, "không có cảnh nào được chọn")
    run = BenchmarkRun(name=body.name, round_no=body.round_no, model_keys=body.model_keys, scene_ids=[s["id"] for s in scenes],
                       variants=body.variants, status="running")
    db.add(run)
    db.flush()
    chars = {c.code: c for c in db.scalars(select(Character))}
    created, skipped = 0, []
    est_total = 0.0
    jobs = []
    for mk in body.model_keys:
        if not db.get(ProviderModel, mk):
            skipped.append({"model_key": mk, "reason": "không có trong registry"})
            continue
        for s in scenes:
            need = [chars.get(c) for c in s["characters"]]
            if any(c is None or not c.current_version_id for c in need):
                skipped.append({"model_key": mk, "scene": s["id"], "reason": "thiếu nhân vật " + ",".join(s["characters"])})
                continue
            req = VisualRequest(
                project_id=f"benchmark-{run.id[:8]}", shot_id=f"{s['id']}-{mk}",
                characters=[CharacterIn(character_version_id=c.current_version_id, position=("left", "right")[i] if len(need) > 1 else None)
                            for i, c in enumerate(need)],
                scene_spec=SceneSpecIn(action=s["action"], emotion=s["emotion"], camera=CameraIn(**s["camera"]), lighting=s["lighting"],
                                       style=s["style"]),
                num_variants=body.variants, seed=1000 + int(s["id"][1:]) * 10, model_key=mk, auto_repair=False)
            try:
                job = create_visual_job(db, req, None, benchmark={"run_id": run.id, "scene_id": s["id"]}, strip_refs=body.round_no == 1)
            except ServiceError as e:
                skipped.append({"model_key": mk, "scene": s["id"], "reason": e.msg})
                continue
            est_total += next((p["est_cost"] for p in job.route_plan if p["model_key"] == mk), 0.0)
            jobs.append(job)
            created += 1
    if body.budget_limit_total and est_total > body.budget_limit_total:
        db.rollback()
        raise HTTPException(422, f"ước tính {est_total:.2f} USD vượt ngân sách lượt chạy {body.budget_limit_total}")
    db.commit()
    for j in jobs:
        try:
            celery_app.send_task("app.workers.tasks.visual_run", args=[j.id], queue="visual")
        except Exception as e:  # noqa: BLE001
            j.state, j.error = "failed", f"enqueue_failed: {e}"[:300]
            db.commit()
    return {"run_id": run.id, "jobs_created": created, "skipped": skipped, "estimated_cost": round(est_total, 4),
            "images_expected": created * body.variants}


def _run_candidates(db: Session, run_id: str):
    return db.execute(select(VisualCandidate, VisualJob).join(VisualJob, VisualJob.id == VisualCandidate.job_id)
                      .where(VisualJob.benchmark_run_id == run_id)).all()


@router.get("/benchmark/runs")
def list_runs(db: Session = Depends(get_db)):
    return [{"id": r.id, "name": r.name, "round_no": r.round_no, "model_keys": r.model_keys, "variants": r.variants,
             "scenes": len(r.scene_ids), "created_at": r.created_at.isoformat() if r.created_at else None}
            for r in db.scalars(select(BenchmarkRun).order_by(BenchmarkRun.created_at.desc()))]


def _weighted(c: int, s: int, a: int, e: int) -> float:
    return round((WEIGHTS["consistency"] * c + WEIGHTS["scene_fit"] * s + WEIGHTS["anatomy"] * a + WEIGHTS["aesthetics"] * e) * 20, 2)


@router.get("/benchmark/runs/{run_id}")
def run_detail(run_id: str, db: Session = Depends(get_db)):
    run = db.get(BenchmarkRun, run_id)
    if not run:
        raise HTTPException(404, "lượt benchmark không tồn tại")
    jobs = db.scalars(select(VisualJob).where(VisualJob.benchmark_run_id == run_id)).all()
    ratings: dict[str, list[BenchmarkRating]] = {}
    for r in db.scalars(select(BenchmarkRating).where(BenchmarkRating.run_id == run_id)):
        ratings.setdefault(r.candidate_id, []).append(r)
    per_model: dict[str, dict] = {}
    for cand, job in _run_candidates(db, run_id):
        att = db.get(VisualAttempt, cand.attempt_id)
        r = per_model.setdefault(att.model_key, {"images": 0, "rated": 0, "passed": 0, "scores": [], "core_errors": 0, "cost": 0.0,
                                                 "wall": [], "gpu_peak": [], "qc_pass": 0, "attempts_seen": set()})
        r["images"] += 1
        if att.id not in r["attempts_seen"]:
            r["attempts_seen"].add(att.id)
            r["cost"] += att.cost_api + att.cost_gpu
            if att.metrics.get("wall_s") is not None:
                r["wall"].append(att.metrics["wall_s"])
            if att.metrics.get("gpu_peak_mb") is not None and att.route == "local":
                r["gpu_peak"].append(att.metrics["gpu_peak_mb"])
        r["qc_pass"] += cand.qc_verdict == "pass"
        rs = ratings.get(cand.id, [])
        if len(rs) >= 2:
            r["rated"] += 1
            mean = sum(x.weighted for x in rs) / len(rs)
            core = any(x.core_error for x in rs)
            r["core_errors"] += core
            r["scores"].append(mean)
            r["passed"] += (not core) and mean >= settings.benchmark_pass_score
    summary = []
    for mk, r in per_model.items():
        summary.append({
            "model_key": mk, "images": r["images"], "rated_by_two": r["rated"], "passed": r["passed"], "core_errors": r["core_errors"],
            "pass_rate": round(r["passed"] / r["rated"], 4) if r["rated"] else None,
            "mean_weighted": round(sum(r["scores"]) / len(r["scores"]), 2) if r["scores"] else None,
            "qc_auto_pass": r["qc_pass"], "cost": round(r["cost"], 5),
            "cost_per_pass": round(r["cost"] / r["passed"], 5) if r["passed"] else None,
            "latency_s": {"p50": REP.pct(r["wall"], 0.5), "p95": REP.pct(r["wall"], 0.95)},
            "gpu_peak_mb_max": max(r["gpu_peak"]) if r["gpu_peak"] else None})
    total = sum(1 for _ in _run_candidates(db, run_id))
    return {"id": run.id, "name": run.name, "round_no": run.round_no, "variants": run.variants,
            "jobs": {"total": len(jobs), "done": sum(j.state in ("review_required", "approved", "needs_human", "rejected") for j in jobs),
                     "failed": sum(j.state == "failed" for j in jobs)},
            "images": total, "ratings": sum(len(v) for v in ratings.values()),
            "ratings_needed": total * 2, "weights": WEIGHTS, "pass_rule": f"không lỗi cốt lõi và điểm có trọng số ≥ {settings.benchmark_pass_score}",
            "models": sorted(summary, key=lambda x: (x["cost_per_pass"] is None, x["cost_per_pass"] or 0))}


@router.get("/benchmark/runs/{run_id}/next")
def next_to_rate(run_id: str, rater: str = Query(min_length=1, max_length=64), db: Session = Depends(get_db)):
    """Ảnh kế tiếp cần chấm cho `rater`. ẨN tên công cụ/model; thứ tự xáo ngẫu nhiên theo rater để không đoán được nhóm."""
    cands = _run_candidates(db, run_id)
    counts: dict[str, set[str]] = {}
    for r in db.scalars(select(BenchmarkRating).where(BenchmarkRating.run_id == run_id)):
        counts.setdefault(r.candidate_id, set()).add(r.rater)
    todo = [(c, j) for c, j in cands if len(counts.get(c.id, ())) < 2 and rater not in counts.get(c.id, ())
            and j.state in ("review_required", "approved", "needs_human", "rejected")]
    if not todo:
        return {"done": True}
    todo.sort(key=lambda cj: hashlib.sha256(f"{cj[0].id}:{rater}".encode()).hexdigest())   # xáo trộn ổn định theo người chấm
    c, j = todo[0]
    sc = next((s for s in load_scenes()["scenes"] if s["id"] == j.benchmark_scene_id), {})
    return {"done": False, "remaining": len(todo), "candidate_id": c.id, "asset_id": c.asset_id,
            "brief": {"group": sc.get("group"), "action": sc.get("action"), "emotion": sc.get("emotion"), "camera": sc.get("camera"),
                      "lighting": sc.get("lighting"), "characters": sc.get("characters")},
            "criteria": {"consistency": "nhất quán nhận dạng (40%)", "scene_fit": "đúng cảnh (30%)", "anatomy": "hình thể và chi tiết (20%)",
                         "aesthetics": "thẩm mỹ (10%)", "core_error": "lỗi loại: sai nhân vật hoặc sai hành động cốt lõi"}}


class RatingBody(BaseModel):
    run_id: str
    candidate_id: str
    rater: str = Field(min_length=1, max_length=64)
    consistency: int = Field(ge=1, le=5)
    scene_fit: int = Field(ge=1, le=5)
    anatomy: int = Field(ge=1, le=5)
    aesthetics: int = Field(ge=1, le=5)
    core_error: bool = False


@router.post("/benchmark/ratings", status_code=201)
def rate(body: RatingBody, db: Session = Depends(get_db)):
    cand = db.get(VisualCandidate, body.candidate_id)
    job = db.get(VisualJob, cand.job_id) if cand else None
    if not job or job.benchmark_run_id != body.run_id:
        raise HTTPException(422, "ảnh không thuộc lượt benchmark này")
    if db.scalar(select(BenchmarkRating).where(BenchmarkRating.candidate_id == body.candidate_id, BenchmarkRating.rater == body.rater)):
        raise HTTPException(409, "người chấm này đã chấm ảnh này")
    n = db.scalar(select(func.count()).select_from(BenchmarkRating).where(BenchmarkRating.candidate_id == body.candidate_id)) or 0
    if n >= 2:
        raise HTTPException(409, "ảnh đã đủ hai người chấm độc lập")
    db.add(BenchmarkRating(run_id=body.run_id, candidate_id=body.candidate_id, rater=body.rater, consistency=body.consistency,
                           scene_fit=body.scene_fit, anatomy=body.anatomy, aesthetics=body.aesthetics, core_error=body.core_error,
                           weighted=_weighted(body.consistency, body.scene_fit, body.anatomy, body.aesthetics)))
    db.commit()
    return {"ok": True}


@router.post("/benchmark/runs/{run_id}/apply")
def apply_results(run_id: str, min_rated: int = Query(20, ge=1), db: Session = Depends(get_db)):
    """Ghi kết quả vào Model Registry (cost_per_pass...) để Model Router ưu tiên theo chi phí/ảnh ĐẠT thay vì ước tính."""
    d = run_detail(run_id, db)
    applied = []
    for m in d["models"]:
        if m["rated_by_two"] < min_rated:
            continue
        pm = db.get(ProviderModel, m["model_key"])
        pm.benchmark = {**(pm.benchmark or {}), "run_id": run_id, "round_no": d["round_no"], "pass_rate": m["pass_rate"],
                        "mean_weighted": m["mean_weighted"], "cost_per_pass": m["cost_per_pass"], "n_rated": m["rated_by_two"],
                        "latency_s": m["latency_s"]}
        applied.append(m["model_key"])
    db.commit()
    return {"applied": applied, "min_rated": min_rated}

