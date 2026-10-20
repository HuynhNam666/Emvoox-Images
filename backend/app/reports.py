"""Báo cáo chi phí, thời gian, bộ nhớ, RAM (báo cáo EMVOOX1 mục 1.2).

C đạt = (C tạo + C sửa + C GPU + C QA + C lưu trữ + C huấn luyện phân bổ) / N đạt

Định nghĩa dùng ở đây (để không tính trùng):
  C tạo   = phí API của các lượt tạo đầu tiên/dự phòng (không phải lượt sửa), gồm cả phí ảnh nhập thủ công và phí lượt lỗi có thể đã tính
  C sửa   = toàn bộ chi phí (API + GPU) của các lượt sửa (tự động hoặc do người yêu cầu)
  C GPU   = chi phí giờ GPU của các lượt tạo (không phải sửa) chạy tự chạy: giây × GPU_COST_PER_HOUR / 3600
  C QA    = số lượt người duyệt × QA_COST_PER_REVIEW
  C lưu trữ = dung lượng ảnh sinh ra × STORAGE_COST_PER_GB_MONTH (một tháng)
  C huấn luyện phân bổ = giờ GPU train LoRA × đơn giá, chia đều cho các ảnh ĐẠT dùng LoRA đó (gán cho SDXL)
  N đạt   = số job ảnh được người phụ trách duyệt (approved), tính theo model của ảnh được chọn
Đơn giá GPU/QA/lưu trữ mặc định 0: khi chưa khai báo, báo cáo hiển thị GIỜ GPU thay vì quy ra tiền (không bịa đơn giá).
Độ trễ API chỉ đo đầu-cuối (gửi → nhận file); chỉ BFL có tách chờ/tải. VRAM/RAM của tuyến API không có ý nghĩa (chạy ngoài máy).
"""
import csv
import io
import statistics
from collections import defaultdict
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import (
    Asset, ModelVersion, ProviderModel, TrainingRun, VideoAttempt, VideoJob, VisualAttempt, VisualCandidate, VisualJob,
)
from app.visual.telemetry import system_snapshot


def pct(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 3)


def _stats(vals: list[float]) -> dict:
    vals = [v for v in vals if v is not None]
    return {"n": len(vals), "p50": pct(vals, 0.5), "p95": pct(vals, 0.95), "max": round(max(vals), 3) if vals else None,
            "mean": round(statistics.fmean(vals), 3) if vals else None}


def _cand_model(db: Session, job: VisualJob) -> str | None:
    c = db.get(VisualCandidate, job.selected_candidate_id) if job.selected_candidate_id else None
    a = db.get(VisualAttempt, c.attempt_id) if c else None
    return a.model_key if a else None


def visual_report(db: Session) -> dict:
    rate = settings.gpu_cost_per_hour / 3600
    attempts = db.scalars(select(VisualAttempt).join(VisualJob, VisualJob.id == VisualAttempt.job_id)
                          .where(VisualJob.benchmark_run_id.is_(None))).all()
    jobs = db.scalars(select(VisualJob).where(VisualJob.benchmark_run_id.is_(None))).all()
    by_model: dict[str, dict] = defaultdict(lambda: {
        "attempts": 0, "failed": 0, "images": 0, "approved": 0, "c_create": 0.0, "c_repair": 0.0, "c_gpu": 0.0, "gpu_seconds": 0.0,
        "wall": [], "queue": [], "infer": [], "gpu_peak": [], "ram_peak": [], "rss_peak": [], "torch_peak": [], "per_image": [], "fails": defaultdict(int)})
    for a in attempts:
        r = by_model[a.model_key]
        r["attempts"] += 1
        m = a.metrics or {}
        if a.state == "failed":
            r["failed"] += 1
            r["fails"][(a.error or "?").split(":")[0]] += 1
        if a.state == "done":
            r["images"] += m.get("n_images", 0)
        repair = a.kind in ("repair", "human_repair")
        if repair:
            r["c_repair"] += a.cost_api + a.cost_gpu
        else:
            r["c_create"] += a.cost_api
            r["c_gpu"] += a.cost_gpu
        if a.route == "local":
            r["gpu_seconds"] += m.get("wall_s") or 0
        if a.state == "done":
            for k, key in (("wall", "wall_s"), ("queue", "queue_s"), ("infer", "infer_s"), ("per_image", "per_image_s")):
                if m.get(key) is not None:
                    r[k].append(m[key])
            if a.route == "local":
                for k, key in (("gpu_peak", "gpu_peak_mb"), ("ram_peak", "ram_peak_mb"), ("rss_peak", "rss_peak_mb"), ("torch_peak", "torch_vram_peak_mb")):
                    if m.get(key) is not None:
                        r[k].append(m[key])
    approved = [j for j in jobs if j.state == "approved"]
    reviews = sum(len(j.review_history or []) for j in jobs)
    train_alloc: dict[str, float] = defaultdict(float)
    lora_uses: dict[str, list[VisualJob]] = defaultdict(list)
    for j in approved:
        mk = _cand_model(db, j)
        if mk:
            by_model[mk]["approved"] += 1
        cand = db.get(VisualCandidate, j.selected_candidate_id) if j.selected_candidate_id else None
        att = db.get(VisualAttempt, cand.attempt_id) if cand else None
        lid = (att.params or {}).get("lora_version_id") if att else None
        if lid:
            lora_uses[lid].append(j)
    train_hours = 0.0
    for lid, js in lora_uses.items():
        mv = db.get(ModelVersion, lid)
        run = db.get(TrainingRun, mv.training_run_id) if mv and mv.training_run_id else None
        if run and run.started_at and run.finished_at:
            sec = (run.finished_at - run.started_at).total_seconds()
            train_hours += sec / 3600
            train_alloc["sdxl-1.0"] += sec * rate
    gen_bytes = sum(a.byte_size or 0 for a in db.scalars(select(Asset).where(Asset.kind == "generated")))
    c_store = gen_bytes / 2**30 * settings.storage_cost_per_gb_month
    c_qa = reviews * settings.qa_cost_per_review

    rows = []
    tot = dict(c_create=0.0, c_repair=0.0, c_gpu=0.0)
    for k, r in sorted(by_model.items()):
        for t in tot:
            tot[t] += r[t]
    n_app_total = max(1, len(approved))
    for k, r in sorted(by_model.items()):
        share = r["approved"] / n_app_total
        c_total = r["c_create"] + r["c_repair"] + r["c_gpu"] + c_qa * share + c_store * share + train_alloc.get(k, 0.0)
        rows.append({
            "model_key": k, "attempts": r["attempts"], "failed": r["failed"], "images": r["images"], "approved": r["approved"],
            "approval_rate": round(r["approved"] / r["images"], 4) if r["images"] else None,
            "fail_reasons": dict(r["fails"]),
            "cost": {"c_create": round(r["c_create"], 5), "c_repair": round(r["c_repair"], 5), "c_gpu": round(r["c_gpu"], 5),
                     "c_qa_alloc": round(c_qa * share, 5), "c_storage_alloc": round(c_store * share, 5),
                     "c_training_alloc": round(train_alloc.get(k, 0.0), 5), "total": round(c_total, 5),
                     "per_approved": round(c_total / r["approved"], 5) if r["approved"] else None},
            "gpu_hours": round(r["gpu_seconds"] / 3600, 4),
            "latency_s": {"wall": _stats(r["wall"]), "queue": _stats(r["queue"]), "infer": _stats(r["infer"]), "per_image": _stats(r["per_image"])},
            "memory_mb": {"gpu_peak": _stats(r["gpu_peak"]), "torch_vram_peak": _stats(r["torch_peak"]),
                          "ram_peak_system": _stats(r["ram_peak"]), "process_rss_peak": _stats(r["rss_peak"])},
        })
    return {
        "formula": "C đạt = (C tạo + C sửa + C GPU + C QA + C lưu trữ + C huấn luyện phân bổ) / N đạt",
        "rates": {"gpu_cost_per_hour": settings.gpu_cost_per_hour, "qa_cost_per_review": settings.qa_cost_per_review,
                  "storage_cost_per_gb_month": settings.storage_cost_per_gb_month,
                  "note": "đơn giá = 0 nghĩa là chưa khai báo: chi phí quy tiền của GPU/QA/lưu trữ chưa được tính, xem gpu_hours"},
        "totals": {"jobs": len(jobs), "approved": len(approved), "reviews": reviews, "stored_gb": round(gen_bytes / 2**30, 4),
                   "c_create": round(tot["c_create"], 5), "c_repair": round(tot["c_repair"], 5), "c_gpu": round(tot["c_gpu"], 5),
                   "c_qa": round(c_qa, 5), "c_storage": round(c_store, 5), "c_training": round(sum(train_alloc.values()), 5),
                   "training_gpu_hours": round(train_hours, 4),
                   "c_total": round(sum(tot.values()) + c_qa + c_store + sum(train_alloc.values()), 5)},
        "models": rows,
    }


def video_report(db: Session) -> dict:
    atts = db.scalars(select(VideoAttempt)).all()
    rendered = [a for a in atts if a.state == "rendered"]
    reviewed = [a for a in rendered if a.review]
    approved = [a for a in reviewed if a.review.get("decision") == "approve"]
    total_cost = sum(a.cost or 0 for a in atts)
    mets = [a.metrics or {} for a in rendered]
    return {
        "attempts": len(atts), "rendered": len(rendered), "failed": sum(a.state == "failed" for a in atts),
        "reviewed": len(reviewed), "approved": len(approved),
        "approval_rate": round(len(approved) / len(reviewed), 4) if reviewed else None,
        "latency_s": {"total": _stats([a.total_seconds for a in rendered if a.total_seconds is not None]),
                      "render_wait": _stats([m.get("wait_s") for m in mets if m.get("wait_s") is not None]),
                      "download": _stats([m.get("download_s") for m in mets if m.get("download_s") is not None])},
        "memory_mb": {"gpu_peak": _stats([m.get("gpu_peak_mb") for m in mets if m.get("gpu_peak_mb") is not None]),
                      "ram_peak": _stats([m.get("ram_peak_mb") for m in mets if m.get("ram_peak_mb") is not None])},
        "gpu_hours": round(sum(a.gpu_seconds or 0 for a in atts) / 3600, 4),
        "cost": {"total": round(total_cost, 5), "per_approved": round(total_cost / len(approved), 5) if approved else None,
                 "unit": "giây GPU" if not settings.video_cost_per_gpu_second else "theo VIDEO_COST_PER_GPU_SECOND"},
        "jobs": db.query(VideoJob).count(),
    }


def full_report(db: Session) -> dict:
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "visual": visual_report(db), "video": video_report(db),
            "system": system_snapshot([settings.model_cache_dir, settings.dataset_dir]),
            "models_registry": [{"key": m.key, "verification": m.verification, "route": m.route, "checked_at": m.checked_at}
                                for m in db.scalars(select(ProviderModel).order_by(ProviderModel.key))]}


def attempts_csv(db: Session) -> str:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["job_id", "shot_id", "attempt_no", "kind", "model_key", "route", "state", "error", "cost_api", "cost_gpu", "wall_s", "queue_s",
                "infer_s", "load_s", "per_image_s", "n_images", "steps", "gpu_peak_mb", "torch_vram_peak_mb", "ram_peak_mb", "rss_peak_mb",
                "gpu_name", "started_at", "finished_at"])
    q = select(VisualAttempt, VisualJob).join(VisualJob, VisualJob.id == VisualAttempt.job_id).order_by(VisualAttempt.created_at)
    for a, j in db.execute(q):
        m = a.metrics or {}
        w.writerow([j.id, j.shot_id, a.attempt_no, a.kind, a.model_key, a.route, a.state, (a.error or "")[:200], a.cost_api, a.cost_gpu,
                    m.get("wall_s"), m.get("queue_s"), m.get("infer_s"), m.get("load_s"), m.get("per_image_s"), m.get("n_images"),
                    m.get("steps"), m.get("gpu_peak_mb"), m.get("torch_vram_peak_mb"), m.get("ram_peak_mb"), m.get("rss_peak_mb"),
                    m.get("gpu_name"), a.started_at, a.finished_at])
    return out.getvalue()


def report_markdown(rep: dict) -> str:
    v, vd, sysd = rep["visual"], rep["video"], rep["system"]
    f = lambda x, d=3: "—" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))  # noqa: E731
    L = ["# Báo cáo vận hành EMVOOX — chi phí, thời gian, bộ nhớ", "", f"Sinh lúc: {rep['generated_at']}", "",
         "Số liệu lấy từ các lượt chạy thật đã ghi trong hệ thống. Ô '—' = chưa có dữ liệu hoặc không đo được (không suy diễn).", "",
         "## 1. Ảnh — tổng hợp", "", f"Công thức: `{v['formula']}`", "",
         f"- Job: {v['totals']['jobs']} · được duyệt: {v['totals']['approved']} · lượt duyệt: {v['totals']['reviews']}",
         f"- C tạo {f(v['totals']['c_create'], 4)} · C sửa {f(v['totals']['c_repair'], 4)} · C GPU {f(v['totals']['c_gpu'], 4)} · C QA {f(v['totals']['c_qa'], 4)} "
         f"· C lưu trữ {f(v['totals']['c_storage'], 4)} · C huấn luyện {f(v['totals']['c_training'], 4)} (USD) → **tổng {f(v['totals']['c_total'], 4)}**",
         f"- Giờ GPU huấn luyện LoRA được phân bổ: {f(v['totals']['training_gpu_hours'], 3)} h",
         f"- Lưu trữ ảnh sinh ra: {f(v['totals']['stored_gb'], 3)} GB", f"- Lưu ý đơn giá: {v['rates']['note']}", "",
         "## 2. Ảnh — theo model", "",
         "| Model | Lượt | Lỗi | Ảnh | Đạt | Tỉ lệ đạt | Chi phí/ảnh đạt | Giờ GPU | p50 / p95 (s) | GPU đỉnh (MB) | RAM đỉnh (MB) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in v["models"]:
        L.append(f"| {r['model_key']} | {r['attempts']} | {r['failed']} | {r['images']} | {r['approved']} | {f(r['approval_rate'])} | "
                 f"{f(r['cost']['per_approved'], 4)} | {f(r['gpu_hours'], 3)} | {f(r['latency_s']['wall']['p50'], 1)} / {f(r['latency_s']['wall']['p95'], 1)} | "
                 f"{f(r['memory_mb']['gpu_peak']['max'], 0)} | {f(r['memory_mb']['ram_peak_system']['max'], 0)} |")
    L += ["", "## 3. Video (ComfyUI + LTX-2.3 + PromptRelay)", "",
          f"- Lượt render: {vd['rendered']} xong · {vd['failed']} lỗi · đã xem {vd['reviewed']} · đạt {vd['approved']} (tỉ lệ {f(vd['approval_rate'])})",
          f"- Độ trễ tổng p50/p95: {f(vd['latency_s']['total']['p50'], 1)} / {f(vd['latency_s']['total']['p95'], 1)} s · chờ render p50 {f(vd['latency_s']['render_wait']['p50'], 1)} s · tải file p50 {f(vd['latency_s']['download']['p50'], 1)} s",
          f"- GPU đỉnh (ComfyUI): max {f(vd['memory_mb']['gpu_peak']['max'], 0)} MB · RAM đỉnh max {f(vd['memory_mb']['ram_peak']['max'], 0)} MB",
          f"- Giờ GPU: {f(vd['gpu_hours'], 3)} h · chi phí: {f(vd['cost']['total'], 2)} ({vd['cost']['unit']}) · chi phí/clip đạt: {f(vd['cost']['per_approved'], 2)}", "",
          "## 4. Máy chủ hiện tại", ""]
    for g in sysd.get("gpus", []):
        L.append(f"- GPU {g['name']}: dùng {g['used_mb']:.0f}/{g['total_mb']:.0f} MB, tải {g['util']:.0f}%")
    if not sysd.get("gpus"):
        L.append("- API không thấy GPU (bình thường: GPU nằm ở container worker/ComfyUI; xem worker status ở /visual/models)")
    if sysd.get("ram"):
        L.append(f"- RAM: dùng {sysd['ram']['used_mb']} / {sysd['ram']['total_mb']} MB (còn {sysd['ram']['available_mb']} MB) · CPU {sysd.get('cpu_percent')}% · {sysd.get('cpu_count')} lõi")
    for p, d in sysd.get("disks", {}).items():
        if d:
            L.append(f"- Đĩa `{p}`: trống {d['free_gb']} / {d['total_gb']} GB")
    L += ["", "## 5. Độ tin cậy của số liệu", "",
          "- Tuyến API: chi phí = giá công khai trong Model Registry (chưa đối chiếu hóa đơn); độ trễ đo đầu-cuối.",
          "- Tuyến tự chạy: VRAM đỉnh lấy từ nvidia-smi (toàn GPU, gồm tiến trình khác) và torch (riêng pipeline); RAM đỉnh là RAM hệ thống đang dùng.",
          "- Video: VRAM/RAM lấy từ /system_stats của ComfyUI nên phản ánh máy chạy ComfyUI.", ""]
    return "\n".join(L)
