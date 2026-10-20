"""Model Router (báo cáo mục 3.3/3.5): chọn tuyến chính + dự phòng theo khả năng, trạng thái và ngân sách.

Nguyên tắc từ báo cáo:
- Khi cùng vượt ngưỡng QC, ưu tiên tuyến rẻ hơn (chi phí / ảnh ĐẠT nếu đã có benchmark, không thì chi phí ước tính/ảnh).
- Không mặc định dùng lại Character LoRA của model chính cho model dự phòng: LoRA chỉ áp cho đúng họ model (SDXL).
- Giữ ít nhất một tuyến khác nhà cung cấp / khác loại (tự chạy vs API) ở vị trí dự phòng đầu tiên.
- Model không khả dụng (thiếu khóa, thiếu VRAM, endpoint chưa xác minh) bị loại, kèm lý do để hiển thị.
"""
import statistics
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import ProviderModel, VisualAttempt
from app.visual.providers.api_providers import BFLProvider
from app.visual.registry import estimate_api_cost

API_KEY_OF = {
    "flux2-pro": lambda: settings.bfl_api_key, "flux2-klein-4b-api": lambda: settings.bfl_api_key,
    "flux3-image": lambda: settings.bfl_api_key and settings.bfl_flux3_endpoint,
    "gemini-nano-banana": lambda: settings.gemini_api_key, "seedream-4-5": lambda: settings.seedream_api_key,
    "gpt-image": lambda: settings.openai_api_key,
}


@dataclass
class Need:
    task: str
    identity: bool          # có nhân vật cần giữ mặt
    n_characters: int
    text: bool
    width: int
    height: int
    n_variants: int
    n_refs: int


def classify_task(spec_task: str | None, n_chars: int, close: bool, text: bool) -> str:
    if spec_task:
        return spec_task
    if text:
        return "text_poster"
    if n_chars == 0:
        return "concept"
    if n_chars >= 2:
        return "multi_person"
    return "closeup" if close else "action_face"


def availability(m: ProviderModel, worker: dict | None) -> tuple[bool, str]:
    if not m.enabled:
        return False, "đã tắt trong registry"
    if m.route == "manual":
        return False, "không có API: nhập ảnh thủ công"
    if m.route == "api":
        if m.key == "flux3-image" and not settings.bfl_flux3_endpoint:
            return False, "endpoint 'FLUX 3 Image' chưa xác minh (đặt BFL_FLUX3_ENDPOINT)"
        return (True, "ok") if API_KEY_OF[m.key]() else (False, "chưa cấu hình khóa API")
    if not worker or not worker.get("cuda"):
        return False, "worker GPU chưa báo cáo CUDA"
    total = worker.get("total_gb") or settings.worker_vram_gb
    need = m.min_vram_gb or 0
    if total and total < need * 0.5:
        return False, f"GPU {total:.0f}GB < ngưỡng tối thiểu {need * 0.5:.0f}GB (cần {need:.0f}GB, hoặc offload)"
    return True, "ok" if not total or total >= need + 2 else f"ok (offload CPU: GPU {total:.0f}GB < {need + 2:.0f}GB, chậm hơn)"


def supports(m: ProviderModel, need: Need, lora_ok: bool) -> tuple[bool, str]:
    c = m.capabilities or {}
    if need.identity and not (c.get("ref_identity") or (lora_ok and c.get("lora"))):
        return False, "không giữ được nhân vật (không có tham chiếu/LoRA)"
    if need.n_characters >= 2 and not c.get("multi_ref"):
        return False, "nhiều nhân vật cần đa tham chiếu để không trộn mặt"
    if need.text and not c.get("text_render"):
        return False, "không đảm bảo chữ trong ảnh"
    if need.n_refs and not (c.get("ref_identity") or c.get("multi_ref") or c.get("edit")) and not need.identity:
        return False, "không nhận ảnh tham chiếu"
    if not need.n_refs and c.get("edit") and not c.get("txt2img"):
        return False, "chỉ sửa ảnh, cần ảnh tham chiếu"
    return True, "ok"


def _hist_cost(db: Session, m: ProviderModel, n: int, rate: float) -> float | None:
    rows = db.scalars(select(VisualAttempt).where(VisualAttempt.model_key == m.key, VisualAttempt.state == "done").limit(50)).all()
    per = [(a.cost_api + a.cost_gpu) / max(1, (a.params or {}).get("n", 1)) for a in rows if (a.cost_api + a.cost_gpu) > 0]
    return round(statistics.median(per) * n, 5) if per else None


def est_cost(db: Session, m: ProviderModel, need: Need) -> float:
    if m.route == "api":
        return estimate_api_cost(m, need.width, need.height, need.n_variants, need.n_refs)
    rate = settings.gpu_cost_per_hour / 3600
    h = _hist_cost(db, m, need.n_variants, rate)
    return h if h is not None else 0.0


def plan_route(db: Session, need: Need, worker: dict | None, lora_ok: bool, pinned: str | None = None) -> tuple[list[dict], list[dict]]:
    """Trả (plan, rejected). plan = [{model_key, route, reason, est_cost, preferred}] theo thứ tự thử."""
    models = db.scalars(select(ProviderModel)).all()
    plan, rejected = [], []
    for m in models:
        if pinned and m.key != pinned:
            continue
        ok, why = availability(m, worker)
        if not ok:
            rejected.append({"model_key": m.key, "reason": why})
            continue
        ok2, why2 = supports(m, need, lora_ok)
        if not ok2 and not pinned:
            rejected.append({"model_key": m.key, "reason": why2})
            continue
        bench = (m.benchmark or {}).get("cost_per_pass")
        plan.append({
            "model_key": m.key, "route": m.route, "vendor": m.vendor, "availability": why,
            "preferred": need.task in (m.tasks or []), "est_cost": est_cost(db, m, need),
            "cost_per_pass": bench, "reason": f"{'đề xuất cho' if need.task in (m.tasks or []) else 'dự phòng cho'} tác vụ {need.task}",
            "verification": m.verification,
        })
    if pinned and not plan:
        raise ValueError(f"model {pinned} không khả dụng: " + (rejected[0]["reason"] if rejected else "không có trong registry"))
    # Ưu tiên: tuyến đã xác minh/được đề xuất cho tác vụ, rồi rẻ nhất theo chi phí/ảnh đạt (nếu có) hoặc ước tính.
    plan.sort(key=lambda p: (not p["preferred"], p["verification"] == "unverified",
                             p["cost_per_pass"] if p["cost_per_pass"] is not None else p["est_cost"]))
    if len(plan) > 2 and not pinned:  # dự phòng đầu tiên khác loại/nhà cung cấp với tuyến chính
        first = plan[0]
        for i in range(1, len(plan)):
            if plan[i]["route"] != first["route"] or plan[i]["vendor"] != first["vendor"]:
                plan.insert(1, plan.pop(i))
                break
    return plan, rejected


def api_ready(model_key: str) -> bool:
    return bool(API_KEY_OF.get(model_key, lambda: "")())


__all__ = ["Need", "classify_task", "plan_route", "availability", "est_cost", "BFLProvider"]
