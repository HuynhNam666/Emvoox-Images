"""Tự kiểm tra server trước khi dùng thật. Chạy trong container api:

  docker compose exec api python scripts/server_selfcheck.py

Mỗi dòng: OK / CẢNH BÁO / LỖI kèm cách sửa. Thoát mã 1 nếu có LỖI. Script CHỈ ĐỌC (không tạo job, không tốn tiền API);
việc render thật được kiểm bằng scripts/smoke_video.py.
"""
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
from sqlalchemy import select, text  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.ml import comfy_client as C  # noqa: E402
from app.models import ProviderModel, SystemStatus  # noqa: E402
from app.visual import export as X  # noqa: E402
from app.visual import router as R  # noqa: E402
from app.visual.model_manifest import CORE, GROUPS  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []


def rec(level: str, name: str, msg: str) -> None:
    RESULTS.append((level, name, msg))
    icon = {"OK": "✓", "WARN": "!", "FAIL": "✗"}[level]
    print(f"[{icon}] {name}: {msg}")


def main() -> int:
    # 1. cơ sở dữ liệu
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
            n = db.query(ProviderModel).count()
        rec("OK" if n else "FAIL", "PostgreSQL", f"kết nối được; Model Registry có {n} model" if n else "registry rỗng: khởi động lại api để seed")
    except Exception as e:  # noqa: BLE001
        rec("FAIL", "PostgreSQL", f"{e}")
        return 1
    # 2. Redis
    try:
        import redis
        redis.Redis.from_url(settings.redis_url, socket_timeout=3).ping()
        rec("OK", "Redis", "ping được")
    except Exception as e:  # noqa: BLE001
        rec("FAIL", "Redis", f"{e}")
    # 3. bảo mật / cấu hình
    rec("OK" if settings.api_token else "WARN", "Đăng nhập", "API_TOKEN đã đặt" if settings.api_token else
        "API_TOKEN rỗng: ai vào được cổng web đều dùng được hệ thống (đặt API_TOKEN trước khi mở ra Internet)")
    rec("OK", "Cờ pháp lý", f"COMMERCIAL_USE={settings.commercial_use} REQUIRE_LABEL={settings.require_label} REQUIRE_C2PA={settings.require_c2pa}")
    # 4. worker GPU
    with SessionLocal() as db:
        row = db.get(SystemStatus, "worker")
    w = row.value if row else None
    if not w:
        rec("FAIL", "Worker GPU", "chưa thấy worker báo trạng thái: `docker compose logs worker-gpu`; đã chạy với --profile gpu chưa?")
    elif not w.get("cuda"):
        rec("FAIL", "Worker GPU", f"worker không thấy CUDA ({w.get('reason')}): kiểm NVIDIA driver + Container Toolkit (`nvidia-smi` trong container)")
    else:
        age = ""
        try:
            at = datetime.fromisoformat(w["at"])
            age = f", báo cáo {int((datetime.now(timezone.utc) - at).total_seconds() // 60)} phút trước"
        except Exception:  # noqa: BLE001
            pass
        rec("OK", "Worker GPU", f"{w.get('name')} · {w.get('total_gb')} GB VRAM · torch {w.get('torch')}{age}")
        face = w.get("face_ok")
        rec("OK" if face else "WARN", "Quality Gate (FaceNet)", "nạp được" if face else
            f"không nạp được ({w.get('face_error')}) → ảnh có nhân vật luôn ở trạng thái 'cần người xem' (chạy models-init nhóm face)")
    # 5. model
    root = Path(settings.model_cache_dir)
    free = shutil.disk_usage(root).free / 2**30 if root.exists() else 0
    rec("OK" if free > 50 else "WARN", "Đĩa model", f"{root}: trống {free:.0f} GB")
    for g in CORE:
        missing = []
        for it in GROUPS[g]["items"]:
            if it["type"] == "comfy_file" and not (root / "comfyui" / it["dest"]).exists():
                missing.append(it["dest"])
            if it["type"] == "hf_snapshot" and not list(root.glob(f"models--{it['repo'].replace('/', '--')}/snapshots/*")):
                missing.append(it["repo"])
        rec("OK" if not missing else "FAIL", f"Model nhóm {g}", "đủ" if not missing else
            f"thiếu {len(missing)}: {', '.join(missing[:3])}… → `docker compose --profile init run --rm models-init`")
    # 6. ComfyUI + workflow
    try:
        wf = C.load_workflow()
        oi = C._json("/object_info", timeout=120)
        errs = C.structure_errors(wf, oi)
        files = C.missing_model_files(wf, oi)
        rec("OK" if not errs else "FAIL", "ComfyUI node", f"{settings.comfyui_url} khớp workflow {wf['version']}" if not errs else "; ".join(errs[:4]))
        rec("OK" if not files else "FAIL", "ComfyUI model", "loader thấy đủ file" if not files else "thiếu: " + "; ".join(files[:3]))
        stats = C._json("/system_stats")
        d = (stats.get("devices") or [{}])[0]
        rec("OK", "ComfyUI GPU", f"{d.get('name')} · VRAM {d.get('vram_total', 0) / 2**30:.0f} GB · ComfyUI {stats.get('system', {}).get('comfyui_version')}")
        rec("OK" if wf["status"] == "verified" else "WARN", "Workflow video",
            "đã render clip thật" if wf["status"] == "verified" else
            "đã khớp schema nhưng chưa có clip thật: chạy scripts/smoke_video.py rồi đặt _meta.status=verified trong ltx23_promptrelay.api.json")
    except C.ComfyError as e:
        rec("FAIL", "ComfyUI", f"{e.code}: {e}")
    # 7. model API
    with SessionLocal() as db:
        for m in db.scalars(select(ProviderModel).where(ProviderModel.route == "api").order_by(ProviderModel.key)):
            ok, why = R.availability(m, None)
            rec("OK" if ok else "WARN", f"API {m.key}", f"đã có khóa ({m.verification}: chưa gọi thử, hãy chạy 1 job thật)" if ok else why)
    # 8. xuất bản
    try:
        X._font(32)
        rec("OK", "Font nhãn AI", "có font tiếng Việt")
    except X.ExportError as e:
        rec("FAIL", "Font nhãn AI", str(e))
    rec("OK" if shutil.which("ffmpeg") else "WARN", "ffmpeg (nhãn AI trên video)", "có" if shutil.which("ffmpeg") else "thiếu ffmpeg")
    ok, why = X.c2pa_configured()
    if ok:
        import io
        from PIL import Image
        b = io.BytesIO()
        Image.new("RGB", (64, 64), (80, 90, 100)).save(b, "PNG")
        _, rep = X.sign_and_verify(b.getvalue(), "image/png", ".png", X.manifest_json("selfcheck", False, {}))
        rec("OK" if rep.get("verified") else "FAIL", "C2PA", f"ký + xác minh được ({rep.get('validation_state')}; trusted={rep.get('trusted')})"
            if rep.get("verified") else f"ký lỗi: {rep.get('error')}")
        if not rep.get("trusted"):
            rec("WARN", "C2PA tin cậy", "chứng thư chưa nằm trong danh sách tin cậy C2PA: người xem công khai sẽ thấy cảnh báo (dùng chứng thư CA được công nhận khi phát hành thật)")
    else:
        rec("FAIL" if settings.require_c2pa else "WARN", "C2PA", f"{why}" + (" — REQUIRE_C2PA=true nên MỌI bản xuất sẽ bị chặn" if settings.require_c2pa else
                                                                         " — bản xuất chỉ có nhãn hiển thị, chưa ký (scripts/make_dev_c2pa_cert.py cho thử nghiệm)"))
    # 9. đơn giá cho báo cáo
    undeclared = [k for k, v in (("GPU_COST_PER_HOUR", settings.gpu_cost_per_hour), ("QA_COST_PER_REVIEW", settings.qa_cost_per_review),
                                 ("STORAGE_COST_PER_GB_MONTH", settings.storage_cost_per_gb_month)) if not v]
    rec("OK" if not undeclared else "WARN", "Đơn giá báo cáo", "đã khai báo" if not undeclared else
        "chưa khai báo " + ", ".join(undeclared) + " → báo cáo chỉ hiện giờ GPU, không quy ra tiền")
    fails = sum(1 for r in RESULTS if r[0] == "FAIL")
    warns = sum(1 for r in RESULTS if r[0] == "WARN")
    print(f"\nTổng: {len(RESULTS) - fails - warns} OK · {warns} cảnh báo · {fails} lỗi")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
