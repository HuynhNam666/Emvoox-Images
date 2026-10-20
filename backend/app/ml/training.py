"""Huấn luyện Character LoRA SDXL qua script DreamBooth-LoRA chính thức của Diffusers (đã ghim phiên bản).

Không dùng shell; argv là list được dựng từ cấu hình đã validate. Loss/bước lấy từ tiến trình thật của trainer.
"""
import hashlib
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select, update

from app.core.config import settings
from app.db.session import SessionLocal
from app.models import (
    Asset, AssetBlob, DatasetVersionItem, ModelVersion, TrainingMetric, TrainingRun,
)

# backend/training/vendor/... (cả khi chạy trực tiếp lẫn trong image: /app/training/vendor/...)
TRAINER_DIR = Path(__file__).resolve().parents[2] / "training" / "vendor" / "diffusers-v0.41.0"
TRAINER_SCRIPT = TRAINER_DIR / "train_dreambooth_lora_sdxl.py"
TRAINER_REVISION = "diffusers v0.41.0 examples/dreambooth/train_dreambooth_lora_sdxl.py"
VAE_FP16_FIX = "madebyollin/sdxl-vae-fp16-fix"
# Heuristic (CHƯA benchmark): VRAM trống tối thiểu gợi ý theo resolution khi bật gradient checkpointing.
HEURISTIC_MIN_FREE_GB = {512: 6.5, 768: 8.5, 1024: 12.0}

_TQDM = re.compile(r"Steps:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)\s*\[[^\]]*?loss=([0-9.eE+-]+),\s*lr=([0-9.eE+-]+)")
_SECRET = re.compile(r"(hf_[A-Za-z0-9]{10,}|(?i:token|api[_-]?key|password)\s*[=:]\s*\S+)")


def mask(text: str) -> str:
    return _SECRET.sub("***", text)


def now() -> datetime:
    return datetime.now(timezone.utc)


def snapshot_revision(model_id: str) -> str | None:
    """Commit HF của base model trong cache cục bộ (nếu đã tải)."""
    snaps = Path(settings.model_cache_dir) / ("models--" + model_id.replace("/", "--")) / "snapshots"
    if snaps.is_dir():
        revs = sorted(p.name for p in snaps.iterdir() if p.is_dir())
        return revs[-1] if revs else None
    return None


def resolve_steps(n_train: int, cfg: dict) -> int:
    if cfg.get("max_train_steps"):
        return int(cfg["max_train_steps"])
    per_epoch = math.ceil(math.ceil(n_train / cfg["train_batch_size"]) / cfg["gradient_accumulation_steps"])
    return per_epoch * int(cfg["num_train_epochs"])


def materialize(run_id: str, version_ids: list[str], trigger: str, work: Path) -> tuple[int, list[str]]:
    """Xuất ảnh train + metadata.jsonl từ PostgreSQL ra đĩa cục bộ. Trả (số ảnh train, prompt val)."""
    data = work / "data"
    if data.exists():
        shutil.rmtree(data)
    data.mkdir(parents=True)
    n, val_prompts = 0, []
    with SessionLocal() as db, open(data / "metadata.jsonl", "w", encoding="utf-8") as meta:
        rows = db.scalars(select(DatasetVersionItem).where(DatasetVersionItem.version_id.in_(version_ids))
                          .order_by(DatasetVersionItem.sha256, DatasetVersionItem.id))
        for r in rows:
            cap = r.caption.strip()
            if trigger not in cap:
                cap = f"{trigger}, {cap}".rstrip(", ")
            if r.split != "train":
                if r.split == "val":
                    val_prompts.append(cap)
                continue
            asset = db.get(Asset, r.asset_id)
            blob = db.get(AssetBlob, r.asset_id)
            ext = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}[asset.mime_type]
            name = f"{n:04d}_{r.sha256[:8]}.{ext}"
            (data / name).write_bytes(blob.data)
            meta.write(json.dumps({"file_name": name, "prompt": cap}, ensure_ascii=False) + "\n")
            n += 1
    return n, val_prompts


def build_argv(run: dict, base_model_id: str, work: Path, resume: bool) -> list[str]:
    c = run["config"]
    argv = [
        sys.executable, str(TRAINER_SCRIPT),
        f"--pretrained_model_name_or_path={base_model_id}",
        f"--dataset_name={work / 'data'}", "--image_column=image", "--caption_column=prompt",
        f"--instance_prompt={run['trigger_token']}",
        f"--output_dir={work / 'out'}", f"--logging_dir={work / 'tb'}", "--report_to=tensorboard",
        f"--resolution={c['resolution']}", "--center_crop",
        f"--train_batch_size={c['train_batch_size']}",
        f"--gradient_accumulation_steps={c['gradient_accumulation_steps']}",
        f"--mixed_precision={c['mixed_precision']}",
        f"--learning_rate={c['learning_rate']}", "--lr_scheduler=constant", "--lr_warmup_steps=0",
        f"--max_train_steps={run['max_steps']}",
        f"--checkpointing_steps={c['checkpointing_steps']}", "--checkpoints_total_limit=2",
        f"--rank={c['rank']}", f"--seed={c['seed']}", "--dataloader_num_workers=0",
    ]
    if c.get("gradient_checkpointing"):
        argv.append("--gradient_checkpointing")
    if c.get("use_vae_fp16_fix"):
        argv.append(f"--pretrained_vae_model_name_or_path={VAE_FP16_FIX}")
    if resume:
        argv.append("--resume_from_checkpoint=latest")
    return argv


def kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        import signal
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)


def _set(run_id: str, **vals) -> None:
    with SessionLocal() as db:
        db.execute(update(TrainingRun).where(TrainingRun.id == run_id).values(**vals))
        db.commit()


def _cancel_requested(run_id: str) -> bool:
    with SessionLocal() as db:
        r = db.get(TrainingRun, run_id)
        return bool(r and r.cancel_requested)


def run_trainer(run_id: str, argv: list[str], work: Path, base_dir: Path) -> tuple[str, str]:
    """Chạy subprocess, ghi log đã che secret, lưu loss thật. Trả (kết_quả, ghi_chú):
    ok | cancelled | failed."""
    log_path = work / "train.log"
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1",
           "HF_HUB_CACHE": str(Path(settings.model_cache_dir).resolve()),
           "HF_DATASETS_CACHE": str(work / "hf_datasets"), "TOKENIZERS_PARALLELISM": "false"}
    kw = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    proc = subprocess.Popen(argv, cwd=str(base_dir), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kw)
    q: queue.Queue = queue.Queue()

    def reader():
        buf = b""
        while True:
            chunk = proc.stdout.read(512)
            if not chunk:
                break
            buf += chunk
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for p in parts:
                if p.strip():
                    q.put(p.decode("utf-8", errors="replace"))
        if buf.strip():
            q.put(buf.decode("utf-8", errors="replace"))
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()
    tail: list[str] = []
    last_step, last_check, last_logged_step = 0, 0.0, -10
    with open(log_path, "a", encoding="utf-8") as log:
        while True:
            try:
                line = q.get(timeout=2)
            except queue.Empty:
                line = ""
            if line is None:
                break
            if line:
                m = _TQDM.search(line)
                if m:
                    step, loss, lr = int(m.group(1)), float(m.group(3)), float(m.group(4))
                    if step != last_step:
                        last_step = step
                        with SessionLocal() as db:
                            exists = db.scalar(select(TrainingMetric.id).where(TrainingMetric.run_id == run_id,
                                                                               TrainingMetric.step == step))
                            if not exists:
                                db.add(TrainingMetric(run_id=run_id, step=step, loss=loss, lr=lr))
                            db.execute(update(TrainingRun).where(TrainingRun.id == run_id)
                                       .values(step=step, heartbeat_at=now(), phase="training"))
                            db.commit()
                        if step - last_logged_step >= 5:
                            log.write(mask(line) + "\n")
                            log.flush()
                            last_logged_step = step
                else:
                    clean = mask(line)
                    log.write(clean + "\n")
                    log.flush()
                    tail = (tail + [clean])[-40:]
            if time.time() - last_check > 3:
                last_check = time.time()
                if _cancel_requested(run_id):
                    kill_tree(proc)
                    proc.wait()
                    return "cancelled", ""
                _set(run_id, heartbeat_at=now())
    code = proc.wait()
    if code == 0:
        return "ok", ""
    text = "\n".join(tail)
    if "out of memory" in text.lower():
        return "failed", ("cuda_oom: hết VRAM khi train. Giảm resolution (512), tăng gradient_accumulation_steps, "
                          "giảm rank, hoặc dùng GPU/worker đủ VRAM. Không tự retry với cùng cấu hình.")
    return "failed", f"trainer_exit_{code}: " + (tail[-1] if tail else "không có log")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def register_lora(run_id: str, work: Path) -> str:
    """Sao chép LoRA ra kho model, tính checksum, tạo ModelVersion ở trạng thái awaiting_review."""
    src = work / "out" / "pytorch_lora_weights.safetensors"
    if not src.exists():
        raise FileNotFoundError("trainer không tạo pytorch_lora_weights.safetensors")
    with SessionLocal() as db:
        run = db.get(TrainingRun, run_id)
        base = db.get(ModelVersion, run.base_model_version_id)
        from app.models import DatasetVersion, Dataset
        dsv = db.get(DatasetVersion, run.dataset_version_id)
        ds = db.get(Dataset, dsv.dataset_id)
        losses = [m.loss for m in db.scalars(select(TrainingMetric).where(TrainingMetric.run_id == run_id)
                                             .order_by(TrainingMetric.step))]
        mv = ModelVersion(
            kind="character_lora" if run.objective == "character" else "style_lora",
            name=f"LoRA {run.trigger_token} ({run.id[:8]})", family=base.family, model_id=None,
            revision=run.id, license=base.license, status="awaiting_review",
            base_model_version_id=base.id, training_run_id=run.id, character_id=ds.character_id,
            trigger_token=run.trigger_token,
            metrics={"steps": run.max_steps, "final_loss": losses[-1] if losses else None,
                     "mean_last_10_loss": sum(losses[-10:]) / len(losses[-10:]) if losses else None,
                     "note": "Loss thấp không chứng minh chất lượng nhận dạng; cần duyệt bằng mắt."},
        )
        db.add(mv)
        db.flush()
        dest_dir = Path(settings.model_cache_dir).resolve() / "loras"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{mv.id}.safetensors"
        shutil.copy2(src, dest)
        mv.artifact_path = str(dest)
        mv.sha256 = sha256_file(dest)
        db.commit()
        return mv.id
