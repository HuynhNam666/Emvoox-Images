"""Adapter from a validated training config to the pinned Diffusers SDXL DreamBooth-LoRA script."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Asset, AssetBlob, DatasetVersionItem

PRESETS = {
    # Exploratory values, not benchmarked. 8 GB cards: batch 1 + checkpointing + 512px first.
    "sdxl-8gb-explore": {
        "rank": 16, "learning_rate": 1e-4, "max_train_steps": 600, "resolution": 512, "train_batch_size": 1,
        "gradient_accumulation_steps": 4, "mixed_precision": "fp16", "gradient_checkpointing": True,
        "train_text_encoder": False, "checkpointing_steps": 200, "lr_scheduler": "constant", "lr_warmup_steps": 0,
        "use_8bit_adam": False, "seed": 42,
    },
    "sdxl-24gb-explore": {
        "rank": 16, "learning_rate": 1e-4, "max_train_steps": 800, "resolution": 1024, "train_batch_size": 1,
        "gradient_accumulation_steps": 2, "mixed_precision": "bf16", "gradient_checkpointing": True,
        "train_text_encoder": False, "checkpointing_steps": 200, "lr_scheduler": "constant", "lr_warmup_steps": 0,
        "use_8bit_adam": False, "seed": 42,
    },
}

# Every flag we emit must appear in the pinned script's --help (checked at preflight).
BOOL_FLAGS = {"gradient_checkpointing", "train_text_encoder", "use_8bit_adam"}
VALUE_FLAGS = ["rank", "learning_rate", "max_train_steps", "resolution", "train_batch_size",
               "gradient_accumulation_steps", "mixed_precision", "checkpointing_steps", "lr_scheduler",
               "lr_warmup_steps", "seed"]
STATIC_FLAGS = ["pretrained_model_name_or_path", "dataset_name", "caption_column", "image_column", "output_dir",
                "instance_prompt", "cache_dir", "revision"]

LOSS_RE = re.compile(r"(\d+)/(\d+).*?loss=([0-9.eE+-]+)(?:.*?lr=([0-9.eE+-]+))?")


def required_flags(config: dict) -> list[str]:
    return STATIC_FLAGS + VALUE_FLAGS + [f for f in BOOL_FLAGS if config.get(f)]


def check_script_flags(config: dict) -> list[str]:
    """Return flags our argv uses that the pinned script does not advertise."""
    script = get_settings().trainer_script
    if not Path(script).exists():
        return [f"trainer_script_missing:{script}"]
    help_text = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True, timeout=120).stdout
    return [f for f in required_flags(config) if f"--{f}" not in help_text]


def materialize_dataset(db: Session, dataset_version_id, target: Path) -> int:
    """Export train-split images + metadata.jsonl (imagefolder layout) to local disk."""
    target.mkdir(parents=True, exist_ok=True)
    rows = (
        db.query(DatasetVersionItem, Asset, AssetBlob)
        .join(Asset, Asset.id == DatasetVersionItem.asset_id)
        .join(AssetBlob, AssetBlob.asset_id == Asset.id)
        .filter(DatasetVersionItem.dataset_version_id == dataset_version_id, DatasetVersionItem.split == "train")
        .all()
    )
    ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
    with open(target / "metadata.jsonl", "w", encoding="utf-8") as meta:
        for item, asset, blob in rows:
            name = f"{item.sha256[:16]}{ext[asset.mime_type]}"
            (target / name).write_bytes(blob.data)
            meta.write(json.dumps({"file_name": name, "text": item.caption}, ensure_ascii=False) + "\n")
    return len(rows)


def build_argv(config: dict, dataset_dir: Path, output_dir: Path, trigger_prompt: str, base_id: str,
               base_revision: str | None) -> list[str]:
    s = get_settings()
    argv = [
        sys.executable, s.trainer_script,
        "--pretrained_model_name_or_path", base_id,
        "--dataset_name", str(dataset_dir), "--caption_column", "text", "--image_column", "image",
        "--instance_prompt", trigger_prompt, "--output_dir", str(output_dir),
        # --report_to is left at the script default (tensorboard): the pinned script forwards the value to
        # accelerate, which rejects "none".
        "--cache_dir", str(Path(s.data_dir) / "hf-cache"),
    ]
    if base_revision:
        argv += ["--revision", base_revision]
    for f in VALUE_FLAGS:
        argv += [f"--{f}", str(config[f])]
    for f in BOOL_FLAGS:
        if config.get(f):
            argv.append(f"--{f}")
    return argv


def parse_progress(line: str) -> tuple[int, int, float, float | None] | None:
    m = LOSS_RE.search(line)
    if not m:
        return None
    return int(m[1]), int(m[2]), float(m[3]), float(m[4]) if m[4] else None


def run_dir(run_id) -> Path:
    return Path(get_settings().data_dir) / "runs" / str(run_id)


def cleanup_materialized(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
