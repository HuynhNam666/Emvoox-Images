"""Smoke test GPU thật: SDXL sinh 1 ảnh -> lưu PostgreSQL -> tải lại -> so SHA-256.

Chạy từ thư mục backend, cần .env có DATABASE_URL trỏ PostgreSQL thật:
    python scripts/smoke_gpu.py [--steps 20] [--aspect 1:1]
KHÔNG dùng mock; nếu thiếu CUDA/torch thì thất bại rõ ràng.
"""
import argparse
import hashlib
import sys
import time

from fastapi.testclient import TestClient

from app.db.session import Base, SessionLocal, engine
from app.main import app
from app.models import AssetBlob
from app.workers import tasks

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--aspect", default="1:1")
ap.add_argument("--prompt", default="portrait of a young Vietnamese woman, black shoulder-length hair, white shirt, library, window light, cinematic photo")
a = ap.parse_args()

from app.workers.celery_app import celery_app

celery_app.send_task = lambda *a, **k: None  # không cần Redis: task được gọi trực tiếp bên dưới

Base.metadata.create_all(engine)
client = TestClient(app)
body = {"prompt_original": a.prompt, "aspect_ratio": a.aspect, "steps": a.steps, "seed": 42}
job_id = client.post("/api/v1/generation-jobs", json=body).json()["job_id"]
print("job:", job_id)

t = time.time()
print("result:", tasks.generate.run(job_id), f"({time.time() - t:.1f}s gồm tải/nạp model)")
job = client.get(f"/api/v1/generation-jobs/{job_id}").json()
print("state:", job["state"], job["error"] or "")
if job["state"] != "review_required" or not job["outputs"]:
    sys.exit("FAIL: job không thành công")

asset_id = job["outputs"][0]["asset_id"]
raw = client.get(f"/api/v1/assets/{asset_id}/content").content
with SessionLocal() as db:
    stored = db.get(AssetBlob, asset_id).data
assert raw == stored and raw[:4] == b"\x89PNG", "bytes tải lại không khớp"
print("OK asset", asset_id, len(raw), "bytes, sha256", hashlib.sha256(raw).hexdigest())
open("smoke_output.png", "wb").write(raw)
print("đã lưu smoke_output.png để xem bằng mắt")
