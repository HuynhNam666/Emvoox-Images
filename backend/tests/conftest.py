"""Test dùng SQLite tạm (không cần Postgres/Redis/GPU). Provider SDXL và hàng đợi được thay trong từng test, có nhãn rõ."""
import io
import os
import random
import tempfile

_db = os.path.join(tempfile.mkdtemp(prefix="emvoox-test-"), "test.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_db}"
os.environ["REDIS_URL"] = "redis://127.0.0.1:1/0"  # không bao giờ được dùng

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from app.db.session import Base, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.workers.celery_app import celery_app  # noqa: E402


@pytest.fixture()
def client(monkeypatch):
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    from app.db.session import SessionLocal
    from app.visual.registry import seed_registry
    from app.visual.seed import seed_pilot_library
    with SessionLocal() as db:
        seed_registry(db)
        seed_pilot_library(db)
    sent = []
    monkeypatch.setattr(celery_app, "send_task", lambda *a, **k: sent.append(a))
    c = TestClient(app)
    c.sent = sent
    return c


def make_image(seed: int, size=(640, 640), fmt="PNG") -> bytes:
    r = random.Random(seed)
    im = Image.new("RGB", size)
    d = ImageDraw.Draw(im)
    for _ in range(25):
        x, y = r.randrange(size[0]), r.randrange(size[1])
        d.ellipse([x, y, x + r.randrange(30, 200), y + r.randrange(30, 200)], fill=tuple(r.randrange(256) for _ in range(3)))
    b = io.BytesIO()
    im.save(b, fmt)
    return b.getvalue()
