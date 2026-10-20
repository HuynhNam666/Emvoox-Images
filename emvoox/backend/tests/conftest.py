import io
import os

import pytest
from PIL import Image

os.environ.setdefault("IMAGE_PROVIDER", "mock")
os.environ.setdefault("DATA_DIR", "/tmp/emvoox-test")
os.environ.setdefault("MIN_IMAGE_SIDE", "64")

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.db import engine  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def schema():
    with engine.begin() as c:
        c.execute(text("drop schema public cascade; create schema public;"))
    command.upgrade(Config("alembic.ini"), "head")
    yield


@pytest.fixture(autouse=True)
def fake_redis(monkeypatch):
    """GPU lease talks to Redis; tests use an in-memory fake unless TEST_REAL_REDIS=1."""
    if os.environ.get("TEST_REAL_REDIS") != "1":
        import fakeredis
        import redis

        server = fakeredis.FakeServer()
        monkeypatch.setattr(redis.Redis, "from_url", classmethod(lambda cls, *a, **k: fakeredis.FakeRedis(server=server)))


@pytest.fixture()
def client(monkeypatch):
    from app.main import app, seed_registry
    import app.services.jobs as jobs

    sent = []
    # No broker in unit/integration tests: capture dispatches instead of calling Celery.
    monkeypatch.setattr(jobs, "dispatch_job", lambda db, jid: sent.append(("gen", jid)))
    import app.api.generation as gen
    monkeypatch.setattr(gen, "dispatch_job", lambda db, jid: sent.append(("gen", jid)))
    seed_registry()
    c = TestClient(app)
    c.sent = sent
    return c


def png_bytes(size=(128, 128), color=(200, 30, 30), noise=0) -> bytes:
    img = Image.new("RGB", size, color)
    px = img.load()
    for i in range(noise):
        px[(i * 7) % size[0], (i * 13) % size[1]] = ((i * 31) % 255, (i * 17) % 255, (i * 5) % 255)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()
