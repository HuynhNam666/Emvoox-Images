import hashlib

import app.workers.tasks as T
from app.db.session import SessionLocal
from app.models import Asset, AssetBlob, ModelVersion
from tests.conftest import make_image

BODY = {"prompt_original": "cô gái", "num_images": 2}


def _fake_provider(p, n=1):
    return [(p.seed + i, make_image(100 + i), 640, 640) for i in range(n)]


def test_idempotency_and_blob_persistence(client, monkeypatch):
    h = {"Idempotency-Key": "k1"}
    jid = client.post("/api/v1/generation-jobs", json=BODY, headers=h)
    assert jid.status_code == 202
    jid = jid.json()["job_id"]
    assert client.post("/api/v1/generation-jobs", json=BODY, headers=h).json()["job_id"] == jid
    assert client.post("/api/v1/generation-jobs", json={**BODY, "seed": 7}, headers=h).status_code == 409
    assert len(client.sent) == 1  # chỉ enqueue một lần

    monkeypatch.setattr(T, "generate_images", _fake_provider)  # CHỈ trong test: thay SDXL thật
    assert T.generate.run(jid) == "review_required"
    assert T.generate.run(jid) == "skipped"  # giao lại task không sinh trùng
    job = client.get(f"/api/v1/generation-jobs/{jid}").json()
    assert job["state"] == "review_required" and len(job["outputs"]) == 2
    aid = job["outputs"][0]["asset_id"]
    raw = client.get(f"/api/v1/assets/{aid}/content").content
    with SessionLocal() as db:
        assert db.get(AssetBlob, aid).data == raw and db.get(Asset, aid).sha256 == hashlib.sha256(raw).hexdigest()


def test_provider_unavailable_marks_failed_not_mocked(client, monkeypatch):
    import app.ml.sdxl as sdxl

    def no_gpu():
        raise sdxl.ProviderUnavailable("Không có CUDA GPU")  # mô phỏng máy không GPU (tránh nạp SDXL thật trong test)

    monkeypatch.setattr(sdxl, "_load", no_gpu)
    jid = client.post("/api/v1/generation-jobs", json=BODY).json()["job_id"]
    assert T.generate.run(jid) == "failed"  # không có provider: phải thất bại rõ ràng, không giả ảnh
    assert "provider_unavailable" in client.get(f"/api/v1/generation-jobs/{jid}").json()["error"]


def test_character_versions_and_references(client):
    c = client.post("/api/v1/characters", json={"code": "T1", "name": "t"}).json()
    assert client.post("/api/v1/characters", json={"code": "T1", "name": "dup"}).status_code == 409
    up = client.post("/api/v1/assets", files=[("file", ("a.png", make_image(1), "image/png"))]).json()
    r = client.post(f"/api/v1/characters/{c['id']}/references", json={"asset_id": up["id"]}).json()
    assert r["current_version"]["version"] == 2 and len(r["versions"]) == 2 and r["versions"][1]["references"] == []
    assert client.post(f"/api/v1/characters/{c['id']}/references", json={"asset_id": up["id"]}).status_code == 409
    assert client.post(f"/api/v1/characters/{c['id']}/references", json={"asset_id": "nope"}).status_code == 422


def test_lora_gates(client):
    a = client.post("/api/v1/characters", json={"code": "A", "name": "a"}).json()
    b = client.post("/api/v1/characters", json={"code": "B", "name": "b"}).json()
    with SessionLocal() as db:
        mv = ModelVersion(kind="character_lora", name="l", family="sdxl", status="awaiting_review", character_id=a["id"])
        db.add(mv)
        db.commit()
        mv_id = mv.id
    base = {**BODY, "lora_version_id": mv_id}
    assert client.post("/api/v1/generation-jobs", json=base).status_code == 422  # chưa duyệt, thiếu evaluation
    cross = {**base, "evaluation": True, "character_version_id": b["current_version"]["id"]}
    assert "nhân vật khác" in client.post("/api/v1/generation-jobs", json=cross).json()["detail"]
    ok = {**base, "evaluation": True, "character_version_id": a["current_version"]["id"]}
    assert client.post("/api/v1/generation-jobs", json=ok).status_code == 202


def test_use_references_requires_adapter_and_refs(client):
    c = client.post("/api/v1/characters", json={"code": "R", "name": "r"}).json()
    body = {**BODY, "use_references": True, "character_version_id": c["current_version"]["id"]}
    assert client.post("/api/v1/generation-jobs", json=body).status_code == 503  # chưa seed ip_adapter
    with SessionLocal() as db:
        db.add(ModelVersion(kind="ip_adapter", name="ip", family="sdxl", status="experimental"))
        db.commit()
    assert client.post("/api/v1/generation-jobs", json=body).status_code == 422  # nhân vật chưa có ảnh chuẩn


def test_history_lists_jobs_with_outputs_and_filters(client, monkeypatch):
    c = client.post("/api/v1/characters", json={"code": "H1", "name": "h"}).json()
    cv = c["current_version"]["id"]
    j1 = client.post("/api/v1/generation-jobs", json={**BODY, "character_version_id": cv}).json()["job_id"]
    import time
    time.sleep(1.1)  # SQLite chỉ phân giải theo giây; Postgres có micro-giây
    j2 = client.post("/api/v1/generation-jobs", json=BODY).json()["job_id"]
    monkeypatch.setattr(T, "generate_images", _fake_provider)
    T.generate.run(j1)
    r = client.get("/api/v1/generation-jobs").json()
    assert r["total"] == 2 and r["items"][0]["id"] == j2  # mới nhất trước
    first = next(i for i in r["items"] if i["id"] == j1)
    assert first["state"] == "review_required" and len(first["outputs"]) == 2
    assert first["character"]["code"] == "H1" and first["config"]["num_images"] == 2
    assert client.get("/api/v1/generation-jobs?state=queued").json()["total"] == 1
    assert client.get(f"/api/v1/generation-jobs?character_id={c['id']}").json()["total"] == 1
    assert len(client.get("/api/v1/generation-jobs?limit=1").json()["items"]) == 1
