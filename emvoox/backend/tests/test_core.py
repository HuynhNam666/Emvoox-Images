import hashlib
import io
import uuid
import zipfile

from sqlalchemy import func

from app.db import SessionLocal
from app.models import AssetBlob, DatasetItem, DatasetVersionItem, GenerationJob, GenerationOutput, ModelVersion
from app.workers.tasks import execute_generation
from tests.conftest import png_bytes

V = "/api/v1"


def _base(client) -> str:
    return next(m["id"] for m in client.get(f"{V}/models?kind=base").json() if m["name"] == "SDXL 1.0 base")


def _job(client, key=None, **kw):
    body = {"prompt_original": "test", "model_version_id": _base(client), "num_images": 2, "seed": 7, **kw}
    h = {"Idempotency-Key": key} if key else {}
    return client.post(f"{V}/generation-jobs", json=body, headers=h)


def test_idempotency_same_and_different_payload(client):
    k = f"k-{uuid.uuid4()}"
    a = _job(client, k)
    b = _job(client, k)
    assert a.status_code == 202 and b.json()["id"] == a.json()["id"]
    c = _job(client, k, seed=8)
    assert c.status_code == 409 and c.json()["detail"]["code"] == "idempotency_conflict"


def test_generation_persists_bytes_and_is_idempotent_on_redelivery(client):
    job_id = uuid.UUID(_job(client).json()["id"])
    assert execute_generation(job_id) == "review_required"
    assert execute_generation(job_id) == "review_required"  # redelivered task: no duplicates
    j = client.get(f"{V}/generation-jobs/{job_id}").json()
    assert len(j["outputs"]) == 2
    out = j["outputs"][0]
    r = client.get(f"{V}/assets/{out['asset_id']}/content")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    meta = client.get(f"{V}/assets/{out['asset_id']}").json()
    assert hashlib.sha256(r.content).hexdigest() == meta["sha256"]
    with SessionLocal() as db:  # bytes really are in PostgreSQL
        assert db.get(AssetBlob, uuid.UUID(out["asset_id"])).data == r.content
        n = db.query(func.count(GenerationOutput.id)).filter_by(job_id=job_id).scalar()
        assert n == 2
    assert meta["provenance"]["ai_generated"] is True


def test_asset_list_has_no_bytes(client):
    r = client.get(f"{V}/assets?limit=5")
    assert r.status_code == 200
    for item in r.json()["items"]:
        assert "data" not in item and "thumbnail" not in item


def test_export_preserves_aspect_by_cropping(client):
    job_id = uuid.UUID(_job(client).json()["id"])
    execute_generation(job_id)
    aid = client.get(f"{V}/generation-jobs/{job_id}").json()["outputs"][0]["asset_id"]
    r = client.get(f"{V}/assets/{aid}/export?width=1080&height=1920")
    from PIL import Image
    assert Image.open(io.BytesIO(r.content)).size == (1080, 1920)


def test_cancel_before_start(client):
    job_id = _job(client).json()["id"]
    r = client.post(f"{V}/generation-jobs/{job_id}/cancel")
    assert r.json()["state"] == "cancelled"
    assert execute_generation(uuid.UUID(job_id)) == "cancelled"
    assert client.post(f"{V}/generation-jobs/{job_id}/cancel").status_code == 409


def test_incompatible_or_unapproved_lora_rejected(client):
    base_id = _base(client)
    with SessionLocal() as db:
        other = ModelVersion(kind="base", name="other", family="flux", status="experimental")
        db.add(other)
        db.flush()
        bad = ModelVersion(kind="character_lora", name="bad", family="flux", parent_base_id=other.id,
                           review_status="approved", artifact_key="x")
        pending = ModelVersion(kind="character_lora", name="pend", family="sdxl", parent_base_id=uuid.UUID(base_id),
                               review_status="awaiting_review", artifact_key="x")
        db.add_all([bad, pending])
        db.commit()
        bad_id, pend_id = str(bad.id), str(pending.id)
    r = _job(client, lora_version_id=bad_id)
    assert r.status_code == 422 and r.json()["detail"]["code"] == "incompatible_lora"
    r = _job(client, lora_version_id=pend_id)
    assert r.status_code == 422 and r.json()["detail"]["code"] == "lora_not_approved"


def _make_zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, d in entries.items():
            z.writestr(n, d)
    return buf.getvalue()


def _dataset(client) -> str:
    return client.post(f"{V}/datasets", json={"name": f"ds-{uuid.uuid4()}", "trigger_token": "a01char"}).json()["id"]


def test_upload_rejects_zip_slip_and_bad_files_and_requires_license(client):
    ds = _dataset(client)
    z = _make_zip({"../evil.png": png_bytes(noise=50), "ok.png": png_bytes(noise=60), "notes.exe": b"MZ",
                   "bad.png": b"not an image"})
    r = client.post(f"{V}/datasets/{ds}/uploads", files=[("files", ("d.zip", z))], data={})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "license_required"
    r = client.post(f"{V}/datasets/{ds}/uploads", files=[("files", ("d.zip", z))], data={"license": "own work"})
    rep = r.json()
    codes = {x["code"] for x in rep["rejected"]}
    assert rep["added"] == 1
    assert {"zip_slip", "corrupt_image"} <= codes or {"zip_slip", "unsupported_file"} <= codes


def test_exact_duplicate_detected(client):
    ds = _dataset(client)
    img = png_bytes(noise=80)
    r = client.post(f"{V}/datasets/{ds}/uploads", files=[("files", ("a.png", img)), ("files", ("b.png", img))],
                    data={"license": "own"})
    assert r.json()["added"] == 1 and r.json()["duplicates"] == 1


def _fill(client, ds: str, n: int, near_dup_pairs: int = 0) -> list[str]:
    files = []
    for i in range(n):
        files.append(("files", (f"i{i}.png", png_bytes(size=(96, 96), color=(i * 11 % 255, i * 7 % 255, 50), noise=40 + i * 3))))
        files.append(("files", (f"i{i}.txt", f"a01char, scene {i}".encode())))
    client.post(f"{V}/datasets/{ds}/uploads", files=files, data={"license": "own"})
    items = client.get(f"{V}/datasets/{ds}/items?limit=500").json()["items"]
    client.post(f"{V}/datasets/{ds}/items/bulk", json={"item_ids": [i["id"] for i in items], "patch": {"status": "approved"}})
    return [i["id"] for i in items]


def test_version_snapshot_is_immutable_and_has_splits(client):
    ds = _dataset(client)
    ids = _fill(client, ds, 24)
    v = client.post(f"{V}/datasets/{ds}/versions")
    assert v.status_code == 201, v.text
    vid = v.json()["id"]
    before = client.get(f"{V}/dataset-versions/{vid}/manifest").json()
    assert {i["split"] for i in before["items"]} >= {"train", "val", "test"}
    client.patch(f"{V}/dataset-items/{ids[0]}", json={"caption": "CHANGED AFTER SNAPSHOT"})
    after = client.get(f"{V}/dataset-versions/{vid}/manifest").json()
    assert before == after
    assert not any(i["caption"] == "CHANGED AFTER SNAPSHOT" for i in after["items"])


def test_near_duplicates_never_straddle_splits(client):
    ds = _dataset(client)
    base = png_bytes(size=(96, 96), color=(10, 200, 30), noise=90)
    from PIL import Image
    files = []
    for k in range(10):  # 10 near-identical variants of the same picture (re-encoded + 1px tweak)
        im = Image.open(io.BytesIO(base)).convert("RGB")
        im.putpixel((k, 0), (k * 20 % 255, 0, 0))
        buf = io.BytesIO()
        im.save(buf, "PNG")
        files.append(("files", (f"n{k}.png", buf.getvalue())))
    client.post(f"{V}/datasets/{ds}/uploads", files=files, data={"license": "own"})
    ids = _fill(client, ds, 16)  # plus distinct images so val/test splits exist
    v = client.post(f"{V}/datasets/{ds}/versions").json()
    manifest = client.get(f"{V}/dataset-versions/{v['id']}/manifest").json()["items"]
    group = [i for i in manifest if i["caption"] == ""]  # the 10 near-identical uploads have no caption file
    assert len(group) == 10
    assert len({i["split"] for i in group}) == 1


def test_prompt_compiler_does_not_fake_translation(client):
    r = client.post(f"{V}/prompt/compile", json={"fields": {"hair": "toc_den_dai", "outfit": "áo dài trắng",
                                                          "style": "dien_anh"}, "trigger_token": "a01char"}).json()
    assert "long black hair" in r["prompt"] and "áo" not in r["prompt"]
    assert r["warnings"]


def test_readiness_reports_components(client):
    r = client.get(f"{V}/health/ready")
    assert "gpu_worker" in r.json() and "database" in r.json()


def test_trainer_progress_parser_and_argv():
    from pathlib import Path
    from app.ml import trainer

    line = "Steps:  10%|#         | 60/600 [01:02<09:30,  1.05s/it, loss=0.123, lr=0.0001]"
    assert trainer.parse_progress(line) == (60, 600, 0.123, 0.0001)
    assert trainer.parse_progress("loading pipeline...") is None
    cfg = {**trainer.PRESETS["sdxl-8gb-explore"]}
    argv = trainer.build_argv(cfg, Path("/d"), Path("/o"), "a01char", "stabilityai/x", None)
    assert isinstance(argv, list) and "--gradient_checkpointing" in argv and "--train_text_encoder" not in argv
    assert argv[argv.index("--rank") + 1] == "16"
