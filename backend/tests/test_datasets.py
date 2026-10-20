import io
import zipfile

from PIL import Image

from tests.conftest import make_image


def _files(*pairs):
    return [("files", (n, d, "image/png")) for n, d in pairs]


def test_upload_report_dedupe_near_dup_lowres_junk(client):
    did = client.post("/api/v1/datasets", json={"name": "t", "group": "identity"}).json()["id"]
    assert client.post("/api/v1/datasets", json={"name": "x", "group": "bogus"}).status_code == 422
    a, b = make_image(1), make_image(2)
    near = io.BytesIO()
    Image.open(io.BytesIO(a)).resize((641, 641)).save(near, "PNG")
    r = client.post(f"/api/v1/datasets/{did}/uploads", files=_files(
        ("a.png", a), ("b.png", b), ("a_copy.png", a), ("a_near.png", near.getvalue()),
        ("small.png", make_image(9, (100, 100))), ("junk.png", b"not an image"))).json()
    assert r["imported"] == 4  # a, b, a_near, small
    assert len(r["duplicates"]) == 1 and len(r["near_duplicates"]) == 1
    assert r["low_resolution"] == ["small.png"] and r["rejected"][0]["file"] == "junk.png"


def test_zip_caption_and_malicious_zip(client):
    did = client.post("/api/v1/datasets", json={"name": "t", "group": "scene"}).json()["id"]
    z = io.BytesIO()
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("set/d.png", make_image(4))
        zf.writestr("set/d.txt", "trigger, smiling")
    ok = client.post(f"/api/v1/datasets/{did}/uploads", files=[("files", ("s.zip", z.getvalue(), "application/zip"))]).json()
    assert ok["imported"] == 1
    items = client.get(f"/api/v1/datasets/{did}/items").json()["items"]
    assert items[0]["caption"] == "trigger, smiling"

    bad = io.BytesIO()
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../../evil.png", make_image(5))
    r = client.post(f"/api/v1/datasets/{did}/uploads", files=[("files", ("bad.zip", bad.getvalue(), "application/zip"))]).json()
    assert r["imported"] == 0 and "không an toàn" in r["rejected"][0]["reason"]

    sl = io.BytesIO()
    with zipfile.ZipFile(sl, "w") as zf:
        zi = zipfile.ZipInfo("link.png")
        zi.external_attr = 0o120777 << 16
        zf.writestr(zi, "/etc/passwd")
    r = client.post(f"/api/v1/datasets/{did}/uploads", files=[("files", ("sl.zip", sl.getvalue(), "application/zip"))]).json()
    assert "symlink" in r["rejected"][0]["reason"]


def test_versions_are_immutable_and_enforce_near_dup_split(client):
    did = client.post("/api/v1/datasets", json={"name": "t", "group": "identity"}).json()["id"]
    a, b = make_image(1), make_image(2)
    near = io.BytesIO()
    Image.open(io.BytesIO(a)).resize((641, 641)).save(near, "PNG")
    client.post(f"/api/v1/datasets/{did}/uploads", files=_files(("a.png", a), ("b.png", b), ("a_near.png", near.getvalue())))
    items = {i["filename"]: i for i in client.get(f"/api/v1/datasets/{did}/items").json()["items"]}
    assert client.post(f"/api/v1/datasets/{did}/versions", json={}).status_code == 422  # chưa duyệt ảnh nào
    ids = [i["id"] for i in items.values()]
    client.post(f"/api/v1/datasets/{did}/items/bulk", json={"item_ids": ids, "status": "approved", "caption_prepend": "emv_x"})
    client.patch(f"/api/v1/dataset-items/{items['a_near.png']['id']}", json={"split": "val"})
    conflict = client.post(f"/api/v1/datasets/{did}/versions", json={})
    assert conflict.status_code == 409 and conflict.json()["detail"]["code"] == "near_duplicate_split_conflict"
    v1 = client.post(f"/api/v1/datasets/{did}/versions", json={"harmonize_splits": True}).json()
    manifest = client.get(f"/api/v1/dataset-versions/{v1['id']}/manifest").text
    client.patch(f"/api/v1/dataset-items/{items['b.png']['id']}", json={"caption": "CHANGED"})
    assert client.get(f"/api/v1/dataset-versions/{v1['id']}/manifest").text == manifest and "CHANGED" not in manifest
    v2 = client.post(f"/api/v1/datasets/{did}/versions", json={}).json()
    assert v2["version"] == 2 and v2["manifest_hash"] != v1["manifest_hash"]


def _locked_dataset(client, license_):
    did = client.post("/api/v1/datasets", json={"name": f"ds-{license_}", "group": "identity", "license": license_}).json()["id"]
    client.post(f"/api/v1/datasets/{did}/uploads", files=_files(("a.png", make_image(11)), ("b.png", make_image(12)), ("c.png", make_image(13))))
    items = client.get(f"/api/v1/datasets/{did}/items").json()["items"]
    client.post(f"/api/v1/datasets/{did}/items/bulk", json={"item_ids": [i["id"] for i in items], "status": "approved"})
    return client.post(f"/api/v1/datasets/{did}/versions", json={}).json()["id"]


def test_commercial_use_blocks_noncommercial_and_unknown_datasets(client, monkeypatch):
    from app.core.config import settings
    from app.db.session import SessionLocal
    from app.models import ModelVersion
    with SessionLocal() as db:
        m = ModelVersion(kind="base", name="b", family="sdxl", model_id="x/y", status="verified", trainable=True)
        db.add(m)
        db.commit()
        mid = m.id
    cfg = {"resolution": 512, "rank": 16, "learning_rate": 1e-4, "max_train_steps": 10, "train_batch_size": 1, "gradient_accumulation_steps": 1,
           "checkpointing_steps": 5, "mixed_precision": "fp16", "gradient_checkpointing": True, "use_vae_fp16_fix": True, "seed": 1}

    def rights(license_):
        v = _locked_dataset(client, license_)
        r = client.post("/api/v1/training-runs/preflight", json={"dataset_version_id": v, "base_model_version_id": mid,
                                                                  "trigger_token": "emv_x", "objective": "character", "config": cfg}).json()
        return next(c for c in r["checks"] if c["name"] == "rights")

    monkeypatch.setattr(settings, "commercial_use", False)
    assert rights("CC BY-NC-SA 4.0")["status"] == "warn"                          # nội bộ: chỉ cảnh báo
    monkeypatch.setattr(settings, "commercial_use", True)
    nc, unk, ok = rights("CC BY-NC-SA 4.0 (non-commercial)"), rights(""), rights("Unsplash Dataset Lite terms")
    assert nc["status"] == "fail" and "COMMERCIAL_USE" in nc["message"]
    assert unk["status"] == "fail" and "chưa ghi license" in unk["message"]
    assert ok["status"] == "ok"
    from app.dataset_rights import rights_status
    assert rights_status("CC0", None, 5, 0)[0] == "ok" and rights_status("Research only", None, 5, 0)[0] == "noncommercial"
    assert rights_status("", None, 4, 4)[0] == "ok"                                # toàn ảnh AI tự sinh
