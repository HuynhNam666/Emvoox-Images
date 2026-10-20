"""Test sửa/xóa an toàn: không xóa thứ còn được dùng, không để lại tham chiếu treo, ảnh chỉ bị xóa khi không còn ai dùng."""
from app.db.session import SessionLocal
from app.models import Asset, Dataset, DatasetItem, GenerationJob, TrainingRun, VideoJob, VisualJob
from tests.conftest import make_image
from tests.test_visual import FakeProvider, body, job, make_char, run, world  # noqa: F401


def _asset(client, seed=1):
    return client.post("/api/v1/assets", files=[("file", ("a.png", make_image(seed), "image/png"))]).json()["id"]


def test_character_rename_and_delete_rules(client):
    a = client.post("/api/v1/characters", json={"code": "X1", "name": "x"}).json()
    b = client.post("/api/v1/characters", json={"code": "X2", "name": "y"}).json()
    assert client.patch(f"/api/v1/characters/{a['id']}", json={"code": "X2"}).status_code == 409
    assert client.patch(f"/api/v1/characters/{a['id']}", json={"name": "Mới", "code": "X9"}).json()["code"] == "X9"
    aid = _asset(client, 3)
    client.post(f"/api/v1/characters/{a['id']}/references", json={"asset_id": aid})
    did = client.post("/api/v1/datasets", json={"name": "d", "group": "identity", "character_id": a["id"]}).json()["id"]
    r = client.delete(f"/api/v1/characters/{a['id']}")
    assert r.status_code == 409 and "dataset" in r.json()["detail"]                       # dataset gắn nhân vật => chặn
    client.patch(f"/api/v1/datasets/{did}", json={"detach_character": True})
    r = client.delete(f"/api/v1/characters/{a['id']}")
    assert r.status_code == 200 and r.json()["assets_purged"] == 1                        # ảnh chuẩn không còn ai dùng => xóa
    assert client.get(f"/api/v1/assets/{aid}/content").status_code == 404
    assert client.get(f"/api/v1/characters/{a['id']}").status_code == 404
    assert client.delete(f"/api/v1/characters/{b['id']}").status_code == 200
    assert client.delete(f"/api/v1/characters/{b['id']}").status_code == 404


def test_dataset_edit_delete_and_version_immutability(client):
    did = client.post("/api/v1/datasets", json={"name": "d", "group": "identity"}).json()["id"]
    files = [("files", (f"{n}.png", make_image(s), "image/png")) for n, s in (("a", 11), ("b", 12), ("c", 13))]
    client.post(f"/api/v1/datasets/{did}/uploads", files=files)
    items = client.get(f"/api/v1/datasets/{did}/items").json()["items"]
    client.post(f"/api/v1/datasets/{did}/items/bulk", json={"item_ids": [i["id"] for i in items], "status": "approved"})
    vid = client.post(f"/api/v1/datasets/{did}/versions", json={}).json()["id"]
    assert client.patch(f"/api/v1/datasets/{did}", json={"group": "pose"}).status_code == 409        # đã có version khóa
    assert client.patch(f"/api/v1/datasets/{did}", json={"name": "Tên mới", "license": "CC0"}).json()["name"] == "Tên mới"
    # xóa 1 ảnh làm việc: version khóa vẫn giữ ảnh (bất biến) nên asset KHÔNG bị xóa
    r = client.delete(f"/api/v1/dataset-items/{items[0]['id']}")
    assert r.json()["assets_purged"] == 0 and client.get(f"/api/v1/assets/{items[0]['asset_id']}/content").status_code == 200
    with SessionLocal() as db:                                                                    # đã dùng để train => chặn xóa
        db.add(TrainingRun(dataset_version_id=vid, base_model_version_id="x", trigger_token="emv_x", config={}))
        db.commit()
    r = client.delete(f"/api/v1/datasets/{did}")
    assert r.status_code == 409 and "huấn luyện" in r.json()["detail"]
    with SessionLocal() as db:
        db.query(TrainingRun).delete()
        db.commit()
    r = client.delete(f"/api/v1/datasets/{did}")
    assert r.status_code == 200 and r.json()["versions"] == 1 and r.json()["assets_purged"] == 3
    with SessionLocal() as db:
        assert db.query(Dataset).count() == 0 and db.query(DatasetItem).count() == 0


def test_bulk_delete_items_purges_unreferenced_images(client):
    did = client.post("/api/v1/datasets", json={"name": "d", "group": "scene"}).json()["id"]
    client.post(f"/api/v1/datasets/{did}/uploads", files=[("files", (f"{n}.png", make_image(30 + n), "image/png")) for n in range(3)])
    items = client.get(f"/api/v1/datasets/{did}/items").json()["items"]
    r = client.post(f"/api/v1/datasets/{did}/items/delete", json={"item_ids": [i["id"] for i in items[:2]]})
    assert r.json() == {"deleted": 2, "assets_purged": 2}
    assert client.get(f"/api/v1/datasets/{did}/items").json()["total"] == 1


def test_scene_and_costume_edit_delete(world):
    c = make_char(world.client)
    world.client.post("/api/v1/scenes", json={"code": "B05", "name": "Phòng", "spec": {}})
    from app.models import Character
    with SessionLocal() as db:
        cid = db.query(Character).filter_by(code="T1").one().id
    co = world.client.post(f"/api/v1/characters/{cid}/costumes", json={"code": "WS", "description_en": "a white shirt"}).json()
    assert world.client.patch(f"/api/v1/costumes/{co['id']}", json={"description_en": "a blue shirt"}).status_code == 200
    assert world.client.patch("/api/v1/scenes/B05", json={"name": "Phòng khách"}).json()["name"] == "Phòng khách"
    world.providers["sdxl-1.0"] = FakeProvider("sdxl-1.0", "local", [True])
    req = body(c, scene_code="B05", model_key="sdxl-1.0")
    req["characters"][0]["costume_code"] = "WS"
    jid = world.client.post("/api/v1/visual/jobs", json=req).json()["job_id"]
    assert world.client.delete(f"/api/v1/visual/jobs/{jid}").status_code == 409                   # đang chờ => không xóa
    assert world.client.delete("/api/v1/scenes/B05").status_code == 409                           # job dùng bối cảnh/trang phục
    assert world.client.delete(f"/api/v1/costumes/{co['id']}").status_code == 409
    run(world, jid)
    out = world.client.delete(f"/api/v1/visual/jobs/{jid}").json()
    assert out["assets_purged"] == 2                                                              # 2 ứng viên không còn ai dùng
    assert world.client.get(f"/api/v1/visual/jobs/{jid}").status_code == 404
    assert world.client.delete(f"/api/v1/costumes/{co['id']}").status_code == 200
    assert world.client.delete("/api/v1/scenes/B05").status_code == 200
    assert world.client.delete("/api/v1/scenes/B05").status_code == 404


def test_visual_job_used_by_video_cannot_be_deleted_until_video_removed(world):
    c = make_char(world.client)
    world.providers["sdxl-1.0"] = FakeProvider("sdxl-1.0", "local", [True])
    jid = world.client.post("/api/v1/visual/jobs", json=body(c, model_key="sdxl-1.0")).json()["job_id"]
    run(world, jid)
    j = job(world, jid)
    world.client.post(f"/api/v1/visual/jobs/{jid}/review", json={"decision": "approve", "reviewer": "Nam"})
    sel = next(cd for cd in j["candidates"] if cd["id"] == j["selected_candidate_id"])
    vj = world.client.post("/api/v1/video/jobs", json={"source_keyframe_asset_id": sel["asset_id"], "global_prompt": "g",
                                                       "motion_segments": [{"start": 0, "end": 100, "prompt": "walks"}]}).json()["job_id"]
    r = world.client.delete(f"/api/v1/visual/jobs/{jid}")
    assert r.status_code == 409 and "video" in r.json()["detail"]
    assert world.client.delete(f"/api/v1/video/jobs/{vj}").status_code == 409                    # đang chờ render
    with SessionLocal() as db:
        db.get(VideoJob, vj).state = "failed"
        db.commit()
    assert world.client.delete(f"/api/v1/video/jobs/{vj}").status_code == 200
    assert world.client.delete(f"/api/v1/visual/jobs/{jid}").status_code == 200
    with SessionLocal() as db:
        assert db.query(VisualJob).count() == 0 and db.query(Asset).filter_by(kind="generated").count() == 0


def test_generation_job_and_benchmark_delete(world):
    make_char(world.client, "A01")
    world.providers["sdxl-1.0"] = FakeProvider("sdxl-1.0", "local", [True])
    d = world.client.post("/api/v1/benchmark/runs", json={"name": "t", "round_no": 1, "model_keys": ["sdxl-1.0"], "scene_ids": ["P01"], "variants": 1}).json()
    assert world.client.delete(f"/api/v1/benchmark/runs/{d['run_id']}").status_code == 409        # còn job chờ
    with SessionLocal() as db:
        jid = db.query(VisualJob).filter(VisualJob.benchmark_run_id == d["run_id"]).one().id
    run(world, jid)
    r = world.client.delete(f"/api/v1/benchmark/runs/{d['run_id']}")
    assert r.status_code == 200 and r.json()["jobs"] == 1
    assert world.client.get("/api/v1/benchmark/runs").json() == []
    gj = world.client.post("/api/v1/generation-jobs", json={"prompt_original": "x"}).json()["job_id"]
    assert world.client.delete(f"/api/v1/generation-jobs/{gj}").status_code == 409                # đang chờ
    with SessionLocal() as db:
        db.get(GenerationJob, gj).state = "failed"
        db.commit()
    assert world.client.delete(f"/api/v1/generation-jobs/{gj}").status_code == 200
