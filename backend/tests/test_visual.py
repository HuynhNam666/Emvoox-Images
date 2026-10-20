"""Test pipeline ảnh (Model Router, Quality Gate, vòng sửa, failover, ngân sách, duyệt, xuất bản, benchmark, báo cáo, auth).

Provider tạo ảnh và FaceNet được THAY bằng đối tượng giả CHỈ để kiểm tra logic điều phối; test không chứng minh chất lượng ảnh
hay kết nối API thật. Riêng C2PA và nhãn AI chạy thật (thư viện c2pa + chứng thư thử nghiệm sinh tại chỗ).
"""
import io
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

import app.visual.pipeline as P
from app.core.config import settings
from app.db.session import SessionLocal
from app.models import SystemStatus
from app.visual import qc as QC
from app.visual.providers.base import GenResult, ImageOut, ProviderError

SIZE = (576, 1024)          # đúng 9:16, cạnh ngắn >= 512


def img(good: bool, seed: int = 0, size=SIZE) -> bytes:
    """Ảnh có nhiễu (không 'trống', đủ nét). Điểm ảnh (0,0): đỏ 200 = 'mặt đúng người', 50 = 'sai người' (FakeFace đọc điểm này)."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    a[0, 0] = (200 if good else 50, 10, 10)
    b = io.BytesIO()
    Image.fromarray(a).save(b, "PNG")
    return b.getvalue()


class FakeFace:
    def embeds(self, image):
        r = image.getpixel((0, 0))[0]
        v = np.zeros(8)
        v[0 if r > 128 else 1] = 1.0
        return [v]


class FakeProvider:
    def __init__(self, key, route, script, cost=0.0, fail=None):
        self.key, self.route, self.script, self.cost, self.fail = key, route, script, cost, fail
        self.calls = 0
        self.prompts = []

    def available(self):
        return True, "ok"

    def generate(self, req, hooks):
        self.calls += 1
        self.prompts.append(req.prompt)
        if self.fail:
            raise self.fail
        good = self.script[min(self.calls - 1, len(self.script) - 1)]
        outs = [ImageOut(img(good, req.seed + i), req.seed + i, *SIZE, {}) for i in range(req.n)]
        return GenResult(outs, self.cost * req.n if self.route == "api" else 0.0, [], {"steps": 4, "infer_s": 0.01})


@pytest.fixture()
def world(client, monkeypatch):
    """Worker 48GB + khóa API giả + FaceNet giả + provider giả (có thể đổi từng test)."""
    with SessionLocal() as db:
        db.merge(SystemStatus(key="worker", value={"cuda": True, "total_gb": 48.0, "name": "FakeGPU"}))
        db.commit()
    for k in ("bfl_api_key", "gemini_api_key", "seedream_api_key", "openai_api_key"):
        monkeypatch.setattr(settings, k, "k")
    QC._embed_cache.clear()
    monkeypatch.setattr(QC, "face_engine", lambda: FakeFace())
    w = type("W", (), {})()
    w.providers = {}

    def fake_get(m):
        return w.providers[m.key]
    monkeypatch.setattr(P, "get_provider", fake_get)
    import app.workers.tasks as T
    monkeypatch.setattr(T, "report_worker_status", lambda: {})   # không ghi đè trạng thái worker giả bằng GPU thật của máy test
    w.client = client
    return w


def make_char(client, code="T1", with_ref=True):
    c = client.post("/api/v1/characters", json={"code": code, "name": code, "identity": {
        "description_vi": "x", "identity_en": f"a young woman {code} with short black hair", "negative_prompt": "blurry"}}).json()
    if with_ref:
        up = client.post("/api/v1/assets", files=[("file", ("r.png", img(True, 99, (640, 640)), "image/png"))]).json()
        c = client.post(f"/api/v1/characters/{c['id']}/references", json={"asset_id": up["id"]}).json()
    return c


def body(c, **kw):
    b = {"project_id": "PILOT", "episode_id": "E01", "shot_id": "S01",
         "characters": [{"character_version_id": c["current_version"]["id"]}],
         "scene_spec": {"action": "looks back over her shoulder", "emotion": "calm", "camera": {"size": "medium shot", "angle": "eye level"},
                        "lighting": "soft window light from the left"}, "num_variants": 2}
    b.update(kw)
    return b


def run(world, jid):
    from app.workers.tasks import visual_run
    return visual_run.run(jid)


def job(world, jid):
    return world.client.get(f"/api/v1/visual/jobs/{jid}").json()


# ---------------------------------------------------------------- registry + router
def test_registry_has_every_model_in_the_report(world):
    r = world.client.get("/api/v1/visual/models").json()
    keys = {m["key"] for m in r["models"]}
    assert {"sdxl-1.0", "flux2-klein-4b", "flux2-klein-base-4b", "flux2-pro", "flux3-image", "gemini-nano-banana", "seedream-4-5",
            "gpt-image", "qwen-image", "qwen-image-edit", "z-image-turbo", "midjourney"} <= keys
    by = {m["key"]: m for m in r["models"]}
    assert by["midjourney"]["available"] is False and by["midjourney"]["route"] == "manual"
    assert by["flux3-image"]["available"] is False and "xác minh" in by["flux3-image"]["availability"]   # endpoint chưa xác minh
    assert by["flux3-image"]["verification"] == "unverified"
    assert by["flux2-klein-4b"]["license"] == "Apache-2.0" and by["flux2-klein-4b"]["available"] is True
    assert by["gemini-nano-banana"]["verification"] == "connector_unverified"


def test_router_excludes_unconfigured_and_incapable(world, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "seedream_api_key", "")
    c = make_char(world.client)
    r = world.client.post("/api/v1/visual/jobs", json=body(c))
    assert r.status_code == 202, r.text
    plan = [p["model_key"] for p in r.json()["route_plan"]]
    ex = {e["model_key"]: e["reason"] for e in r.json()["excluded"]}
    assert "z-image-turbo" in ex and "giữ được nhân vật" in ex["z-image-turbo"]       # không có tham chiếu danh tính
    assert "qwen-image" in ex
    assert "gemini-nano-banana" in ex and "khóa" in ex["gemini-nano-banana"]
    assert "midjourney" in ex and "midjourney" not in plan
    assert plan[0] in ("sdxl-1.0", "flux2-klein-4b", "flux2-klein-4b-api")                # tuyến rẻ, đủ năng lực
    assert len({world.client.get("/api/v1/visual/models").json()["models"][0]["route"]}) == 1


def test_small_gpu_removes_big_local_models(world):
    with SessionLocal() as db:
        db.merge(SystemStatus(key="worker", value={"cuda": True, "total_gb": 8.0}))
        db.commit()
    c = make_char(world.client)
    r = world.client.post("/api/v1/visual/jobs", json=body(c, model_key="qwen-image-edit"))
    assert r.status_code == 422 and "GPU" in r.json()["detail"]
    r = world.client.post("/api/v1/visual/jobs", json=body(c, model_key="sdxl-1.0"))
    assert r.status_code == 202


def test_two_characters_need_multi_ref(world):
    a, b = make_char(world.client, "T1"), make_char(world.client, "T2")
    req = body(a)
    req["characters"].append({"character_version_id": b["current_version"]["id"], "position": "right"})
    r = world.client.post("/api/v1/visual/jobs", json=req)
    assert r.status_code == 202
    assert r.json()["task"] == "multi_person"
    assert "sdxl-1.0" not in [p["model_key"] for p in r.json()["route_plan"]]
    assert any(e["model_key"] == "sdxl-1.0" and "đa tham chiếu" in e["reason"] for e in r.json()["excluded"])


def test_prompt_compiler_order_and_roles(world):
    c = make_char(world.client)
    with SessionLocal() as db:
        from app.models import Character
        ch = db.query(Character).filter_by(code="T1").one()
    world.client.post(f"/api/v1/characters/{ch.id}/costumes", json={"code": "WS", "description_en": "a white shirt"})
    sc = world.client.post("/api/v1/scenes", json={"code": "B02", "name": "Corridor B02", "spec": {
        "layout": "narrow corridor", "time_of_day": "morning", "light_direction": "camera left", "props": ["bench"]}})
    assert sc.status_code in (201, 409)
    req = body(c, scene_code="B02", model_key="flux2-klein-4b")
    req["characters"][0]["costume_code"] = "WS"
    r = world.client.post("/api/v1/visual/jobs", json=req)
    assert r.status_code == 202, r.text
    j = job(world, r.json()["job_id"])
    p = j["prompt"]
    order = ["Reference images", "short black hair", "looks back", "Wearing: a white shirt", "Setting:", "medium shot", "window light", "Preserve:"]
    pos = [p.index(k) for k in order]
    assert pos == sorted(pos), p                                  # nhận dạng → hành động → trang phục → bối cảnh → cỡ cảnh → ánh sáng → giữ
    assert "Image 1: face reference of T1" in p
    assert j["pins"]["scene_version_id"] and j["scene"]["code"] == "B02"


# ---------------------------------------------------------------- vòng điều phối
def test_happy_path_review_and_approve(world):
    c = make_char(world.client)
    world.providers["sdxl-1.0"] = FakeProvider("sdxl-1.0", "local", [True])
    r = world.client.post("/api/v1/visual/jobs", json=body(c, model_key="sdxl-1.0"), headers={"Idempotency-Key": "a1"})
    jid = r.json()["job_id"]
    assert world.client.post("/api/v1/visual/jobs", json=body(c, model_key="sdxl-1.0"), headers={"Idempotency-Key": "a1"}).json()["replayed"]
    assert world.client.post("/api/v1/visual/jobs", json=body(c, model_key="sdxl-1.0", seed=5), headers={"Idempotency-Key": "a1"}).status_code == 409
    assert len([x for x in world.client.sent if x[0] == "app.workers.tasks.visual_run"]) == 1
    run(world, jid)
    j = job(world, jid)
    assert j["state"] == "review_required" and len(j["candidates"]) == 2
    assert all(cd["qc_verdict"] == "pass" for cd in j["candidates"])
    ids = {ch["name"] for cd in j["candidates"] for ch in cd["qc"]["checks"]}
    assert {"decode", "min_resolution", "aspect", "not_blank", "sharpness", "exposure", "identity_1", "faces"} <= ids
    a = j["attempts"][0]
    assert a["state"] == "done" and a["metrics"]["wall_s"] >= 0 and a["metrics"]["queue_s"] is not None
    assert run(world, jid) == "skipped"                                                    # giao lại task không chạy lại
    # duyệt: thiếu tên người duyệt bị chặn
    assert world.client.post(f"/api/v1/visual/jobs/{jid}/review", json={"decision": "approve"}).status_code == 422
    ok = world.client.post(f"/api/v1/visual/jobs/{jid}/review", json={"decision": "approve", "reviewer": "Nam", "note": "đẹp"})
    assert ok.status_code == 200 and ok.json()["state"] == "approved" and ok.json()["reviewer"] == "Nam" and ok.json()["reviewed_at"]
    assert ok.json()["review_history"][0]["reviewer"] == "Nam"
    assert world.client.post(f"/api/v1/visual/jobs/{jid}/review", json={"decision": "reject", "reviewer": "X", "reason": "r"}).status_code == 409


def test_qc_failure_triggers_repair_then_passes(world):
    c = make_char(world.client)
    world.providers["sdxl-1.0"] = FakeProvider("sdxl-1.0", "local", [False, True])       # lần 1 sai người, lần 2 đúng
    jid = world.client.post("/api/v1/visual/jobs", json=body(c, model_key="sdxl-1.0")).json()["job_id"]
    run(world, jid)
    j = job(world, jid)
    assert j["state"] == "review_required" and j["repair_count"] == 1
    assert [a["kind"] for a in j["attempts"]] == ["initial", "repair"]
    assert "reference faces" in j["attempts"][1]["instruction"]
    ps = world.providers["sdxl-1.0"].prompts
    assert "Correction:" not in ps[0] and "Correction:" in ps[1] and "reference faces" in ps[1]   # lượt sửa gửi lại bước biên dịch với chỉ dẫn
    sel = next(cd for cd in j["candidates"] if cd["id"] == j["selected_candidate_id"])
    assert sel["qc_verdict"] == "pass"


def test_repair_cap_two_then_needs_human_and_last_repair_switches_model(world):
    c = make_char(world.client)
    world.providers["sdxl-1.0"] = FakeProvider("sdxl-1.0", "local", [False])
    world.providers["flux2-klein-4b"] = FakeProvider("flux2-klein-4b", "local", [False])
    jid = world.client.post("/api/v1/visual/jobs", json=body(c)).json()["job_id"]
    run(world, jid)
    j = job(world, jid)
    assert j["state"] == "needs_human" and j["repair_count"] == 2 and len(j["attempts"]) == 3
    models = [a["model_key"] for a in j["attempts"]]
    assert models[1] == models[0] and models[2] != models[0]                      # lượt sửa cuối đổi sang tuyến dự phòng
    assert "chuyển người xử lý" in j["error"]
    # người duyệt vẫn quyết được: ảnh lỗi QC phải xác nhận có chủ đích
    cid = j["selected_candidate_id"]
    assert world.client.post(f"/api/v1/visual/jobs/{jid}/review", json={"decision": "approve", "reviewer": "Nam", "candidate_id": cid}).status_code == 409
    assert world.client.post(f"/api/v1/visual/jobs/{jid}/review", json={"decision": "approve", "reviewer": "Nam", "candidate_id": cid,
                                                                       "override_qc": True}).json()["review_history"][0]["override_qc"] is True


def test_provider_failure_falls_back_without_using_a_repair(world):
    c = make_char(world.client)
    world.providers["sdxl-1.0"] = FakeProvider("sdxl-1.0", "local", [True], fail=ProviderError("cuda_oom", "hết VRAM"))
    for k in ("flux2-klein-4b", "flux2-klein-base-4b", "flux2-klein-4b-api", "flux2-pro", "gemini-nano-banana", "seedream-4-5", "gpt-image",
              "qwen-image-edit"):
        world.providers[k] = FakeProvider(k, "api" if "api" in k or k in ("flux2-pro", "gemini-nano-banana", "seedream-4-5", "gpt-image") else "local",
                                          [True], cost=0.02)
    jid = world.client.post("/api/v1/visual/jobs", json=body(c, auto_repair=True)).json()["job_id"]
    plan = job(world, jid)["route_plan"]
    first = plan[0]["model_key"]
    world.providers[first] = FakeProvider(first, plan[0]["route"], [True], fail=ProviderError("unavailable", "tạm hỏng"))
    run(world, jid)
    j = job(world, jid)
    assert j["state"] == "review_required" and j["repair_count"] == 0
    assert j["attempts"][0]["state"] == "failed" and "unavailable" in j["attempts"][0]["error"]
    assert j["attempts"][1]["model_key"] != first and j["attempts"][1]["state"] == "done"


def test_all_providers_failing_marks_job_failed_with_reason(world):
    c = make_char(world.client)
    jid = world.client.post("/api/v1/visual/jobs", json=body(c, model_key="sdxl-1.0")).json()["job_id"]
    world.providers["sdxl-1.0"] = FakeProvider("sdxl-1.0", "local", [True], fail=ProviderError("unavailable", "không có CUDA"))
    run(world, jid)
    j = job(world, jid)
    assert j["state"] == "failed" and "không có CUDA" in j["error"] and not j["candidates"]


def test_uncertain_billing_timeout_charges_estimate_and_changes_vendor(world):
    c = make_char(world.client)
    jid = world.client.post("/api/v1/visual/jobs", json=body(c, model_key="gemini-nano-banana")).json()["job_id"]
    world.providers["gemini-nano-banana"] = FakeProvider("gemini-nano-banana", "api", [True],
                                                         fail=ProviderError("timeout", "quá 180s", uncertain_billing=True))
    run(world, jid)
    j = job(world, jid)
    assert j["state"] == "failed" and j["attempts"][0]["cost_api"] > 0 and j["attempts"][0]["metrics"]["uncertain_billing"] is True
    assert j["cost_spent"] > 0                                                      # phí có thể đã tính => vẫn cộng vào job


def test_budget_blocks_before_spending(world):
    c = make_char(world.client)
    r = world.client.post("/api/v1/visual/jobs", json=body(c, model_key="gemini-nano-banana", budget_limit=0.001))
    assert r.status_code == 422 and "ngân sách" in r.json()["detail"]
    world.providers["gemini-nano-banana"] = FakeProvider("gemini-nano-banana", "api", [False], cost=0.03)
    jid = world.client.post("/api/v1/visual/jobs", json=body(c, model_key="gemini-nano-banana", budget_limit=0.08)).json()["job_id"]
    run(world, jid)
    j = job(world, jid)
    # lượt 1 tốn ~0.0672 (2 ảnh x 0.0336 ước tính, thực tế 0.06); lượt sửa sẽ vượt trần 0.08 => dừng, không tiêu thêm
    assert j["state"] == "needs_human" and "budget_exhausted" in j["error"]
    assert j["cost_spent"] <= 0.08 and len(j["attempts"]) == 2 and j["attempts"][1]["state"] == "failed"


def test_human_repair_does_not_use_auto_quota_and_respects_review_rules(world):
    c = make_char(world.client)
    world.providers["sdxl-1.0"] = FakeProvider("sdxl-1.0", "local", [True, True])
    jid = world.client.post("/api/v1/visual/jobs", json=body(c, model_key="sdxl-1.0")).json()["job_id"]
    run(world, jid)
    assert world.client.post(f"/api/v1/visual/jobs/{jid}/review", json={"decision": "repair", "reviewer": "N"}).status_code == 422   # thiếu chỉ dẫn
    r = world.client.post(f"/api/v1/visual/jobs/{jid}/review", json={"decision": "repair", "reviewer": "Nam", "instruction": "brighter eyes"})
    assert r.json()["state"] == "queued"
    run(world, jid)
    j = job(world, jid)
    assert j["state"] == "review_required" and j["repair_count"] == 0
    assert [a["kind"] for a in j["attempts"]] == ["initial", "human_repair"] and j["attempts"][1]["instruction"] == "brighter eyes"
    assert world.client.post(f"/api/v1/visual/jobs/{jid}/review", json={"decision": "reject", "reviewer": "Nam"}).status_code == 422
    assert world.client.post(f"/api/v1/visual/jobs/{jid}/review", json={"decision": "reject", "reviewer": "Nam", "reason": "sai tay"}).json()["state"] == "rejected"


def test_qc_without_face_engine_requires_human_review(world, monkeypatch):
    monkeypatch.setattr(QC, "face_engine", lambda: None)
    monkeypatch.setattr(QC, "_engine_err", "không có FaceNet (test)")
    c = make_char(world.client)
    world.providers["sdxl-1.0"] = FakeProvider("sdxl-1.0", "local", [True])
    jid = world.client.post("/api/v1/visual/jobs", json=body(c, model_key="sdxl-1.0")).json()["job_id"]
    run(world, jid)
    j = job(world, jid)
    assert j["state"] == "review_required"
    assert all(cd["qc_verdict"] == "review" for cd in j["candidates"])                  # không đo được nhận dạng => không tự coi là đạt
    assert any(ch["status"] == "skipped" for ch in j["candidates"][0]["qc"]["checks"])


def test_wrong_aspect_and_blank_fail_qc():
    blank = io.BytesIO()
    Image.new("RGB", SIZE, (120, 120, 120)).save(blank, "PNG")
    q = QC.evaluate(blank.getvalue(), target_aspect=(9, 16), expected_faces=0, ref_embeds=[])
    assert q["verdict"] == "fail" and any(c["name"] == "not_blank" and c["status"] == "fail" for c in q["checks"])
    sq = QC.evaluate(img(True, 1, (640, 640)), target_aspect=(9, 16), expected_faces=0, ref_embeds=[])
    assert any(c["name"] == "aspect" and c["status"] == "fail" for c in sq["checks"])


def test_manual_import_midjourney(world):
    c = make_char(world.client)
    world.providers["sdxl-1.0"] = FakeProvider("sdxl-1.0", "local", [True])
    jid = world.client.post("/api/v1/visual/jobs", json=body(c, model_key="sdxl-1.0")).json()["job_id"]
    run(world, jid)
    files = [("files", ("mj1.png", img(True, 7), "image/png"))]
    r = world.client.post(f"/api/v1/visual/jobs/{jid}/manual-import", files=files, data={"model_key": "midjourney", "cost": "0.05"})
    assert r.status_code == 202
    run(world, jid)
    j = job(world, jid)
    assert j["attempts"][-1]["kind"] == "manual" and j["attempts"][-1]["cost_api"] == 0.05
    assert any(cd["attempt_id"] == j["attempts"][-1]["id"] and cd["qc_verdict"] == "pass" for cd in j["candidates"])
    assert world.client.post(f"/api/v1/visual/jobs/{jid}/manual-import", files=files, data={"model_key": "sdxl-1.0"}).status_code == 404


def test_continuity_warns_on_lighting_and_costume_change(world):
    c = make_char(world.client)
    world.client.post("/api/v1/scenes", json={"code": "B02", "name": "Corridor", "spec": {}})
    world.providers["sdxl-1.0"] = FakeProvider("sdxl-1.0", "local", [True])
    a = world.client.post("/api/v1/visual/jobs", json=body(c, scene_code="B02", model_key="sdxl-1.0", shot_id="S01")).json()["job_id"]
    run(world, a)
    world.client.post(f"/api/v1/visual/jobs/{a}/review", json={"decision": "approve", "reviewer": "Nam"})
    spec = body(c, scene_code="B02", model_key="sdxl-1.0", shot_id="S02")
    spec["scene_spec"]["lighting"] = "hard red neon light"
    b = world.client.post("/api/v1/visual/jobs", json=spec).json()["job_id"]
    w = job(world, b)["continuity_warnings"]
    assert w and "ánh sáng khác" in w[0]


# ---------------------------------------------------------------- video dùng ảnh đã duyệt từ pipeline mới
def test_video_accepts_only_selected_candidate_of_approved_visual_job(world):
    c = make_char(world.client)
    world.providers["sdxl-1.0"] = FakeProvider("sdxl-1.0", "local", [True])
    jid = world.client.post("/api/v1/visual/jobs", json=body(c, model_key="sdxl-1.0")).json()["job_id"]
    run(world, jid)
    j = job(world, jid)
    sel = next(cd for cd in j["candidates"] if cd["id"] == j["selected_candidate_id"])
    other = next(cd for cd in j["candidates"] if cd["id"] != j["selected_candidate_id"])
    vb = {"source_keyframe_asset_id": sel["asset_id"], "global_prompt": "g", "motion_segments": [{"start": 0, "end": 100, "prompt": "walks"}]}
    assert world.client.post("/api/v1/video/jobs", json=vb).status_code == 409                    # chưa duyệt
    world.client.post(f"/api/v1/visual/jobs/{jid}/review", json={"decision": "approve", "reviewer": "Nam"})
    assert world.client.post("/api/v1/video/jobs", json={**vb, "source_keyframe_asset_id": other["asset_id"]}).status_code == 409   # ảnh không được chọn
    ok = world.client.post("/api/v1/video/jobs", json=vb)
    assert ok.status_code == 202
    vj = world.client.get(f"/api/v1/video/jobs/{ok.json()['job_id']}").json()
    assert vj["character_version_id"] == c["current_version"]["id"]                                # ghim phiên bản nhân vật từ job ảnh


# ---------------------------------------------------------------- xuất bản
@pytest.fixture()
def dev_cert(tmp_path, monkeypatch):
    subprocess.run([sys.executable, "scripts/make_dev_c2pa_cert.py", "--out", str(tmp_path)], check=True, capture_output=True)
    monkeypatch.setattr(settings, "c2pa_cert_path", str(tmp_path / "chain.pem"))
    monkeypatch.setattr(settings, "c2pa_key_path", str(tmp_path / "key.pem"))


def _approved_job(world):
    c = make_char(world.client)
    world.providers["sdxl-1.0"] = FakeProvider("sdxl-1.0", "local", [True])
    jid = world.client.post("/api/v1/visual/jobs", json=body(c, model_key="sdxl-1.0")).json()["job_id"]
    run(world, jid)
    world.client.post(f"/api/v1/visual/jobs/{jid}/review", json={"decision": "approve", "reviewer": "Nam"})
    return jid


def test_export_requires_approval(world):
    c = make_char(world.client)
    world.providers["sdxl-1.0"] = FakeProvider("sdxl-1.0", "local", [True])
    jid = world.client.post("/api/v1/visual/jobs", json=body(c, model_key="sdxl-1.0")).json()["job_id"]
    run(world, jid)
    assert world.client.post(f"/api/v1/visual/jobs/{jid}/export", json={}).status_code == 409


def test_export_label_resize_and_c2pa_signed_and_verified(world, dev_cert):
    jid = _approved_job(world)
    assert world.client.post(f"/api/v1/visual/jobs/{jid}/export", json={"label": False}).status_code == 422       # REQUIRE_LABEL
    r = world.client.post(f"/api/v1/visual/jobs/{jid}/export", json={})
    assert r.status_code == 201, r.text
    e = r.json()
    assert e["status"] == "released" and e["target"] == "1080x1920"
    assert e["c2pa"]["signed"] and e["c2pa"]["verified"] and e["c2pa"]["trusted"] is False      # chứng thư thử nghiệm: Valid, chưa Trusted
    assert e["label"]["text"] == "Hình ảnh được tạo bằng AI" and e["label"]["margin_px"] >= 48 and 28 <= e["label"]["font_px"] <= 36
    assert e["provenance"]["source"] == "ai" and e["provenance"]["reviewer"] == "Nam" and "prompt" not in e["provenance"]
    f = world.client.get(f"/api/v1/exports/{e['id']}/file")
    assert f.status_code == 200
    im = Image.open(io.BytesIO(f.content))
    assert im.size == (1080, 1920)
    import c2pa
    import tempfile, os
    p = os.path.join(tempfile.mkdtemp(), "x.png")
    open(p, "wb").write(f.content)
    with c2pa.Reader(p) as rd:                                                                   # đọc lại từ file thật đã tải về
        assert str(rd.get_validation_state()).lower() == "valid"
        assert "trainedAlgorithmicMedia" in rd.json()


def test_export_blocked_when_c2pa_required_but_missing(world, monkeypatch):
    jid = _approved_job(world)
    monkeypatch.setattr(settings, "require_c2pa", True)
    monkeypatch.setattr(settings, "c2pa_cert_path", "")
    e = world.client.post(f"/api/v1/visual/jobs/{jid}/export", json={}).json()
    assert e["status"] == "blocked" and e["export_asset_id"] is None and "C2PA_CERT_PATH" in e["c2pa"]["error"]
    assert world.client.get(f"/api/v1/exports/{e['id']}/file").status_code == 409                # không phát hành khi thiếu dấu bắt buộc


def test_export_without_c2pa_is_released_but_flagged_when_not_required(world, monkeypatch):
    jid = _approved_job(world)
    monkeypatch.setattr(settings, "c2pa_cert_path", "")
    e = world.client.post(f"/api/v1/visual/jobs/{jid}/export", json={}).json()
    assert e["status"] == "released" and e["c2pa"]["signed"] is False        # nhãn hiển thị có, C2PA chưa ký: ghi rõ trong hồ sơ


# ---------------------------------------------------------------- benchmark chấm mù + báo cáo
def test_benchmark_blind_rating_and_registry_feedback(world):
    a, b = make_char(world.client, "A01"), make_char(world.client, "A02")
    assert a and b
    for k in ("sdxl-1.0", "flux2-klein-4b"):
        world.providers[k] = FakeProvider(k, "local", [True])
    # chỉ 5 cảnh nhóm chân dung để test nhanh
    r = world.client.post("/api/v1/benchmark/runs", json={"name": "t", "round_no": 2, "model_keys": ["sdxl-1.0", "flux2-klein-4b"],
                                                          "scene_ids": ["P01", "P02", "T01"], "variants": 2})
    assert r.status_code == 201, r.text
    d = r.json()
    assert d["jobs_created"] == 6 and d["images_expected"] == 12
    with SessionLocal() as db:
        from app.models import VisualJob
        ids = [j.id for j in db.query(VisualJob).filter(VisualJob.benchmark_run_id == d["run_id"])]
    for jid in ids:
        run(world, jid)
    nxt = world.client.get(f"/api/v1/benchmark/runs/{d['run_id']}/next", params={"rater": "r1"}).json()
    assert not nxt["done"] and "model" not in str(nxt).lower().replace("model_key", "") and "sdxl" not in str(nxt) and "klein" not in str(nxt)
    assert nxt["brief"]["group"] and "candidate_id" in nxt
    seen = set()
    for rater, core in (("r1", False), ("r2", False)):
        while True:
            n = world.client.get(f"/api/v1/benchmark/runs/{d['run_id']}/next", params={"rater": rater}).json()
            if n["done"]:
                break
            seen.add(n["candidate_id"])
            s = 5 if rater == "r1" else 4
            rr = world.client.post("/api/v1/benchmark/ratings", json={"run_id": d["run_id"], "candidate_id": n["candidate_id"], "rater": rater,
                                                                      "consistency": s, "scene_fit": s, "anatomy": s, "aesthetics": s, "core_error": core})
            assert rr.status_code == 201
    assert len(seen) == 12
    dup = world.client.post("/api/v1/benchmark/ratings", json={"run_id": d["run_id"], "candidate_id": next(iter(seen)), "rater": "r1",
                                                               "consistency": 3, "scene_fit": 3, "anatomy": 3, "aesthetics": 3})
    assert dup.status_code == 409
    third = world.client.post("/api/v1/benchmark/ratings", json={"run_id": d["run_id"], "candidate_id": next(iter(seen)), "rater": "r3",
                                                                 "consistency": 3, "scene_fit": 3, "anatomy": 3, "aesthetics": 3})
    assert third.status_code == 409                                                           # đã đủ hai người chấm độc lập
    det = world.client.get(f"/api/v1/benchmark/runs/{d['run_id']}").json()
    assert det["images"] == 12 and det["ratings"] == 24
    for m in det["models"]:
        assert m["rated_by_two"] == 6 and m["passed"] == 6 and m["pass_rate"] == 1.0
        assert m["mean_weighted"] == 90.0                      # (5+4)/2=4.5 -> 4.5*0.4+... = 4.5*1.0*20 = 90
    ap = world.client.post(f"/api/v1/benchmark/runs/{d['run_id']}/apply", params={"min_rated": 3}).json()
    assert set(ap["applied"]) == {"sdxl-1.0", "flux2-klein-4b"}
    mods = {m["key"]: m for m in world.client.get("/api/v1/visual/models").json()["models"]}
    assert mods["sdxl-1.0"]["benchmark"]["pass_rate"] == 1.0


def test_benchmark_core_error_is_a_reject_regardless_of_score(world):
    make_char(world.client, "A01")
    world.providers["sdxl-1.0"] = FakeProvider("sdxl-1.0", "local", [True])
    d = world.client.post("/api/v1/benchmark/runs", json={"name": "t", "round_no": 1, "model_keys": ["sdxl-1.0"], "scene_ids": ["P01"], "variants": 1}).json()
    with SessionLocal() as db:
        from app.models import VisualJob
        jid = db.query(VisualJob).filter(VisualJob.benchmark_run_id == d["run_id"]).one().id
    run(world, jid)
    n = world.client.get(f"/api/v1/benchmark/runs/{d['run_id']}/next", params={"rater": "a"}).json()
    for rater, core in (("a", True), ("b", False)):
        world.client.post("/api/v1/benchmark/ratings", json={"run_id": d["run_id"], "candidate_id": n["candidate_id"], "rater": rater,
                                                             "consistency": 5, "scene_fit": 5, "anatomy": 5, "aesthetics": 5, "core_error": core})
    m = world.client.get(f"/api/v1/benchmark/runs/{d['run_id']}").json()["models"][0]
    assert m["passed"] == 0 and m["core_errors"] == 1 and m["mean_weighted"] == 100.0


def test_report_costs_latency_memory_and_formula(world, monkeypatch):
    monkeypatch.setattr(settings, "gpu_cost_per_hour", 3.6)          # 0.001 USD / giây
    monkeypatch.setattr(settings, "qa_cost_per_review", 0.5)
    c = make_char(world.client)
    world.providers["gemini-nano-banana"] = FakeProvider("gemini-nano-banana", "api", [False, True], cost=0.02)
    world.providers["sdxl-1.0"] = FakeProvider("sdxl-1.0", "local", [True])
    j1 = world.client.post("/api/v1/visual/jobs", json=body(c, model_key="gemini-nano-banana")).json()["job_id"]
    run(world, j1)                                                   # lỗi QC -> 1 lượt sửa
    world.client.post(f"/api/v1/visual/jobs/{j1}/review", json={"decision": "approve", "reviewer": "Nam"})
    j2 = world.client.post("/api/v1/visual/jobs", json=body(c, model_key="sdxl-1.0", shot_id="S02")).json()["job_id"]
    run(world, j2)
    world.client.post(f"/api/v1/visual/jobs/{j2}/review", json={"decision": "approve", "reviewer": "Nam"})
    rep = world.client.get("/api/v1/reports/summary").json()["visual"]
    gem = next(m for m in rep["models"] if m["model_key"] == "gemini-nano-banana")
    assert gem["attempts"] == 2 and gem["approved"] == 1
    assert gem["cost"]["c_create"] == pytest.approx(0.04) and gem["cost"]["c_repair"] == pytest.approx(0.04)    # 2 ảnh x 0.02 mỗi lượt
    total = gem["cost"]["c_create"] + gem["cost"]["c_repair"] + gem["cost"]["c_gpu"] + gem["cost"]["c_qa_alloc"] + gem["cost"]["c_storage_alloc"] + gem["cost"]["c_training_alloc"]
    assert gem["cost"]["total"] == pytest.approx(total, abs=1e-4)
    assert gem["cost"]["per_approved"] == pytest.approx(gem["cost"]["total"] / 1, abs=1e-4)
    assert rep["totals"]["reviews"] == 2 and rep["totals"]["c_qa"] == pytest.approx(1.0)
    sd = next(m for m in rep["models"] if m["model_key"] == "sdxl-1.0")
    assert sd["gpu_hours"] > 0 and sd["cost"]["c_gpu"] > 0 and sd["latency_s"]["wall"]["n"] == 1
    assert sd["memory_mb"]["process_rss_peak"]["n"] == 1                       # RAM/RSS thật từ psutil
    md = world.client.get("/api/v1/reports/report.md")
    assert md.status_code == 200 and "Chi phí/ảnh đạt" in md.text and "gemini-nano-banana" in md.text
    csv_ = world.client.get("/api/v1/reports/attempts.csv").text
    assert csv_.splitlines()[0].startswith("job_id,shot_id") and len(csv_.splitlines()) >= 4
    st = world.client.get("/api/v1/system/stats").json()
    assert "ram" in st["api_host"] and st["worker"]["total_gb"] == 48.0


# ---------------------------------------------------------------- auth + migrate
def test_token_auth_and_cookie(client, monkeypatch):
    monkeypatch.setattr(settings, "api_token", "s3cret")
    assert client.get("/api/v1/health/live").status_code == 200
    assert client.get("/api/v1/visual/models").status_code == 401
    assert client.get("/api/v1/auth/status").json() == {"required": True, "authenticated": False}
    assert client.get("/api/v1/visual/models", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/api/v1/visual/models", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.post("/api/v1/auth/login", json={"token": "bad"}).status_code == 401
    assert client.post("/api/v1/auth/login", json={"token": "s3cret"}).status_code == 200
    assert client.get("/api/v1/visual/models").status_code == 200          # cookie HttpOnly được giữ trong phiên
    assert client.get("/api/v1/auth/status").json()["authenticated"] is True


def test_sync_columns_adds_missing_column():
    from sqlalchemy import create_engine, inspect, text
    from app.db import migrate
    eng = create_engine("sqlite://")
    with eng.begin() as c:
        c.execute(text("CREATE TABLE video_attempts (id VARCHAR(36) PRIMARY KEY, job_id VARCHAR(36), attempt_no INTEGER)"))
    added = migrate.sync_columns(eng)
    assert "video_attempts.metrics" in added
    assert "metrics" in {c["name"] for c in inspect(eng).get_columns("video_attempts")}
    assert migrate.sync_columns(eng) == []                                 # chạy lại không đổi gì


def test_pilot_library_seeded_and_idempotent(client):
    from app.models import Costume, Scene
    from app.visual.seed import seed_pilot_library
    client.post("/api/v1/characters", json={"code": "A01", "name": "A01", "identity": {"identity_en": "a young man"}})
    with SessionLocal() as db:
        seed_pilot_library(db)
        seed_pilot_library(db)
        assert db.query(Scene).filter_by(code="B02").count() == 1
        assert db.query(Costume).filter_by(code="WHITE_SHIRT").count() == 1


# ---------------------------------------------------------------- xuất video (ffmpeg thật + C2PA thật)
def _ffmpeg():
    import os
    import shutil
    return os.environ.get("TEST_FFMPEG") or shutil.which("ffmpeg")


@pytest.mark.skipif(not _ffmpeg(), reason="cần ffmpeg (đặt TEST_FFMPEG=đường dẫn nếu không có trong PATH)")
def test_video_export_label_resize_audio_kept_and_c2pa(client, dev_cert, monkeypatch, tmp_path):
    from app.models import VideoJob
    monkeypatch.setattr(settings, "ffmpeg_bin", _ffmpeg())
    mp4 = tmp_path / "src.mp4"
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=512x896:rate=25:duration=1",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                    str(mp4)], check=True)
    data = mp4.read_bytes()
    with SessionLocal() as db:
        key = Asset_(db, "image/png", img(True, 3, (576, 1024)))
        vid = Asset_(db, "video/mp4", data, kind="generated_video")
        j = VideoJob(source_keyframe_asset_id=key, global_prompt="g", motion_segments=[], config={"width": 512, "height": 896, "fps": 25,
                                                                                                    "frame_count": 25, "seed": 1},
                     workflow_version="t", state="approved", output_asset_id=vid)
        db.add(j)
        db.commit()
        jid = j.id
    r = client.post(f"/api/v1/video/jobs/{jid}/export", json={})
    assert r.status_code == 201, r.text
    e = r.json()
    assert e["status"] == "released" and e["target"] == "1080x1920" and e["c2pa"]["signed"] and e["c2pa"]["verified"]
    assert e["label"]["text"] == "Hình ảnh được tạo bằng AI"
    f = client.get(f"/api/v1/exports/{e['id']}/file")
    out = tmp_path / "out.mp4"
    out.write_bytes(f.content)
    probe = subprocess.run([_ffmpeg(), "-i", str(out)], capture_output=True, text=True).stderr
    assert "1080x1920" in probe and "Audio:" in probe                       # đúng khung đích, âm thanh được giữ
    import c2pa
    with c2pa.Reader(str(out)) as rd:
        assert str(rd.get_validation_state()).lower() == "valid"
    # chưa duyệt => không xuất
    with SessionLocal() as db:
        db.get(VideoJob, jid).state = "review_required"
        db.commit()
    assert client.post(f"/api/v1/video/jobs/{jid}/export", json={}).status_code == 409


def Asset_(db, mime, data, kind="generated"):
    import hashlib
    from app.models import Asset, AssetBlob
    a = Asset(kind=kind, mime_type=mime, byte_size=len(data), sha256=hashlib.sha256(data).hexdigest(), width=0, height=0)
    db.add(a)
    db.flush()
    db.add(AssetBlob(asset_id=a.id, data=data))
    db.flush()
    return a.id
