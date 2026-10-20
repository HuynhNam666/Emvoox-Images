"""Test luồng video. ComfyUI được thay bằng máy chủ HTTP giả (CHỈ để kiểm tra logic client/worker/API);
chúng KHÔNG chứng minh LTX-2.3 + PromptRelay chạy được — việc đó phải thử trên ComfyUI thật."""
import json
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import app.workers.tasks as T
from app.core.config import settings
from app.db.session import SessionLocal
from app.ml import comfy_client as C
from app.models import Asset, VideoAttempt
from tests.conftest import make_image

MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64


class FakeComfy:
    def __init__(self):
        self.prompts, self.uploads, self.freed = [], 0, 0
        self.missing: set[str] = set()
        self.fail_exec = False
        self.never_done = False
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj, code=200, raw=None):
                body = raw if raw is not None else json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.startswith("/object_info"):
                    graph = json.load(open(C.workflow_dir() / "ltx23_promptrelay.api.json", encoding="utf8"))
                    # Schema THẬT của ComfyUI 0.42 + PromptRelay + KJNodes + VHS (trích từ /object_info), chỉ chèn thêm file model giả lập
                    real = json.load(open(Path(__file__).parent / "fixtures" / "comfy_object_info.json", encoding="utf8"))
                    oi = {k: v for k, v in real.items() if k not in fake.missing}
                    nodes = [n for n in graph.values() if isinstance(n, dict) and "class_type" in n]
                    for cls, inputs in C.MODEL_INPUTS.items():
                        for inp in inputs:
                            if cls in oi:
                                oi[cls]["input"]["required"][inp] = [sorted({m["inputs"][inp] for m in nodes if m["class_type"] == cls})]
                    return self._send(oi)
                if self.path.startswith("/history/"):
                    pid = self.path.rsplit("/", 1)[1]
                    if fake.never_done:
                        return self._send({})
                    if fake.fail_exec:
                        return self._send({pid: {"status": {"status_str": "error", "messages": [["execution_error", {"exception_message": "boom"}]]}}})
                    return self._send({pid: {"status": {"status_str": "success", "completed": True},
                                             "outputs": {"20": {"gifs": [{"filename": "v.mp4", "subfolder": "emvoox", "type": "output"}]}}}})
                if self.path.startswith("/view"):
                    return self._send(None, raw=MP4)
                self._send({}, 404)

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(n)
                if self.path == "/upload/image":
                    fake.uploads += 1
                    return self._send({"name": "up.png", "subfolder": "", "type": "input"})
                if self.path == "/prompt":
                    fake.prompts.append(json.loads(body)["prompt"])
                    return self._send({"prompt_id": f"pid-{len(fake.prompts)}", "node_errors": {}})
                if self.path == "/free":
                    fake.freed += 1
                    return self._send({})
                self._send({}, 404)

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_port}"

    def close(self):
        self.srv.shutdown()


@pytest.fixture()
def comfy(monkeypatch):
    f = FakeComfy()
    monkeypatch.setattr(settings, "comfyui_url", f.url)
    monkeypatch.setattr(settings, "comfyui_poll_s", 0.01)
    monkeypatch.setattr(T, "release_pipeline", lambda: None)  # không nạp SDXL trong test
    yield f
    f.close()


SEGS = [{"start": 0, "end": 25, "prompt": "He looks forward and breathes naturally."},
        {"start": 25, "end": 65, "prompt": "He slowly turns his head toward the doorway."},
        {"start": 65, "end": 100, "prompt": "He pauses and looks concerned."}]


def _approved_keyframe(client) -> tuple[str, str]:
    """Ảnh nguồn đi đúng đường thật: job ảnh -> review approved."""
    jid = client.post("/api/v1/generation-jobs", json={"prompt_original": "x"}).json()["job_id"]
    with SessionLocal() as db:
        from app.models import AssetBlob, GenerationJob, GenerationOutput
        data = make_image(5)
        a = Asset(kind="generated", mime_type="image/png", width=640, height=640, byte_size=len(data), sha256="a" * 64)
        db.add(a); db.flush()
        db.add(AssetBlob(asset_id=a.id, data=data))
        db.add(GenerationOutput(job_id=jid, asset_id=a.id, output_index=0, seed=1))
        db.get(GenerationJob, jid).state = "review_required"
        db.commit()
        return jid, a.id


def _body(asset_id, **kw):
    return {"source_keyframe_asset_id": asset_id, "global_prompt": "Cinematic medium shot, same young man.",
            "motion_segments": SEGS, **kw}


# ---------- thuần logic ----------
def test_segment_and_param_validation():
    assert C.compile_local_prompts(C.validate_segments(SEGS), "pipe_labeled").startswith("[0-25] He looks forward")
    assert "[25-65]" in C.compile_local_prompts(SEGS, "pipe_labeled") and " | " in C.compile_local_prompts(SEGS, "pipe_labeled")
    assert C.compile_local_prompts(SEGS, "pipe_plain").count("|") == 2
    assert C.segment_frame_lengths(SEGS, 97) == [24, 39, 34] or sum(C.segment_frame_lengths(SEGS, 97)) == 97
    with pytest.raises(ValueError):
        C.validate_segments([{"start": 0, "end": 50, "prompt": "a"}, {"start": 60, "end": 100, "prompt": "b"}])  # hở
    with pytest.raises(ValueError):
        C.validate_segments([{"start": 0, "end": 90, "prompt": "a"}])  # không phủ tới 100
    with pytest.raises(ValueError):
        C.validate_segments([{"start": 0, "end": 100, "prompt": "a | b"}])  # ký tự dành riêng
    C.validate_video_params(512, 896, 97, 25)
    for bad in [(500, 896, 97, 25), (512, 896, 100, 25), (512, 896, 97, 5)]:
        with pytest.raises(ValueError):
            C.validate_video_params(*bad)


def test_workflow_builds_from_mapping():
    wf = C.load_workflow()
    assert wf["verified"] and wf["status"] == "schema_validated"  # đã khớp schema ComfyUI nhưng chưa render thật
    g = C.build_prompt(wf, image_name="up.png", width=512, height=896, frame_count=97, fps=25, seed=7,
                       global_prompt="G", segments=SEGS, audio_mode="generated", filename_prefix="p")
    assert g["5"]["inputs"]["image"] == "up.png" and g["16"]["inputs"]["noise_seed"] == 7
    assert g["7"]["inputs"]["length"] == g["9"]["inputs"]["frames_number"] == 97
    assert g["11"]["inputs"]["local_prompts"].startswith("He looks forward") and g["11"]["inputs"]["local_prompts"].count("|") == 2
    lens = [int(x) for x in g["11"]["inputs"]["segment_lengths"].split(",")]
    assert sum(lens) == 97 and len(lens) == 3 and lens[1] > lens[0]   # 25/40/35 % của 97 frame
    assert "audio" in g["20"]["inputs"]
    g2 = C.build_prompt(wf, image_name="u", width=512, height=896, frame_count=49, fps=25, seed=1,
                        global_prompt="G", segments=SEGS, audio_mode="none", filename_prefix="p")
    assert "audio" not in g2["20"]["inputs"]
    assert wf["graph"]["20"]["inputs"]["audio"] == ["19", 0]  # graph gốc không bị sửa


# ---------- API ----------
def test_unapproved_keyframe_blocked(client):
    up = client.post("/api/v1/assets", files=[("file", ("a.png", make_image(2), "image/png"))]).json()
    r = client.post("/api/v1/video/jobs", json=_body(up["id"]))
    assert r.status_code == 409 and "chưa được duyệt" in r.json()["detail"]
    assert not client.sent


def test_validation_errors_and_audio_asset(client):
    _, aid = _approved_keyframe(client)
    assert client.post("/api/v1/video/jobs", json=_body(aid, frame_count=100)).status_code == 422
    assert client.post("/api/v1/video/jobs", json=_body(aid, width=500)).status_code == 422
    bad = [{"start": 0, "end": 50, "prompt": "a"}]
    assert client.post("/api/v1/video/jobs", json=_body(aid, motion_segments=bad)).status_code == 422
    assert client.post("/api/v1/video/jobs", json=_body(aid, audio_mode="asset")).status_code == 422  # tuyến Image+Audio chưa hỗ trợ


def test_full_flow_idempotency_provenance_and_review(client, comfy):
    jid, aid = _approved_keyframe(client)
    assert client.post(f"/api/v1/generation-jobs/{jid}/review", json={"decision": "approved"}).json()["state"] == "approved"
    h = {"Idempotency-Key": "v1"}
    r = client.post("/api/v1/video/jobs", json=_body(aid), headers=h)
    assert r.status_code == 202
    vid, att = r.json()["job_id"], r.json()["attempt_id"]
    assert client.post("/api/v1/video/jobs", json=_body(aid), headers=h).json()["job_id"] == vid
    assert client.post("/api/v1/video/jobs", json=_body(aid, seed=9), headers=h).status_code == 409
    vsent = [x for x in client.sent if x[0] == "app.workers.tasks.generate_video"]
    assert len(vsent) == 1  # chỉ enqueue video một lần

    assert T.generate_video.run(vid, att) == "review_required"
    assert T.generate_video.run(vid, att) == "skipped"  # giao lại task không render trùng
    assert len(comfy.prompts) == 1 and comfy.uploads == 1 and comfy.freed >= 1
    sent = comfy.prompts[0]
    assert sent["11"]["inputs"]["local_prompts"].count("|") == 2 and sent["11"]["inputs"]["segment_lengths"].count(",") == 2

    job = client.get(f"/api/v1/video/jobs/{vid}").json()
    assert job["state"] == "review_required" and job["output_asset_id"] is None
    a = job["attempts"][0]
    assert a["state"] == "rendered" and a["prompt_id"] == "pid-1" and a["cost"] > 0
    with SessionLocal() as db:
        asset = db.get(Asset, a["output_asset_id"])
        assert asset.mime_type == "video/mp4" and asset.kind == "generated_video"
        assert asset.provenance["source_keyframe_asset_id"] == aid
        assert asset.provenance["source_generation_job_id"] == jid
        assert asset.provenance["workflow_verified"] is True  # khớp schema; "verified" thật chỉ khi đã render clip
    full = client.get(f"/api/v1/assets/{a['output_asset_id']}/content")
    assert full.content == MP4 and full.headers["accept-ranges"] == "bytes"
    part = client.get(f"/api/v1/assets/{a['output_asset_id']}/content", headers={"Range": "bytes=4-7"})
    assert part.status_code == 206 and part.content == MP4[4:8] and part.headers["content-range"] == f"bytes 4-7/{len(MP4)}"
    assert client.get(f"/api/v1/assets/{a['output_asset_id']}/thumbnail").status_code == 415

    ok = client.post(f"/api/v1/video/jobs/{vid}/review", json={"decision": "approve", "scores": {"identity": 5}}).json()
    assert ok["state"] == "approved" and ok["output_asset_id"] == a["output_asset_id"]
    m = client.get("/api/v1/video/metrics").json()
    assert m["approved"] == 1 and m["approval_rate"] == 1.0 and m["cost_per_approved"] > 0


def test_reject_routes_and_repair_cap(client, comfy, monkeypatch):
    jid, aid = _approved_keyframe(client)
    client.post(f"/api/v1/generation-jobs/{jid}/review", json={"decision": "approved"})

    # Sai mặt do ảnh nguồn => quay về công đoạn ảnh, không retry video
    vid = client.post("/api/v1/video/jobs", json=_body(aid)).json()
    T.generate_video.run(vid["job_id"], vid["attempt_id"])
    assert client.post(f"/api/v1/video/jobs/{vid['job_id']}/review", json={"decision": "reject"}).status_code == 422
    r = client.post(f"/api/v1/video/jobs/{vid['job_id']}/review", json={"decision": "reject", "reason": "keyframe_identity"}).json()
    assert r["state"] == "returned_to_image"
    assert client.post(f"/api/v1/video/jobs/{vid['job_id']}/retry", json={}).status_code == 409

    # Mặt đổi khi chuyển động => sửa tuyến video, tối đa 2 lượt rồi needs_human
    v2 = client.post("/api/v1/video/jobs", json=_body(aid, seed=2)).json()
    T.generate_video.run(v2["job_id"], v2["attempt_id"])
    for n in (1, 2):
        assert client.post(f"/api/v1/video/jobs/{v2['job_id']}/review", json={"decision": "reject", "reason": "motion_identity"}).json()["state"] == "rejected"
        rt = client.post(f"/api/v1/video/jobs/{v2['job_id']}/retry", json={"seed": 100 + n})
        assert rt.status_code == 202 and rt.json()["repair_count"] == n
        T.generate_video.run(v2["job_id"], rt.json()["attempt_id"])
    client.post(f"/api/v1/video/jobs/{v2['job_id']}/review", json={"decision": "reject", "reason": "motion_identity"})
    cap = client.post(f"/api/v1/video/jobs/{v2['job_id']}/retry", json={})
    assert cap.status_code == 409 and "repair_limit" in cap.json()["detail"]
    j = client.get(f"/api/v1/video/jobs/{v2['job_id']}").json()
    assert j["state"] == "needs_human" and len(j["attempts"]) == 3 and j["attempts"][2]["kind"] == "repair"
    assert j["attempts"][2]["params"]["seed"] == 102
    m = client.get("/api/v1/video/metrics").json()
    assert m["reject_reasons"]["motion_identity"] == 3 and m["reject_reasons"]["keyframe_identity"] == 1


def test_failures_recorded_with_cost(client, comfy):
    jid, aid = _approved_keyframe(client)
    client.post(f"/api/v1/generation-jobs/{jid}/review", json={"decision": "approved"})

    comfy.missing = {"PromptRelayEncode"}
    v = client.post("/api/v1/video/jobs", json=_body(aid)).json()
    assert T.generate_video.run(v["job_id"], v["attempt_id"]) == "failed"
    j = client.get(f"/api/v1/video/jobs/{v['job_id']}").json()
    assert j["state"] == "failed" and "comfy_nodes_missing" in j["error"] and "PromptRelayEncode" in j["error"]
    assert j["cost_spent"] > 0 and not comfy.prompts  # chưa gửi prompt khi thiếu node

    comfy.missing = set()
    comfy.fail_exec = True
    v = client.post("/api/v1/video/jobs", json=_body(aid, seed=3)).json()
    assert T.generate_video.run(v["job_id"], v["attempt_id"]) == "failed"
    j = client.get(f"/api/v1/video/jobs/{v['job_id']}").json()
    assert "comfy_execution_error" in j["error"] and j["attempts"][0]["prompt_id"] == "pid-1"
    comfy.fail_exec = False
    rt = client.post(f"/api/v1/video/jobs/{v['job_id']}/retry", json={})  # lỗi thực thi cũng retry được (tính 1 lượt sửa)
    assert rt.status_code == 202
    assert T.generate_video.run(v["job_id"], rt.json()["attempt_id"]) == "review_required"


def test_comfyui_unreachable_is_explicit(client, monkeypatch):
    monkeypatch.setattr(settings, "comfyui_url", "http://127.0.0.1:1")
    monkeypatch.setattr(T, "release_pipeline", lambda: None)
    jid, aid = _approved_keyframe(client)
    client.post(f"/api/v1/generation-jobs/{jid}/review", json={"decision": "approved"})
    v = client.post("/api/v1/video/jobs", json=_body(aid)).json()
    assert T.generate_video.run(v["job_id"], v["attempt_id"]) == "failed"
    assert "comfy_unreachable" in client.get(f"/api/v1/video/jobs/{v['job_id']}").json()["error"]
    assert client.get("/api/v1/video/health").json()["ok"] is False


def test_timeout_keeps_prompt_id_and_resume_does_not_resubmit(client, comfy, monkeypatch):
    jid, aid = _approved_keyframe(client)
    client.post(f"/api/v1/generation-jobs/{jid}/review", json={"decision": "approved"})
    v = client.post("/api/v1/video/jobs", json=_body(aid)).json()
    comfy.never_done = True
    monkeypatch.setattr(settings, "comfyui_timeout_s", 0.1)
    assert T.generate_video.run(v["job_id"], v["attempt_id"]) == "failed"
    att = client.get(f"/api/v1/video/jobs/{v['job_id']}").json()["attempts"][0]
    assert "comfy_timeout" in att["error"] and att["prompt_id"] == "pid-1"

    # worker chết giữa chừng: attempt kẹt 'running' có prompt_id => nối lại bằng /history, không gửi prompt mới
    with SessionLocal() as db:
        db.get(VideoAttempt, att["id"]).state = "running"
        db.commit()
    comfy.never_done = False
    assert T.generate_video.run(v["job_id"], att["id"]) == "review_required"
    assert len(comfy.prompts) == 1 and comfy.uploads == 1


def test_budget_blocks(client, comfy):
    jid, aid = _approved_keyframe(client)
    client.post(f"/api/v1/generation-jobs/{jid}/review", json={"decision": "approved"})
    v = client.post("/api/v1/video/jobs", json=_body(aid, budget_limit=0.0001)).json()
    T.generate_video.run(v["job_id"], v["attempt_id"])  # lượt đầu chạy (chi phí > trần)
    client.post(f"/api/v1/video/jobs/{v['job_id']}/review", json={"decision": "reject", "reason": "wrong_action"})
    r = client.post(f"/api/v1/video/jobs/{v['job_id']}/retry", json={})
    assert r.status_code == 409 and "budget_exhausted" in r.json()["detail"]
    assert client.get(f"/api/v1/video/jobs/{v['job_id']}").json()["state"] == "needs_human"


def test_health_reports_unverified_workflow(client, comfy):
    h = client.get("/api/v1/video/health").json()
    assert h["missing_nodes"] == [] and h["workflow_verified"] is True and h["workflow_status"] == "schema_validated" and h["ok"] is True and "CHƯA có clip" in h["warning"]
    comfy.missing = {"VHS_VideoCombine"}
    assert client.get("/api/v1/video/health").json()["missing_nodes"] == ["VHS_VideoCombine"]


def test_idempotent_replay_survives_later_changes(client, comfy):
    jid, aid = _approved_keyframe(client)
    client.post(f"/api/v1/generation-jobs/{jid}/review", json={"decision": "approved"})
    h = {"Idempotency-Key": "replay"}
    first = client.post("/api/v1/video/jobs", json=_body(aid), headers=h).json()["job_id"]
    client.post(f"/api/v1/generation-jobs/{jid}/review", json={"decision": "rejected"})  # ảnh bị loại sau đó
    again = client.post("/api/v1/video/jobs", json=_body(aid), headers=h)
    assert again.status_code == 202 and again.json()["job_id"] == first  # trả job cũ, không báo 409 "chưa duyệt"
    assert client.post("/api/v1/video/jobs", json=_body(aid, seed=5)).status_code == 409  # job mới thì bị chặn


def test_rejected_image_job_blocks_new_video(client):
    jid, aid = _approved_keyframe(client)
    client.post(f"/api/v1/generation-jobs/{jid}/review", json={"decision": "rejected"})
    assert client.post("/api/v1/video/jobs", json=_body(aid)).status_code == 409


def test_workflow_change_midflight_is_not_run_silently(client, comfy, monkeypatch):
    jid, aid = _approved_keyframe(client)
    client.post(f"/api/v1/generation-jobs/{jid}/review", json={"decision": "approved"})
    v = client.post("/api/v1/video/jobs", json=_body(aid)).json()
    real = C.load_workflow

    def changed(name=None):
        wf = real(name)
        wf["version"] += "-edited"
        return wf

    monkeypatch.setattr(C, "load_workflow", changed)
    assert T.generate_video.run(v["job_id"], v["attempt_id"]) == "failed"
    j = client.get(f"/api/v1/video/jobs/{v['job_id']}").json()
    assert "workflow_changed" in j["error"] and not comfy.prompts
    rt = client.post(f"/api/v1/video/jobs/{v['job_id']}/retry", json={})  # sửa lại => ghim phiên bản mới
    assert rt.status_code == 202
    assert T.generate_video.run(v["job_id"], rt.json()["attempt_id"]) == "review_required"
    assert client.get(f"/api/v1/video/jobs/{v['job_id']}").json()["workflow_version"].endswith("-edited")


def test_double_review_and_retry_only_once(client, comfy):
    jid, aid = _approved_keyframe(client)
    client.post(f"/api/v1/generation-jobs/{jid}/review", json={"decision": "approved"})
    v = client.post("/api/v1/video/jobs", json=_body(aid)).json()
    T.generate_video.run(v["job_id"], v["attempt_id"])
    assert client.post(f"/api/v1/video/jobs/{v['job_id']}/review", json={"decision": "approve"}).status_code == 200
    assert client.post(f"/api/v1/video/jobs/{v['job_id']}/review", json={"decision": "reject", "reason": "other"}).status_code == 409
    assert client.post(f"/api/v1/video/jobs/{v['job_id']}/retry", json={}).status_code == 409  # đã duyệt: không sửa


def test_enqueue_failure_does_not_burn_repair(client, comfy, monkeypatch):
    from app.workers.celery_app import celery_app
    jid, aid = _approved_keyframe(client)
    client.post(f"/api/v1/generation-jobs/{jid}/review", json={"decision": "approved"})
    v = client.post("/api/v1/video/jobs", json=_body(aid)).json()
    T.generate_video.run(v["job_id"], v["attempt_id"])
    client.post(f"/api/v1/video/jobs/{v['job_id']}/review", json={"decision": "reject", "reason": "other"})

    def boom(*a, **k):
        raise RuntimeError("redis down")

    monkeypatch.setattr(celery_app, "send_task", boom)
    assert client.post(f"/api/v1/video/jobs/{v['job_id']}/retry", json={}).status_code == 503
    assert client.get(f"/api/v1/video/jobs/{v['job_id']}").json()["repair_count"] == 0
