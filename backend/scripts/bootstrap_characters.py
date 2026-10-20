"""Tạo lại hai nhân vật pilot A01/A02 trên server mới: hồ sơ nhận dạng chi tiết → ảnh neo → ảnh chuẩn.

  # Bước 1: cập nhật hồ sơ nhận dạng và sinh 4 ảnh neo mỗi nhân vật (cần model ảnh đã tải)
  python scripts/bootstrap_characters.py anchors --api http://localhost:8000 --token "$API_TOKEN"

  # Bước 2: NGƯỜI chọn ảnh neo ưng ý bằng mắt (tab "Ảnh theo shot"), rồi chốt:
  python scripts/bootstrap_characters.py finalize --api ... --token ... --reviewer Nam --pick A01=<candidate_id> --pick A02=<candidate_id>

Bước 2 duyệt job ảnh neo (ghi tên người duyệt) và đặt ảnh được chọn làm ảnh chuẩn. Việc chọn ảnh KHÔNG tự động: Quality Gate chỉ so khuôn mặt với
ảnh chuẩn nên ảnh chuẩn đầu tiên phải do người quyết định (như báo cáo: người phụ trách duyệt). Sau đó chạy scripts/expand_library.py.
Mô tả nhận dạng lấy từ bộ A01/A02 đã dùng ở các thử nghiệm trước (scripts/gen_character_dataset.py).
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

CHARACTERS = {
    "A01": {
        "description_vi": "Nữ Việt Nam khoảng 25 tuổi, tóc đen dài thẳng có mái bay lệch, mặt trái xoan, mắt nâu đen to, nốt ruồi nhỏ trên má phải, dáng mảnh.",
        "identity_en": ("young Vietnamese woman, around 25 years old, long straight black hair with side-swept bangs, soft oval face, "
                        "large dark brown eyes, small beauty mark on the right cheek, slim build"),
    },
    "A02": {
        "description_vi": "Nam Việt Nam khoảng 30 tuổi, tóc đen undercut ngắn, hàm sắc, lông mày mảnh thẳng, mắt nâu đen hẹp, nốt ruồi nhỏ dưới mắt trái, râu mờ, vóc dáng cân đối.",
        "identity_en": ("Vietnamese man, around 30 years old, short black undercut hair, sharp jawline, thin straight eyebrows, "
                        "narrow dark brown eyes, small mole under the left eye, light stubble, athletic build"),
    },
}
NEGATIVE = "blurry, low quality, deformed hands, extra fingers, bad anatomy, watermark, text, cartoon, anime, painting, monochrome"
ANCHOR = {"project_id": "BOOTSTRAP", "episode_id": None, "shot_id": None, "aspect_ratio": "9:16", "num_variants": 4, "auto_repair": False,
          "scene_spec": {"action": "standing, looking at the camera, neutral relaxed expression", "emotion": "neutral",
                         "camera": {"size": "medium close-up portrait, head and shoulders", "angle": "eye level"},
                         "lighting": "soft diffused daylight, plain light grey background",
                         "style": "photorealistic color photo, natural skin texture, sharp focus",
                         "keep": ["a plain white t-shirt"]}}


class Api:
    def __init__(self, base: str, token: str):
        self.base, self.token = base.rstrip("/"), token

    def call(self, method, path, body=None, headers=None):
        h = {"Content-Type": "application/json", **(headers or {})}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.base + path, method=method, data=json.dumps(body).encode() if body is not None else None, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            sys.exit(f"HTTP {e.code} {method} {path}: {e.read()[:500].decode('utf8', 'replace')}")


def find(api: Api, code: str) -> dict:
    for c in api.call("GET", "/api/v1/characters"):
        if c["code"] == code:
            return c
    sys.exit(f"không có nhân vật {code} (khởi động lại api để seed A01/A02)")


def cmd_anchors(a) -> int:
    api = Api(a.api, a.token)
    pending = {}
    for code, spec in CHARACTERS.items():
        ch = find(api, code)
        cur = (ch["current_version"] or {}).get("identity", {})
        if cur.get("identity_en") != spec["identity_en"]:
            ch = api.call("POST", f"/api/v1/characters/{ch['id']}/versions",
                          {"identity": {**spec, "negative_prompt": NEGATIVE}})
            print(f"{code}: đã cập nhật hồ sơ nhận dạng → v{ch['current_version']['version']}")
        ver = ch["current_version"]["id"]
        body = {**ANCHOR, "shot_id": f"anchor-{code}", "seed": a.seed, "characters": [{"character_version_id": ver}], "model_key": a.model}
        r = api.call("POST", "/api/v1/visual/jobs", body, {"Idempotency-Key": f"bootstrap:{code}:{a.seed}:{ch['current_version']['version']}"})
        pending[code] = r["job_id"]
        print(f"{code}: job {r['job_id']} · tuyến {[p['model_key'] for p in r.get('route_plan', [])]}")
    t0 = time.time()
    while pending and time.time() - t0 < a.timeout:
        for code, jid in list(pending.items()):
            j = api.call("GET", f"/api/v1/visual/jobs/{jid}")
            if j["state"] in ("review_required", "needs_human", "failed", "approved"):
                pending.pop(code)
                if j["state"] == "failed":
                    print(f"{code}: LỖI {j['error']}")
                    continue
                print(f"\n{code}: job {jid} xong ({j['state']}), {len(j['candidates'])} ảnh neo:")
                for i, c in enumerate(j["candidates"]):
                    print(f"   [{i}] candidate_id={c['id']}  asset={c['asset_id']}  QC={c['qc_verdict']}")
        if pending:
            time.sleep(10)
    print("\nMở tab 'Ảnh theo shot' (project BOOTSTRAP), xem ảnh, chọn một ảnh mỗi nhân vật, rồi chạy:")
    print("  python scripts/bootstrap_characters.py finalize --api ... --token ... --reviewer TÊN --pick A01=<candidate_id> --pick A02=<candidate_id>")
    return 0


def cmd_finalize(a) -> int:
    api = Api(a.api, a.token)
    picks = dict(p.split("=", 1) for p in a.pick)
    for code, cid in picks.items():
        ch = find(api, code)
        jobs = [j for j in api.call("GET", "/api/v1/visual/jobs?limit=100&project_id=BOOTSTRAP")["items"] if j["shot_id"] == f"anchor-{code}"]
        job = None
        for j in jobs:
            full = api.call("GET", f"/api/v1/visual/jobs/{j['id']}")
            if any(c["id"] == cid for c in full["candidates"]):
                job = full
                break
        if not job:
            sys.exit(f"{code}: candidate_id {cid} không thuộc job ảnh neo nào")
        cand = next(c for c in job["candidates"] if c["id"] == cid)
        if job["state"] != "approved":
            api.call("POST", f"/api/v1/visual/jobs/{job['id']}/review",
                     {"decision": "approve", "candidate_id": cid, "reviewer": a.reviewer, "note": "ảnh neo/ảnh chuẩn", "override_qc": cand["qc_verdict"] == "fail"})
        cur = ch["current_version"]
        if cand["asset_id"] in cur["references"]:
            print(f"{code}: ảnh đã là ảnh chuẩn")
            continue
        out = api.call("POST", f"/api/v1/characters/{ch['id']}/references", {"asset_id": cand["asset_id"]})
        print(f"{code}: đã đặt ảnh chuẩn → character version v{out['current_version']['version']}")
    print("\nTiếp theo (sinh ~240 ảnh/nhân vật vào dataset chờ duyệt):")
    for code in picks:
        print(f"  python scripts/expand_library.py --api {a.api} --token \"$API_TOKEN\" --code {code} --per-group 30")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("anchors", "finalize"):
        p = sub.add_parser(name)
        p.add_argument("--api", default="http://localhost:8000")
        p.add_argument("--token", default="")
    sub.choices["anchors"].add_argument("--seed", type=int, default=4242)
    sub.choices["anchors"].add_argument("--model", default=None, help="ghim model (vd sdxl-1.0); bỏ trống để Model Router chọn")
    sub.choices["anchors"].add_argument("--timeout", type=int, default=1800)
    sub.choices["finalize"].add_argument("--reviewer", required=True)
    sub.choices["finalize"].add_argument("--pick", action="append", required=True, help="CODE=candidate_id (lặp cho mỗi nhân vật)")
    a = ap.parse_args()
    return cmd_anchors(a) if a.cmd == "anchors" else cmd_finalize(a)


if __name__ == "__main__":
    sys.exit(main())
