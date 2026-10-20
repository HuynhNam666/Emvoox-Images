"""Mở rộng thư viện ảnh nhân vật lên 200–300 ảnh/nhân vật (báo cáo mục 3.1) bằng pipeline /visual/jobs mới.

  python scripts/expand_library.py --api http://localhost:8080 --token $API_TOKEN --code A01 --per-group 30 [--model flux2-klein-4b] [--budget 0.5]

Mỗi nhóm (identity, body, expression, pose, outfit, appearance, style, scene) sinh --per-group ảnh bằng cách tổ hợp các trục biến thiên
(góc nhìn × biểu cảm × trang phục × ánh sáng × bối cảnh) với seed khác nhau. Mọi ảnh đi qua Model Router + Quality Gate (FaceNet so với
ảnh chuẩn); chỉ ảnh QC 'pass' (hoặc 'review' nếu --include-review) được đưa vào dataset 'CODE — nhóm' ở trạng thái CHỜ DUYỆT
(synthetic=true, source_asset_id). Người duyệt vẫn quyết định cuối cùng; ảnh gần trùng bị Dataset Manager gắn cờ.
Chạy lại an toàn: Idempotency-Key cố định theo (nhân vật, nhóm, chỉ số). Ảnh phải thuộc nhân vật đã có ảnh chuẩn (nếu chưa: tạo ảnh chuẩn trước).
"""
import argparse
import itertools
import json
import random
import sys
import time
import zlib
import urllib.error
import urllib.request

GROUP_NAMES = {"identity": "khuôn mặt", "body": "toàn thân", "expression": "biểu cảm", "pose": "tư thế", "outfit": "trang phục",
               "appearance": "tóc & phụ kiện", "style": "phong cách", "scene": "bối cảnh"}
VIEWS = ["front view", "left three-quarter view", "right three-quarter view", "left profile", "right profile", "slightly from above", "slightly from below"]
EXPR = ["neutral", "gentle smile", "laughing", "sad", "angry", "surprised", "worried", "thoughtful", "tired", "determined"]
OUTFIT = ["a plain white shirt", "a black suit with white shirt", "a grey hoodie and jeans", "a navy bomber jacket over a white t-shirt",
          "a beige trench coat", "a casual denim jacket", "a formal dark blazer", "a cotton sweater"]
POSE = ["standing relaxed", "sitting on a chair", "walking forward mid-stride", "running", "bending forward to pick something up",
        "holding a coffee cup with both hands", "leaning on a wall", "looking over the shoulder", "reaching up to a shelf", "sitting cross-legged on the floor"]
LIGHT = ["soft window light from the left", "warm evening side light", "overcast daylight", "cool neon rim light", "golden hour backlight",
         "hard top light with deep shadows", "soft frontal studio light", "warm indoor lamp light"]
PLACE = ["in a quiet corridor", "in a library between bookshelves", "in a small cafe", "on a rainy night street", "in a bright office",
         "in a living room", "on a rooftop at dusk", "at a train station platform"]
ACCESS = ["black-framed glasses", "a black baseball cap", "a silver wristwatch", "a thin scarf", "small earrings", "a leather backpack", "a hair tie, hair tied back", "a face mask pulled down"]
SIZE_BY_GROUP = {"identity": "close-up", "expression": "close-up", "appearance": "medium close-up", "body": "full body shot", "pose": "full body shot",
                 "outfit": "full body shot", "style": "medium shot", "scene": "medium wide shot"}


def plan(group: str, n: int, seed: int) -> list[dict]:
    rnd = random.Random(f"{seed}:{group}")
    if group == "identity":
        axes = list(itertools.product(VIEWS, ["neutral", "gentle smile"], LIGHT[:4]))
        make = lambda t: dict(action=f"{t[0]}, {t[1]} expression", light=t[2], outfit=OUTFIT[0], place="against a plain grey background")  # noqa: E731
    elif group == "body":
        axes = list(itertools.product(["front view", "back view", "left side view", "right side view", "three-quarter view"], OUTFIT[:4], LIGHT[:3]))
        make = lambda t: dict(action=f"standing, {t[0]}", light=t[2], outfit=t[1], place="against a plain studio background")  # noqa: E731
    elif group == "expression":
        axes = list(itertools.product(EXPR, ["front view", "slight left three-quarter", "slight right three-quarter"], LIGHT[:4]))
        make = lambda t: dict(action=f"{t[1]}, {t[0]} expression", light=t[2], outfit=OUTFIT[2], place="against a soft neutral background")  # noqa: E731
    elif group == "pose":
        axes = list(itertools.product(POSE, OUTFIT[:3], LIGHT[:4]))
        make = lambda t: dict(action=t[0], light=t[2], outfit=t[1], place="against a light grey wall")  # noqa: E731
    elif group == "outfit":
        axes = list(itertools.product(OUTFIT, ["front view", "back view", "side view"], LIGHT[:3]))
        make = lambda t: dict(action=f"standing, {t[1]}", light=t[2], outfit=t[0], place="against a plain background")  # noqa: E731
    elif group == "appearance":
        axes = list(itertools.product(ACCESS, ["front view", "three-quarter view"], LIGHT[:3]))
        make = lambda t: dict(action=f"wearing {t[0]}, {t[1]}", light=t[2], outfit=OUTFIT[0], place="against a plain background")  # noqa: E731
    elif group == "style":
        axes = list(itertools.product(["cinematic film still, warm color grading, shallow depth of field", "studio portrait, dramatic rim lighting, black background",
                                       "outdoor portrait at golden hour, backlit warm sunlight", "documentary photo, natural light, muted colors"], POSE[:4], OUTFIT[:3]))
        make = lambda t: dict(action=f"{t[1]}", light=t[0], outfit=t[2], place="")  # noqa: E731
    else:
        axes = list(itertools.product(PLACE, POSE[:5], LIGHT))
        make = lambda t: dict(action=t[1], light=t[2], outfit=OUTFIT[3], place=t[0])  # noqa: E731
    rnd.shuffle(axes)
    return [make(t) for t in axes[:n]]


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
            raise SystemExit(f"HTTP {e.code} {method} {path}: {e.read()[:400].decode('utf8', 'replace')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8080")
    ap.add_argument("--token", default="")
    ap.add_argument("--code", required=True, help="mã nhân vật, vd A01")
    ap.add_argument("--per-group", type=int, default=30)
    ap.add_argument("--groups", default=",".join(GROUP_NAMES))
    ap.add_argument("--variants", type=int, default=2, help="số ảnh mỗi job (job = per-group/variants)")
    ap.add_argument("--model", default=None, help="ghim model; bỏ trống để Model Router chọn")
    ap.add_argument("--budget", type=float, default=0.0, help="USD tối đa mỗi job (0 = không giới hạn)")
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--include-review", action="store_true", help="nhận cả ảnh QC 'review' (khi FaceNet không khả dụng)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--timeout", type=int, default=7200)
    a = ap.parse_args()
    api = Api(a.api, a.token)

    chars = {c["code"]: c for c in api.call("GET", "/api/v1/characters")}
    if a.code not in chars:
        sys.exit(f"không có nhân vật {a.code}")
    ch = chars[a.code]
    ver = ch["current_version"]
    if not ver or not ver["references"]:
        sys.exit(f"{a.code} chưa có ảnh chuẩn (references): tạo và đặt ảnh chuẩn trước, nếu không Quality Gate không đo được nhận dạng")
    identity = ver["identity"]["identity_en"]
    groups = [g for g in a.groups.split(",") if g in GROUP_NAMES]
    existing = {d["name"]: d for d in api.call("GET", "/api/v1/datasets")}

    jobs: list[tuple[str, str, dict]] = []   # (group, job_id, spec)
    for g in groups:
        specs = plan(g, a.per_group, a.seed)
        for k in range(0, len(specs), a.variants):
            sp = specs[k]
            scene = f"{sp['place']}".strip()
            body = {"project_id": f"library-{a.code}", "shot_id": f"{g}-{k // a.variants:03d}",
                    "characters": [{"character_version_id": ver["id"]}],
                    "scene_spec": {"action": sp["action"], "emotion": "", "camera": {"size": SIZE_BY_GROUP[g], "angle": "eye level"},
                                   "lighting": sp["light"], "style": "photorealistic, natural skin texture, sharp focus",
                                   "keep": ["facial identity", "hairstyle"] + ([f"wearing {sp['outfit']}"] if sp["outfit"] else []) + ([scene] if scene else [])},
                    "num_variants": a.variants, "seed": a.seed + k * 13 + zlib.crc32(g.encode()) % 997, "budget_limit": a.budget, "model_key": a.model}
            if a.dry_run:
                print(g, body["shot_id"], sp["action"], "|", sp["light"])
                continue
            r = api.call("POST", "/api/v1/visual/jobs", body, {"Idempotency-Key": f"lib:{a.code}:{g}:{k // a.variants}:{a.seed}"})
            jobs.append((g, r["job_id"], sp))
    if a.dry_run:
        print(f"Dự kiến {len(groups)} nhóm × {a.per_group} ảnh = {len(groups) * a.per_group} ảnh/nhân vật (chưa gửi job nào)")
        return 0
    print(f"Đã gửi {len(jobs)} job; chờ hoàn tất…")
    t0, done = time.time(), {}
    while len(done) < len(jobs) and time.time() - t0 < a.timeout:
        for g, jid, sp in jobs:
            if jid in done:
                continue
            j = api.call("GET", f"/api/v1/visual/jobs/{jid}")
            if j["state"] in ("review_required", "approved", "needs_human", "failed", "rejected"):
                done[jid] = j
        print(f"  {len(done)}/{len(jobs)} xong ({int(time.time() - t0)}s)", flush=True)
        if len(done) < len(jobs):
            time.sleep(15)
    cost = sum(j["cost_spent"] for j in done.values())
    added = {g: 0 for g in groups}
    skipped = {"fail": 0, "review": 0, "failed_job": 0}
    for (g, jid, sp) in jobs:
        j = done.get(jid)
        if not j or j["state"] == "failed":
            skipped["failed_job"] += 1
            continue
        name = f"{a.code} — {GROUP_NAMES[g]}"
        ds = existing.get(name) or api.call("POST", "/api/v1/datasets", {
            "name": name, "group": g, "character_id": ch["id"], "source": f"pipeline /visual/jobs ({a.model or 'Model Router'})",
            "license": "Ảnh AI tự sinh trong hệ thống"})
        existing[name] = ds
        items = []
        for c in j["candidates"]:
            if c["qc_verdict"] == "pass" or (a.include_review and c["qc_verdict"] == "review"):
                cap = f"{identity.split(',')[0]}, {sp['action']}, {sp['light']}".strip()
                items.append({"asset_id": c["asset_id"], "caption": cap})
            else:
                skipped[c["qc_verdict"] if c["qc_verdict"] in skipped else "fail"] += 1
        if items:
            rep = api.call("POST", f"/api/v1/datasets/{ds['id']}/items/from-assets", {"items": items})
            added[g] += rep["added"]
    print(f"\nChi phí API+GPU ghi nhận: {cost:.4f}")
    print("Đã thêm vào dataset (chờ duyệt):", json.dumps(added, ensure_ascii=False))
    print("Bỏ qua:", json.dumps(skipped, ensure_ascii=False), "· duyệt ở tab Dataset rồi khóa version trước khi train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
