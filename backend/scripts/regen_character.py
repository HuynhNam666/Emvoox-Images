"""Sinh bổ sung ảnh cho các nhóm dataset bị lệch yêu cầu (vd kính/mũ/biểu cảm), thêm vào plan + dataset (chờ duyệt).

    python scripts/regen_character.py --code A01 --spec ../docs/evidence/a01_regen.json

spec: danh sách {"group", "caption", "suffix", "scale", "seed"}. scale = reference_scale IP-Adapter (thấp hơn = nghe prompt hơn,
nhưng nhận dạng giảm: luôn đo lại bằng FaceNet). Ảnh mới được nối vào <code>_plan.json với chỉ số tiếp theo.
"""
import argparse
import json
import time
import urllib.request
import uuid

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True)
ap.add_argument("--spec", required=True)
ap.add_argument("--api", default="http://localhost:8010")
ap.add_argument("--steps", type=int, default=25)
a = ap.parse_args()

NEG_EXTRA = ", collage, multiple people, split screen, grid of photos, duplicate"
STYLE = {"A01": "photorealistic color photo, natural skin tones, natural skin texture, sharp focus",
         "A02": "photorealistic photo, natural skin texture, sharp focus"}[a.code]
NEG = "blurry, low quality, deformed hands, extra fingers, bad anatomy, watermark, text, cartoon, anime, painting, monochrome, black and white"


def call(method, path, body=None, headers=None):
    req = urllib.request.Request(a.api + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


plan_path = f"../data/{a.code.lower()}_plan.json"
plan = json.load(open(plan_path, encoding="utf-8"))
spec = json.load(open(a.spec, encoding="utf-8"))
ch = next(c for c in call("GET", "/api/v1/characters") if c["code"] == a.code)
cur = ch["current_version"]
en = cur["identity"]["identity_en"]
tag = uuid.uuid4().hex[:6]
new = []
for i, it in enumerate(spec):
    r = call("POST", "/api/v1/generation-jobs", {
        "prompt_original": f"{a.code} — {it['caption']}", "prompt_compiled": f"{en}, {it['suffix']}, {STYLE}",
        "negative_prompt": NEG + NEG_EXTRA, "character_version_id": cur["id"], "aspect_ratio": "1:1-small", "num_images": 1,
        "seed": it["seed"], "steps": a.steps, "guidance_scale": 6.5, "use_references": True, "reference_scale": it["scale"]},
        headers={"Idempotency-Key": f"regen-{a.code}-{tag}-{i}"})
    new.append({"group": it["group"], "caption": it["caption"], "job_id": r["job_id"], "regen": True})
print(f"đã gửi {len(new)} job bổ sung")
pending = set(range(len(new)))
t0 = time.time()
while pending and time.time() - t0 < 3600:
    for i in sorted(pending):
        j = call("GET", f"/api/v1/generation-jobs/{new[i]['job_id']}")
        if j["state"] == "review_required" and j["outputs"]:
            new[i]["asset_id"] = j["outputs"][0]["asset_id"]
            pending.discard(i)
        elif j["state"] in ("failed", "cancelled"):
            new[i]["error"] = j["error"]
            pending.discard(i)
    print(f"  còn {len(pending)} ({int(time.time() - t0)}s)", flush=True)
    if pending:
        time.sleep(20)

start = len(plan["jobs"])
for g, ds_id in plan["datasets"].items():
    items = [{"asset_id": j["asset_id"], "caption": f"emv_{a.code.lower()}, {j['caption']}"} for j in new if j["group"] == g and "asset_id" in j]
    if items:
        rep = call("POST", f"/api/v1/datasets/{ds_id}/items/from-assets", {"items": items})
        print(f"  {g}: +{rep['added']}")
plan["jobs"] += new
json.dump(plan, open(plan_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"plan: {start} -> {len(plan['jobs'])} ảnh (chỉ số mới {start}…{len(plan['jobs']) - 1})")
