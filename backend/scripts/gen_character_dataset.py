"""Sinh bộ ảnh nhân vật theo 8 nhóm bằng SDXL thật (qua API), đưa vào 8 dataset ở trạng thái chờ duyệt.

    python scripts/gen_character_dataset.py --code A02 --trigger emv_a02 [--api http://localhost:8010]

Cần API + worker GPU đang chạy. Ảnh AI vào dataset với synthetic=true, source_asset_id; vẫn phải duyệt tay.
Seed cố định giữa các ảnh để khuôn mặt gần nhau hơn (KHÔNG bảo đảm cùng một người) — bước duyệt là bắt buộc.
"""
import argparse
import json
import time
import urllib.request
import uuid

ap = argparse.ArgumentParser()
ap.add_argument("--code", default="A02")
ap.add_argument("--trigger", default="emv_a02")
ap.add_argument("--api", default="http://localhost:8010")
ap.add_argument("--seed", type=int, default=4242)
ap.add_argument("--steps", type=int, default=25)
ap.add_argument("--refs", action="store_true", help="dùng ảnh chuẩn của nhân vật (IP-Adapter) + seed khác nhau cho mỗi ảnh")
ap.add_argument("--out", default=None, help="file plan JSON (mặc định ../data/<code>_plan.json)")
a = ap.parse_args()

CHARACTERS = {
    "A02": dict(
        en=("Vietnamese man, around 30 years old, short black undercut hair, sharp jawline, thin straight eyebrows, "
            "narrow dark brown eyes, small mole under the left eye, light stubble, athletic build"),
        vi=("Nam Việt Nam khoảng 30 tuổi, tóc đen undercut ngắn, hàm sắc, lông mày mảnh thẳng, mắt nâu đen hẹp, "
            "nốt ruồi nhỏ dưới mắt trái, râu mờ, vóc dáng cân đối."),
        style="photorealistic photo, natural skin texture, sharp focus"),
    "A01": dict(
        en=("young Vietnamese woman, around 25 years old, long straight black hair with side-swept bangs, soft oval face, "
            "large dark brown eyes, small beauty mark on the right cheek, slim build"),
        vi=("Nữ Việt Nam khoảng 25 tuổi, tóc đen dài thẳng có mái bay lệch, mặt trái xoan, mắt nâu đen to, "
            "nốt ruồi nhỏ trên má phải, dáng mảnh."),
        style="photorealistic color photo, natural skin tones, natural skin texture, sharp focus"),
}
NEGATIVE = "blurry, low quality, deformed hands, extra fingers, bad anatomy, watermark, text, cartoon, anime, painting, monochrome, black and white"
# IP-Adapter: ảnh chuẩn càng nhỏ trong khung / càng cần nghe prompt thì scale càng thấp (rò rỉ trang phục/khung hình ở scale cao)
REF_SCALE = {"identity": 0.55, "expression": 0.5, "appearance": 0.5, "style": 0.5, "scene": 0.5, "pose": 0.4, "body": 0.4, "outfit": 0.35}

C = CHARACTERS[a.code]
IDENTITY_EN, IDENTITY_VI, STYLE = C["en"], C["vi"], C["style"]

# (nhóm, caption, phần prompt biến thiên)
PLAN = [
    ("identity", "front view close-up, neutral expression", "close-up portrait, front view, looking at camera, neutral expression, white t-shirt, plain grey background"),
    ("identity", "left profile close-up", "close-up portrait, left side profile view, white t-shirt, plain grey background"),
    ("identity", "right profile close-up", "close-up portrait, right side profile view, white t-shirt, plain grey background"),
    ("identity", "three-quarter view from the left, close-up", "close-up portrait, three-quarter view turned to the left, white t-shirt, plain grey background"),
    ("identity", "three-quarter view from the right, close-up", "close-up portrait, three-quarter view turned to the right, white t-shirt, plain grey background"),
    ("identity", "chin raised, looking up, close-up", "close-up portrait, chin slightly raised, looking up, white t-shirt, plain grey background"),
    ("body", "full body, front view", "full body shot, standing, front view, white shirt and dark trousers, plain studio background"),
    ("body", "full body, back view", "full body shot, standing, seen from behind, white shirt and dark trousers, plain studio background"),
    ("body", "full body, left side view", "full body shot, standing, left side view, white shirt and dark trousers, plain studio background"),
    ("body", "full body, right side view", "full body shot, standing, right side view, white shirt and dark trousers, plain studio background"),
    ("body", "waist-up, front view", "medium shot waist up, front view, white shirt, plain studio background"),
    ("expression", "neutral expression", "close-up portrait, neutral expression, grey t-shirt, soft light"),
    ("expression", "happy, smiling", "close-up portrait, happy smiling expression showing teeth, grey t-shirt, soft light"),
    ("expression", "sad expression", "close-up portrait, sad expression, downcast eyes, grey t-shirt, soft light"),
    ("expression", "angry expression", "close-up portrait, angry expression, frowning, grey t-shirt, soft light"),
    ("expression", "surprised expression", "close-up portrait, surprised expression, wide open eyes, grey t-shirt, soft light"),
    ("expression", "worried expression", "close-up portrait, worried anxious expression, grey t-shirt, soft light"),
    ("pose", "standing, arms at sides", "full body, standing relaxed with arms at sides, navy shirt, light grey wall"),
    ("pose", "sitting on a chair", "full body, sitting on a wooden chair, navy shirt, light grey wall"),
    ("pose", "walking", "full body, walking forward mid-stride, navy shirt, light grey wall"),
    ("pose", "running", "full body, running, dynamic motion, navy shirt, light grey wall"),
    ("pose", "bending forward", "full body, bending forward to pick something up, navy shirt, light grey wall"),
    ("pose", "holding a coffee cup", "medium shot, holding a coffee cup in both hands, navy shirt, light grey wall"),
    ("outfit", "white shirt, full body", "full body, wearing a crisp white dress shirt and black trousers, plain background"),
    ("outfit", "black suit, full body", "full body, wearing a black suit with white shirt and tie, plain background"),
    ("outfit", "casual hoodie and jeans, full body", "full body, wearing a casual grey hoodie and blue jeans, plain background"),
    ("outfit", "navy jacket, upper body detail", "medium shot, wearing a navy bomber jacket over a white t-shirt, plain background"),
    ("appearance", "wearing black-framed glasses", "close-up portrait, wearing black-framed glasses, white t-shirt, plain background"),
    ("appearance", "wearing a black cap", "close-up portrait, wearing a black baseball cap, white t-shirt, plain background"),
    ("appearance", "wearing a silver wristwatch", "medium shot, adjusting a silver wristwatch on the wrist, white shirt, plain background"),
    ("style", "cinematic film look, warm tones", "cinematic film still, warm color grading, shallow depth of field, upper body, dark jacket"),
    ("style", "studio portrait lighting", "studio portrait, dramatic rim lighting, black background, upper body, dark sweater"),
    ("style", "outdoor golden hour", "outdoor portrait at golden hour, backlit warm sunlight, upper body, light shirt"),
    ("scene", "in a library", "medium shot, standing in a library between bookshelves, beige sweater, window light"),
    ("scene", "on a city street in the daytime", "medium shot, standing on a busy city street in the daytime, denim jacket"),
    ("scene", "on a city street at night", "medium shot, standing on a city street at night with neon lights, dark jacket"),
    ("scene", "in an office", "medium shot, sitting at a desk in a modern office, white shirt"),
    ("scene", "in a bedroom", "medium shot, sitting on a bed in a bedroom, grey t-shirt, warm lamp light"),
]


def call(method, path, body=None, headers=None):
    req = urllib.request.Request(a.api + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} {path}: {e.read().decode()[:400]}")


chars = call("GET", "/api/v1/characters")
ch = next(c for c in chars if c["code"] == a.code)
cur = ch["current_version"]
if cur["identity"].get("identity_en") != IDENTITY_EN:
    ch = call("POST", f"/api/v1/characters/{ch['id']}/versions",
              {"identity": {"description_vi": IDENTITY_VI, "identity_en": IDENTITY_EN, "negative_prompt": NEGATIVE}})
    cur = ch["current_version"]
print(f"nhân vật {a.code} version {cur['version']}")

run_key = uuid.uuid4().hex[:8]
jobs = []
for i, (group, caption, suffix) in enumerate(PLAN):
    body = {
        "prompt_original": f"{a.code} — {caption}", "prompt_compiled": f"{IDENTITY_EN}, {suffix}, {STYLE}",
        "negative_prompt": NEGATIVE, "character_version_id": cur["id"], "aspect_ratio": "1:1-small",
        "num_images": 1, "seed": a.seed + (i * 17 if a.refs else 0), "steps": a.steps, "guidance_scale": 6.0}
    if a.refs:
        body.update({"use_references": True, "reference_scale": REF_SCALE[group]})
    r = call("POST", "/api/v1/generation-jobs", body, headers={"Idempotency-Key": f"ds-{a.code}-{run_key}-{i}"})
    jobs.append({"group": group, "caption": caption, "job_id": r["job_id"]})
print(f"đã gửi {len(jobs)} job; chờ worker…")

pending = set(range(len(jobs)))
t0 = time.time()
while pending and time.time() - t0 < 3600:
    for i in sorted(pending):
        j = call("GET", f"/api/v1/generation-jobs/{jobs[i]['job_id']}")
        if j["state"] == "review_required" and j["outputs"]:
            jobs[i]["asset_id"] = j["outputs"][0]["asset_id"]
            pending.discard(i)
        elif j["state"] in ("failed", "cancelled"):
            jobs[i]["error"] = j["error"]
            pending.discard(i)
    print(f"  còn {len(pending)} job ({int(time.time() - t0)}s)", flush=True)
    if pending:
        time.sleep(20)

group_names = {"identity": "khuôn mặt", "body": "toàn thân", "expression": "biểu cảm", "pose": "tư thế",
               "outfit": "trang phục", "appearance": "tóc & phụ kiện", "style": "phong cách", "scene": "bối cảnh"}
existing = {d["name"]: d for d in call("GET", "/api/v1/datasets")}
ds_ids = {}
for g, label in group_names.items():
    name = f"{a.code} — {label}"
    d = existing.get(name) or call("POST", "/api/v1/datasets", {
        "name": name, "group": g, "character_id": ch["id"],
        "source": "Tự sinh bằng SDXL 1.0 (EMVOOX Character Studio)", "license": "nội bộ/nghiên cứu — ảnh AI tự sinh"})
    ds_ids[g] = d["id"]
    items = [{"asset_id": j["asset_id"], "caption": f"{a.trigger}, {j['caption']}"}
             for j in jobs if j["group"] == g and "asset_id" in j]
    if items:
        rep = call("POST", f"/api/v1/datasets/{d['id']}/items/from-assets", {"items": items})
        print(f"  {name}: +{rep['added']} (đã có {len(rep['already_in_dataset'])}, gần trùng {len(rep['near_duplicates'])})")
failed = [j for j in jobs if "asset_id" not in j]
json.dump({"jobs": jobs, "datasets": ds_ids, "character_version_id": cur["id"]}, open(a.out or f"../data/{a.code.lower()}_plan.json", "w"), ensure_ascii=False, indent=1)
print(f"xong: {len(jobs) - len(failed)}/{len(jobs)} ảnh, {len(failed)} lỗi")
