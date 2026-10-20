"""Đánh giá nhận dạng LoRA trên cảnh CHƯA HỌC, so với không LoRA. Cần API + worker GPU đang chạy.

    FACE_DEVICE=cpu python scripts/eval_identity.py --lora <model_version_id> --code A02 --trigger emv_a02

Ba nhánh, cùng prompt cảnh và cùng seed:
  A  không LoRA, mô tả nhân vật bằng chữ (cách hiện tại khi chưa có LoRA)
  B  CÓ LoRA, chỉ trigger — không mô tả chữ (LoRA phải tự mang nhận dạng)
  C  không LoRA, chỉ "a man" — đối chứng dưới (người ngẫu nhiên)
Chỉ số: cosine FaceNet giữa mặt trong ảnh sinh và vector trung bình của ảnh tham chiếu đã duyệt.
Mốc tham chiếu: phân phối leave-one-out của chính ảnh tham chiếu. Đây là chỉ số hỗ trợ, KHÔNG phải ngưỡng "cùng người".
"""
import argparse
import io
import json
import time
import urllib.request
import uuid

import numpy as np
from PIL import Image, ImageDraw

from scripts.face_sim import cos, embed, fetch

ap = argparse.ArgumentParser()
ap.add_argument("--lora", required=True)
ap.add_argument("--code", default="A02")
ap.add_argument("--trigger", default="emv_a02")
ap.add_argument("--api", default="http://localhost:8010")
ap.add_argument("--out", default="../data/eval")
ap.add_argument("--arms", default="ABC", help="A..G; D = LoRA+chữ; E = IP+chữ; F = IP+'a man'; G = LoRA+IP+chữ")
ap.add_argument("--reuse", default="", help="file json eval cũ (phân tách bằng dấu phẩy) để đo lại với cùng mốc tham chiếu")
ap.add_argument("--scale", type=float, default=0.6, help="reference_scale cho nhánh có IP-Adapter")
a = ap.parse_args()

IDENTITY_EN = ("Vietnamese man, around 30 years old, short black undercut hair, sharp jawline, thin straight eyebrows, "
               "narrow dark brown eyes, small mole under the left eye, light stubble, athletic build")
SCENES = ["standing on a rooftop at sunset, city skyline behind", "sitting in a cafe reading a book, warm light",
          "walking in the rain holding an umbrella on a street", "cooking in a bright home kitchen",
          "riding a bicycle in a green park", "waiting on a train station platform"]
SEEDS = [101, 202]
NEG = "blurry, low quality, deformed hands, extra fingers, bad anatomy, watermark, text, cartoon, anime, painting"
STYLE = "photorealistic photo, natural skin texture, sharp focus, medium shot"


def call(method, path, body=None, headers=None):
    req = urllib.request.Request(a.api + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} {path}: {e.read().decode()[:300]}")


# --- tham chiếu: ảnh đã duyệt (split train/val/test) có mặt lớn ---
plan = json.load(open("../data/a02_plan.json", encoding="utf-8"))
ref_groups = ["identity", "expression", "style", "scene", "outfit"]
char_version = next(c for c in call("GET", "/api/v1/characters") if c["code"] == a.code)["current_version"]
exclude = set(char_version["references"])  # ảnh chuẩn dùng để điều kiện IP-Adapter không được tính vào mốc đo
print(f"ảnh chuẩn (loại khỏi mốc đo): {len(exclude)}")
ref_assets = []
for g in ref_groups:
    for it in call("GET", f"/api/v1/datasets/{plan['datasets'][g]}/items?status=approved&limit=200")["items"]:
        if it["asset_id"] not in exclude:
            ref_assets.append(it["asset_id"])
ref_emb = [e for e in (embed(fetch(a.api, x)) for x in ref_assets) if e is not None]
mean = np.mean(ref_emb, axis=0)
mean /= np.linalg.norm(mean)
loo = []
for i, e in enumerate(ref_emb):
    m = np.mean([x for j, x in enumerate(ref_emb) if j != i], axis=0)
    loo.append(cos(e, m / np.linalg.norm(m)))
p10 = float(np.percentile(loo, 10))
print(f"tham chiếu: {len(ref_emb)} ảnh; leave-one-out sim median {np.median(loo):.3f}, p10 {p10:.3f}, min {min(loo):.3f}")

# --- gửi job ---
tag = uuid.uuid4().hex[:6]
ALL = {"A": "không LoRA + mô tả chữ", "B": "LoRA (chỉ trigger)", "C": "không LoRA, 'a man'", "D": "LoRA + trigger + mô tả chữ",
       "E": "IP-Adapter + mô tả chữ", "F": "IP-Adapter, 'a man'", "G": "LoRA + IP-Adapter + mô tả chữ"}
arms = {k: v for k, v in ALL.items() if k in a.arms}
jobs = []
for scene in SCENES:
    for seed in SEEDS:
        for arm in arms:
            prompt = {"A": f"{IDENTITY_EN}, {scene}, {STYLE}", "B": f"{a.trigger}, {scene}, {STYLE}",
                      "C": f"a man, {scene}, {STYLE}",
                      "D": f"{a.trigger}, {IDENTITY_EN}, {scene}, {STYLE}",
                      "E": f"{IDENTITY_EN}, {scene}, {STYLE}", "F": f"a man, {scene}, {STYLE}",
                      "G": f"{a.trigger}, {IDENTITY_EN}, {scene}, {STYLE}"}[arm]
            body = {"prompt_original": f"eval {arm} {scene}", "prompt_compiled": prompt, "negative_prompt": NEG,
                    "aspect_ratio": "1:1-small", "num_images": 1, "seed": seed, "steps": 25, "guidance_scale": 6.0}
            if arm in ("B", "D", "G"):
                body.update({"lora_version_id": a.lora, "lora_scale": 1.0, "evaluation": True})
            if arm in ("E", "F", "G"):
                body.update({"character_version_id": char_version["id"], "use_references": True, "reference_scale": a.scale})
            r = call("POST", "/api/v1/generation-jobs", body, {"Idempotency-Key": f"eval-{tag}-{len(jobs)}"})
            jobs.append({"arm": arm, "scene": scene, "seed": seed, "job_id": r["job_id"]})
print(f"đã gửi {len(jobs)} job")
pending = set(range(len(jobs)))
t0 = time.time()
while pending and time.time() - t0 < 5400:
    for i in sorted(pending):
        j = call("GET", f"/api/v1/generation-jobs/{jobs[i]['job_id']}")
        if j["state"] == "review_required" and j["outputs"]:
            jobs[i]["asset_id"] = j["outputs"][0]["asset_id"]
            pending.discard(i)
        elif j["state"] in ("failed", "cancelled"):
            jobs[i]["error"] = j["error"]
            pending.discard(i)
    print(f"  còn {len(pending)} ({int(time.time() - t0)}s)", flush=True)
    if pending:
        time.sleep(30)

# --- đo ---
for f in [x for x in a.reuse.split(",") if x]:
    for j in json.load(open(f, encoding="utf-8"))["jobs"]:
        if "asset_id" in j and j["arm"] not in arms:
            j.pop("sim", None)
            jobs.append({**j, "reused": True})
for j in jobs:
    if "asset_id" in j:
        e = embed(fetch(a.api, j["asset_id"]))
        j["face"] = e is not None
        j["sim"] = None if e is None else round(cos(e, mean), 3)
res = {}
shown = {k: v for k, v in ALL.items() if any(j["arm"] == k for j in jobs)}
for arm, label in shown.items():
    rows = [j for j in jobs if j["arm"] == arm]
    sims = [j["sim"] for j in rows if j.get("sim") is not None]
    res[arm] = {"label": label, "n": len(rows), "face_detected": len(sims), "mean_sim": round(float(np.mean(sims)), 3) if sims else None,
                "std": round(float(np.std(sims)), 3) if sims else None,
                "frac_ge_loo_p10": round(sum(s >= p10 for s in sims) / max(1, len(rows)), 3), "errors": sum("error" in j for j in rows)}
    print(f"{arm} {label:28} n={len(rows)} mặt={len(sims)} sim={res[arm]['mean_sim']}±{res[arm]['std']} ≥p10(ref)={res[arm]['frac_ge_loo_p10']} lỗi={res[arm]['errors']}")
json.dump({"loo_p10": p10, "loo_median": float(np.median(loo)), "arms": res, "jobs": jobs}, open(a.out + ".json", "w"), ensure_ascii=False, indent=1)

# --- contact sheet: mỗi hàng một (scene, seed), cột A|B|C ---
S = 300
W = len(arms)
rows_ = [(s, sd) for s in SCENES for sd in SEEDS]
sheet = Image.new("RGB", (W * S, len(rows_) * (S + 14)), "white")
d = ImageDraw.Draw(sheet)
for r_i, (s, sd) in enumerate(rows_):
    for c_i, arm in enumerate(arms):
        j = next(x for x in jobs if x["arm"] == arm and x["scene"] == s and x["seed"] == sd)
        x0, y0 = c_i * S, r_i * (S + 14)
        d.text((x0 + 2, y0 + 1), f"{arm} sim={j.get('sim')} | {s[:30]}", fill="black")
        if "asset_id" in j:
            im = Image.open(io.BytesIO(urllib.request.urlopen(f"{a.api}/api/v1/assets/{j['asset_id']}/thumbnail").read())).convert("RGB").resize((S, S))
            sheet.paste(im, (x0, y0 + 14))
sheet.save(a.out + "_sheet.jpg", quality=85)
print("đã lưu", a.out + ".json", a.out + "_sheet.jpg")
