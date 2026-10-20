"""Duyệt bộ ảnh A02 theo kết quả xem bằng mắt (contact sheet) + độ giống FaceNet. Ghi lại quyết định để truy vết.

Chạy sau gen_character_dataset.py. Mỗi ảnh: (chỉ số trong a02_plan.json) -> (status, split, caption đã sửa cho ĐÚNG nội dung ảnh).
Nguyên tắc: caption mô tả điều thực sự nhìn thấy; ảnh không đúng yêu cầu (vd 'mặt sau' nhưng quay mặt) bị loại hoặc sửa caption.
"""
import json
import sys
import urllib.request

API = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8010"
T = "emv_a02"
BW = "black and white photo"

# idx: (status, split, caption phần sau trigger)
D = {
    0: ("approved", "train", f"front view close-up, neutral expression, {BW}"),
    1: ("approved", "train", f"close-up portrait, head slightly turned, {BW}"),
    2: ("approved", "train", f"close-up portrait, head slightly turned, {BW}"),
    3: ("approved", "train", f"close-up portrait, slight three-quarter view, {BW}"),
    4: ("approved", "train", f"close-up portrait, slight three-quarter view, {BW}"),
    5: ("approved", "train", f"close-up portrait, chin slightly raised, {BW}"),
    10: ("approved", "train", f"close-up portrait, front view, {BW}"),
    6: ("approved", "train", "full body, standing, front view, white shirt and dark trousers, grey studio background"),
    7: ("rejected", "train", "rejected: prompt 'back view' không được tuân theo (quay mặt)"),
    8: ("approved", "train", "full body, standing, body turned slightly, hand in pocket, white shirt and dark trousers"),
    9: ("approved", "train", "full body, standing, body turned slightly, white shirt and dark trousers"),
    11: ("approved", "train", f"close-up portrait, neutral expression, grey t-shirt, {BW}"),
    12: ("approved", "train", "close-up portrait, happy smiling expression showing teeth, grey t-shirt, soft light"),
    13: ("approved", "train", f"close-up portrait, serious expression, grey t-shirt, {BW}"),
    14: ("rejected", "train", "rejected: kiểu line-art, không phải ảnh"),
    15: ("approved", "train", f"close-up portrait, neutral expression, wide eyes, grey t-shirt, {BW}"),
    16: ("approved", "train", f"close-up portrait, neutral expression, grey t-shirt, {BW}"),
    17: ("approved", "train", "full body, standing relaxed, navy trousers and grey-blue shirt, grey wall"),
    18: ("rejected", "train", "rejected: khuôn mặt khác rõ so với các ảnh khác"),
    19: ("approved", "test", "full body, walking forward, grey-blue shirt and navy trousers, grey wall"),
    20: ("rejected", "train", "rejected: khuôn mặt khác rõ so với các ảnh khác"),
    21: ("rejected", "train", "rejected: không đúng 'cúi người' (là ảnh chân dung)"),
    22: ("approved", "train", "medium shot, holding a white coffee cup, navy t-shirt, muted colors"),
    23: ("approved", "train", "full body, wearing a crisp white dress shirt and black trousers, grey background"),
    24: ("rejected", "train", "rejected: kiểu line-art, không phải ảnh"),
    25: ("approved", "train", "upper body, wearing a blue hoodie, grey background"),
    26: ("approved", "train", "upper body, wearing a navy bomber jacket over a white t-shirt, grey background"),
    27: ("rejected", "train", "rejected: kiểu nét vẽ, không phải ảnh"),
    28: ("rejected", "train", "rejected: kiểu nét vẽ, không phải ảnh"),
    29: ("approved", "train", "full body, standing, white shirt with a wristwatch, dark trousers, grey background"),
    30: ("approved", "test", "cinematic film still, warm color grading, shallow depth of field, upper body, dark jacket"),
    31: ("approved", "train", f"studio portrait, dramatic rim lighting, black background, upper body, dark sweater, {BW}"),
    32: ("approved", "val", "outdoor portrait at golden hour, backlit warm sunlight, upper body, grey t-shirt"),
    33: ("approved", "train", "medium shot, in a library between bookshelves, beige cardigan over grey t-shirt, warm light"),
    34: ("approved", "val", "medium shot, standing on a busy city street in the daytime, denim jacket"),
    35: ("approved", "test", "close-up, on a city street at night, neon lights, dark jacket"),
    36: ("rejected", "train", "rejected: khuôn mặt khác rõ so với các ảnh khác"),
    37: ("approved", "train", "medium shot, sitting on a bed in a bedroom, grey t-shirt, warm lamp light"),
}


def call(method, path, body=None):
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


plan = json.load(open("../data/a02_plan.json", encoding="utf-8"))
asset_to_idx = {j["asset_id"]: i for i, j in enumerate(plan["jobs"])}
assert set(D) == set(range(len(plan["jobs"]))), "thiếu quyết định cho một số ảnh"
n = {"approved": 0, "rejected": 0}
for g, ds_id in plan["datasets"].items():
    for it in call("GET", f"/api/v1/datasets/{ds_id}/items?limit=200")["items"]:
        status, split, cap = D[asset_to_idx[it["asset_id"]]]
        body = {"status": status, "split": split}
        if status == "approved":
            body["caption"] = f"{T}, {cap}"
        else:
            body["caption"] = f"{it['caption']}  [{cap}]"  # giữ lý do loại trong caption để truy vết
        call("PATCH", f"/api/v1/dataset-items/{it['id']}", body)
        n[status] += 1
print("đã duyệt:", n)

versions = {}
for g, ds_id in plan["datasets"].items():
    try:
        v = call("POST", f"/api/v1/datasets/{ds_id}/versions", {"harmonize_splits": False})
    except urllib.error.HTTPError as e:
        body = json.loads(e.read())
        print(g, "xung đột:", body["detail"])
        v = call("POST", f"/api/v1/datasets/{ds_id}/versions", {"harmonize_splits": True})
    versions[g] = v["id"]
    print(f"{g:11} v{v['version']} {v['item_count']} ảnh {v['split_counts']}")
json.dump(versions, open("../data/a02_versions.json", "w"), indent=1)
