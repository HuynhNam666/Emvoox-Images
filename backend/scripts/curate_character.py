"""Áp dụng quyết định duyệt cho dataset của một nhân vật rồi khóa 8 version.

    python scripts/curate_character.py --code A01 --trigger emv_a01 --decisions ../docs/evidence/a01_decisions.json

File decisions: {"<chỉ số ảnh trong plan>": ["approved"|"rejected", "train"|"val"|"test", "caption sau trigger / lý do loại"]}
Mọi ảnh trong plan phải có quyết định. Caption được đặt theo ĐÚNG nội dung nhìn thấy (đã xem contact sheet).
"""
import argparse
import json
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True)
ap.add_argument("--trigger", required=True)
ap.add_argument("--decisions", required=True)
ap.add_argument("--plan", default=None)
ap.add_argument("--api", default="http://localhost:8010")
a = ap.parse_args()


def call(method, path, body=None):
    req = urllib.request.Request(a.api + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


plan = json.load(open(a.plan or f"../data/{a.code.lower()}_plan.json", encoding="utf-8"))
D = {int(k): v for k, v in json.load(open(a.decisions, encoding="utf-8")).items()}
asset_to_idx = {j["asset_id"]: i for i, j in enumerate(plan["jobs"]) if "asset_id" in j}
missing = set(asset_to_idx.values()) - set(D)
assert not missing, f"thiếu quyết định cho ảnh: {sorted(missing)}"

n = {"approved": 0, "rejected": 0}
for g, ds_id in plan["datasets"].items():
    for it in call("GET", f"/api/v1/datasets/{ds_id}/items?limit=200")["items"]:
        status, split, cap = D[asset_to_idx[it["asset_id"]]]
        body = {"status": status, "split": split,
                "caption": f"{a.trigger}, {cap}" if status == "approved" else f"{it['caption']}  [rejected: {cap}]"}
        call("PATCH", f"/api/v1/dataset-items/{it['id']}", body)
        n[status] += 1
print("đã duyệt:", n)

versions = {}
for g, ds_id in plan["datasets"].items():
    try:
        v = call("POST", f"/api/v1/datasets/{ds_id}/versions", {"harmonize_splits": False})
    except urllib.error.HTTPError as e:
        print(g, "xung đột:", json.loads(e.read())["detail"].get("message"))
        v = call("POST", f"/api/v1/datasets/{ds_id}/versions", {"harmonize_splits": True})
    versions[g] = v["id"]
    print(f"{g:11} v{v['version']} {v['item_count']} ảnh {v['split_counts']}")
json.dump(versions, open(f"../data/{a.code.lower()}_versions.json", "w"), indent=1)
