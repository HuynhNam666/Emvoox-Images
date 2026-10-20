"""Duyệt hàng loạt 3 dataset công khai sau khi xem contact sheet (docs/evidence/public_*.jpg), chia split, khóa version.

Quyết định (ghi lại để truy vết):
- COCO: loại ảnh có caption nhắc trẻ em/em bé (tránh train trên ảnh trẻ nhỏ); ảnh cạnh ngắn <480px đã bị hệ thống tự loại.
- Unsplash / FFHQ: duyệt toàn bộ ảnh không bị tự loại (mẫu xem bằng mắt đạt). FFHQ không lọc được trẻ em theo caption.
- Split: ~5% val, ~5% test (ngẫu nhiên, seed cố định); ảnh gần trùng bị hệ thống ép cùng split.
"""
import json
import random
import re
import sys
import urllib.error
import urllib.request

API = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8010"
CHILD = re.compile(r"\b(child|children|kid|kids|baby|babies|boy|boys|girl|girls|toddler|infant|little)\b", re.I)


def call(method, path, body=None):
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def all_items(ds_id):
    out, off = [], 0
    while True:
        page = call("GET", f"/api/v1/datasets/{ds_id}/items?limit=200&offset={off}")
        out += page["items"]
        off += 200
        if off >= page["total"]:
            return out


for d in call("GET", "/api/v1/datasets"):
    if not d["name"].startswith(("COCO", "Unsplash", "FFHQ")):
        continue
    items = all_items(d["id"])
    pending = [i for i in items if i["status"] == "pending"]
    child = [i for i in pending if d["name"].startswith("COCO") and CHILD.search(i["caption"] or "")]
    child_ids = {i["id"] for i in child}
    for i in child:
        call("PATCH", f"/api/v1/dataset-items/{i['id']}", {"status": "rejected", "caption": f"{i['caption']}  [rejected: có trẻ em]"})
    keep = [i["id"] for i in pending if i["id"] not in child_ids]
    call("POST", f"/api/v1/datasets/{d['id']}/items/bulk", {"item_ids": keep, "status": "approved"})
    rnd = random.Random(7)
    rnd.shuffle(keep)
    k = max(1, round(len(keep) * 0.05))
    call("POST", f"/api/v1/datasets/{d['id']}/items/bulk", {"item_ids": keep[:k], "split": "val"})
    call("POST", f"/api/v1/datasets/{d['id']}/items/bulk", {"item_ids": keep[k:2 * k], "split": "test"})
    try:
        v = call("POST", f"/api/v1/datasets/{d['id']}/versions", {"harmonize_splits": True})
        print(f"{d['name']}: duyệt {len(keep)} · loại (trẻ em) {len(child)} · v{v['version']} {v['split_counts']}")
        for w in v["warnings"]:
            print("   ⚠", w)
    except urllib.error.HTTPError as e:
        print(d["name"], "lỗi version:", e.read().decode()[:300])
