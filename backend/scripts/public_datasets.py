"""Chuẩn bị tập con của dataset công khai (ảnh + caption .txt cùng tên) rồi nạp vào Dataset Manager qua API.

    python scripts/public_datasets.py prepare unsplash --n 600
    python scripts/public_datasets.py prepare coco --n 400
    python scripts/public_datasets.py prepare ffhq --n 300
    python scripts/public_datasets.py prepare mpii --n 300
    python scripts/public_datasets.py import  unsplash|coco|ffhq [--api http://localhost:8010]

Thư mục gốc dữ liệu: PUBLIC_DIR (mặc định D:/emvoox-data/public). KHÔNG tự đăng ký/ký agreement: chỉ dùng các bộ tải trực tiếp.
Ảnh nạp vào trạng thái chờ duyệt; nguồn/license ghi ở cấp dataset. Chỉ dùng nội bộ/nghiên cứu, không phân phối lại ảnh.
"""
import argparse
import io
import json
import os
import random
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(os.environ.get("PUBLIC_DIR", "./data/public"))
SEED = 42

META = {
    "unsplash": dict(
        name="Unsplash Lite — bối cảnh", group="scene",
        source="Unsplash Lite Dataset 1.4.0 — https://unsplash.com/data (ảnh tải từ photo_image_url, tập con ngẫu nhiên)",
        license="Unsplash Dataset Lite Terms: tải/lưu và train ML nội bộ; KHÔNG phân phối lại ảnh"),
    "coco": dict(
        name="COCO val2017 — người (tư thế)", group="pose",
        source="COCO 2017 val — https://cocodataset.org (ảnh một người, đủ keypoints; caption gốc COCO)",
        license="Annotations CC BY 4.0; ảnh Flickr license hỗn hợp — chỉ nghiên cứu/nội bộ"),
    "mpii": dict(
        name="MPII Human Pose — tư thế (nghiên cứu)", group="pose",
        source="Voxel51/MPII_Human_Pose_Dataset (Hugging Face), ảnh gốc MPII — https://human-pose.mpi-inf.mpg.de",
        license="Chú thích BSD-2-Clause; ảnh gốc từ video YouTube — chỉ nghiên cứu phi thương mại"),
    "ffhq": dict(
        name="FFHQ — khuôn mặt ảnh thật", group="style",
        source="FFHQ (NVlabs) — https://github.com/NVlabs/ffhq-dataset (tập con ngẫu nhiên, 1024×1024)",
        license="CC BY-NC-SA 4.0 (metadata); từng ảnh có license Flickr riêng — chỉ nghiên cứu phi thương mại"),
}


# ------------------------------------------------------------------ prepare
def prepare_unsplash(n: int) -> None:
    import csv
    base = ROOT / "unsplash"
    out = base / "subset"
    out.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(10**8)
    with open(base / "x" / "photos.tsv000", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t")
                if (r["ai_description"] or r["photo_description"]).strip() and int(r["photo_width"] or 0) >= 1024]
    random.Random(SEED).shuffle(rows)
    rows = rows[:n]
    print(f"Unsplash: chọn {len(rows)} ảnh; tải w=1024 (4 luồng, lịch sự)")

    def get(r):
        pid = r["photo_id"]
        dst = out / f"{pid}.jpg"
        cap = (r["ai_description"] or r["photo_description"]).strip().replace("\n", " ")
        if dst.exists() and dst.stat().st_size > 10_000:
            (out / f"{pid}.txt").write_text(cap, encoding="utf-8")
            return True
        url = r["photo_image_url"] + ("&" if "?" in r["photo_image_url"] else "?") + "w=1024&q=85&fm=jpg"
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "emvoox-research/1.0"})
                data = urllib.request.urlopen(req, timeout=60).read()
                if len(data) > 10_000:
                    dst.write_bytes(data)
                    (out / f"{pid}.txt").write_text(cap, encoding="utf-8")
                    return True
            except Exception:
                time.sleep(2 * (attempt + 1))
        return False

    with ThreadPoolExecutor(4) as ex:
        ok = sum(ex.map(get, rows))
    print(f"Unsplash: tải được {ok}/{len(rows)} ảnh -> {out}")


def prepare_coco(n: int) -> None:
    base = ROOT / "coco"
    ann_dir, img_dir = base / "annotations", base / "val2017"
    if not ann_dir.exists():
        zipfile.ZipFile(base / "annotations_trainval2017.zip").extractall(base)
    if not img_dir.exists():
        zipfile.ZipFile(base / "val2017.zip").extractall(base)
    kp = json.load(open(ann_dir / "person_keypoints_val2017.json", encoding="utf-8"))
    caps = json.load(open(ann_dir / "captions_val2017.json", encoding="utf-8"))
    imgs = {i["id"]: i for i in kp["images"]}
    per_img: dict[int, list] = {}
    for a in kp["annotations"]:
        per_img.setdefault(a["image_id"], []).append(a)
    cap_of: dict[int, str] = {}
    for c in caps["annotations"]:
        cap_of.setdefault(c["image_id"], c["caption"].strip())
    keep = []
    for iid, anns in per_img.items():
        persons = [a for a in anns if not a["iscrowd"]]
        if len(persons) != 1 or persons[0]["num_keypoints"] < 12:
            continue
        im = imgs[iid]
        bw, bh = persons[0]["bbox"][2], persons[0]["bbox"][3]
        if bw * bh < 0.25 * im["width"] * im["height"] or iid not in cap_of:
            continue
        keep.append(iid)
    random.Random(SEED).shuffle(keep)
    keep = keep[:n]
    out = base / "subset"
    out.mkdir(exist_ok=True)
    for iid in keep:
        name = imgs[iid]["file_name"]
        (out / name).write_bytes((img_dir / name).read_bytes())
        (out / (Path(name).stem + ".txt")).write_text(cap_of[iid], encoding="utf-8")
    print(f"COCO: {len(keep)} ảnh một người (≥12 keypoints, người ≥25% khung) -> {out}")


def prepare_mpii(n: int) -> None:
    """MPII qua Hugging Face (streaming, không tải toàn bộ ~12GB). Chỉ giữ ảnh cạnh ngắn >= 480px."""
    from datasets import load_dataset
    out = ROOT / "mpii" / "subset"
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("Voxel51/MPII_Human_Pose_Dataset", split="train", streaming=True).shuffle(seed=SEED, buffer_size=2000)
    cnt = 0
    for i, row in enumerate(ds):
        im = row["image"]
        if min(im.size) < 480:
            continue
        name = f"mpii_{i:06d}"
        im.convert("RGB").save(out / f"{name}.jpg", quality=92)
        (out / f"{name}.txt").write_text("a photo of a person in a natural pose", encoding="utf-8")
        cnt += 1
        if cnt >= n:
            break
    print(f"MPII: {cnt} ảnh -> {out}")


def prepare_ffhq(n: int) -> None:
    base = ROOT / "ffhq"
    sys.path.insert(0, str(ROOT / "_tools"))  # script chính thức của NVlabs đã tải về, chỉ dùng hàm tải
    import ffhq_dl as F  # noqa: E402
    os.chdir(base)
    if not (base / "ffhq-dataset-v2.json").exists():
        # File >100MB trên Google Drive hiện trả trang "Virus scan warning"; dùng URL tải trực tiếp công khai + confirm=t,
        # rồi kiểm size + MD5 theo script chính thức của NVlabs.
        import hashlib
        import requests
        spec = F.json_spec
        fid = spec["file_url"].split("id=")[1]
        tmp = base / "ffhq-dataset-v2.json.part"
        h = hashlib.md5()
        with requests.get("https://drive.usercontent.google.com/download", params={"id": fid, "export": "download", "confirm": "t"},
                          stream=True, timeout=120) as r, open(tmp, "wb") as f:
            r.raise_for_status()
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                h.update(chunk)
        if tmp.stat().st_size != spec["file_size"] or h.hexdigest() != spec["file_md5"]:
            raise SystemExit(f"FFHQ metadata sai size/MD5 ({tmp.stat().st_size} B) — có thể Drive đang giới hạn lượt tải; thử lại sau")
        tmp.replace(base / "ffhq-dataset-v2.json")
    data = json.load(open(base / "ffhq-dataset-v2.json", encoding="utf-8"))
    keys = sorted(data, key=int)
    random.Random(SEED).shuffle(keys)
    specs = [data[k]["image"] for k in keys[:n]]
    print(f"FFHQ: tải {len(specs)} ảnh 1024² ({sum(s['file_size'] for s in specs) / 2**20:.0f} MB) qua Google Drive")
    F.download_files(specs, num_threads=6)
    out = base / "subset"
    out.mkdir(exist_ok=True)
    cnt = 0
    for k, spec in zip(keys[:n], specs):
        src = base / spec["file_path"]
        if src.exists():
            dst = out / f"ffhq_{int(k):05d}.png"
            if not dst.exists():
                src.replace(dst)
            (out / f"ffhq_{int(k):05d}.txt").write_text("a close-up photo of a person's face, front view", encoding="utf-8")
            cnt += 1
    print(f"FFHQ: có {cnt}/{len(specs)} ảnh -> {out}")


# ------------------------------------------------------------------ import
def call(api, method, path, body=None):
    req = urllib.request.Request(api + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def upload_zip(api, dataset_id, zbytes, name):
    boundary = "----emvoox" + os.urandom(8).hex()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"{name}\"\r\n"
            f"Content-Type: application/zip\r\n\r\n").encode() + zbytes + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{api}/api/v1/datasets/{dataset_id}/uploads", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def import_dataset(kind: str, api: str, batch: int = 100) -> None:
    m = META[kind]
    folder = ROOT / kind / "subset"
    imgs = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    ds = next((d for d in call(api, "GET", "/api/v1/datasets") if d["name"] == m["name"]), None) or \
        call(api, "POST", "/api/v1/datasets", {"name": m["name"], "group": m["group"], "source": m["source"], "license": m["license"]})
    total = {"imported": 0, "duplicates": 0, "near_duplicates": 0, "low_resolution": 0, "rejected": 0}
    for i in range(0, len(imgs), batch):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
            for p in imgs[i:i + batch]:
                z.write(p, p.name)
                t = p.with_suffix(".txt")
                if t.exists():
                    z.write(t, t.name)
        r = upload_zip(api, ds["id"], buf.getvalue(), f"{kind}-{i // batch}.zip")
        for k in total:
            total[k] += r[k] if isinstance(r[k], int) else len(r[k])
        print(f"  lô {i // batch + 1}: +{r['imported']}", flush=True)
    print(f"{m['name']}: {total}  (dataset {ds['id']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "import"])
    ap.add_argument("kind", choices=list(META))
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--api", default="http://localhost:8010")
    a = ap.parse_args()
    if a.cmd == "prepare":
        {"unsplash": prepare_unsplash, "coco": prepare_coco, "ffhq": prepare_ffhq, "mpii": prepare_mpii}[a.kind](a.n)
    else:
        import_dataset(a.kind, a.api)
