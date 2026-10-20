"""Tải trọng số cho server. Chạy trong container (đã có volume models) hoặc trực tiếp.

  python scripts/download_models.py --list
  python scripts/download_models.py --groups core --dry-run          # chỉ in kế hoạch + kiểm đĩa
  python scripts/download_models.py --groups core                    # video + sdxl + klein + face
  python scripts/download_models.py --groups qwen,zimage             # nhóm nặng, tải khi cần
  python scripts/download_models.py --groups video --verify          # băm sha256 so với Hugging Face (chậm với file lớn)

- Tải có thể ngắt/tiếp tục (huggingface_hub resume); file đã đủ dung lượng thì bỏ qua.
- Kiểm dung lượng đĩa trống TRƯỚC khi tải; ghi MODELS.lock.json (repo, file, size, sha256 theo HF) để truy vết phiên bản.
- HF_TOKEN (tùy chọn) để tăng giới hạn tải; mọi repo trong danh mục hiện đều mở, không gated.
Đường dẫn: MODEL_CACHE_DIR (mặc định ./data/models); file cho ComfyUI nằm ở $MODEL_CACHE_DIR/comfyui/.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")
from app.core.config import settings  # noqa: E402
from app.visual.model_manifest import ALL, CORE, GROUPS  # noqa: E402


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick(spec: str) -> list[str]:
    if spec == "core":
        return CORE
    if spec == "all":
        return ALL
    names = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [n for n in names if n not in GROUPS]
    if bad:
        raise SystemExit(f"nhóm không có: {bad}; có: {ALL}")
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="core")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.list:
        for g, d in GROUPS.items():
            print(f"{g:7} ≈{d['size_gb']:>6.1f} GB  " + ", ".join(i.get("repo", i["type"]) for i in d["items"]))
        print(f"\ncore = {CORE}  (≈{sum(GROUPS[g]['size_gb'] for g in CORE):.0f} GB)   all ≈{sum(d['size_gb'] for d in GROUPS.values()):.0f} GB")
        return 0
    groups = pick(a.groups)
    root = Path(settings.model_cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    need = sum(GROUPS[g]["size_gb"] for g in groups)
    free = shutil.disk_usage(root).free / 2**30
    print(f"Nhóm: {groups} · cần ≈{need:.1f} GB · đĩa trống {free:.1f} GB tại {root}")
    if free < need * 1.1:
        print("KHÔNG ĐỦ ĐĨA TRỐNG (cần dư 10%). Dừng.")
        return 2
    if a.dry_run:
        for g in groups:
            for it in GROUPS[g]["items"]:
                print(f"  [{g}] {it['type']:11} {it.get('repo', 'facenet')} {it.get('file', '')} ≈{it['size_gb']} GB")
        return 0

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    lock = []
    for g in groups:
        for it in GROUPS[g]["items"]:
            t = it["type"]
            if t == "comfy_file":
                dest = root / "comfyui" / it["dest"]
                info = api.get_paths_info(it["repo"], [it["file"]], expand=True)[0]
                size, lfs = info.size, (info.lfs.sha256 if info.lfs else None)
                if dest.exists() and dest.stat().st_size == size:
                    print(f"  ✓ có sẵn {dest.name}")
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = root / "comfyui" / ".dl"
                    print(f"  ↓ {it['repo']}/{it['file']} ({size / 2**30:.2f} GB)")
                    got = hf_hub_download(it["repo"], it["file"], local_dir=str(tmp))
                    os.replace(got, dest)
                if dest.stat().st_size != size:
                    print(f"  ✗ sai dung lượng {dest} ({dest.stat().st_size} ≠ {size})")
                    return 3
                if a.verify and lfs:
                    ok = sha256_file(dest) == lfs
                    print(f"  {'✓' if ok else '✗'} sha256 {dest.name}")
                    if not ok:
                        return 3
                lock.append({"group": g, "repo": it["repo"], "file": it["file"], "dest": str(dest), "size": size, "sha256": lfs})
            elif t == "hf_snapshot":
                print(f"  ↓ snapshot {it['repo']} {it['allow'][:3]}…")
                path = snapshot_download(it["repo"], allow_patterns=it["allow"], cache_dir=str(root))
                lock.append({"group": g, "repo": it["repo"], "snapshot": path, "revision": Path(path).name})
            elif t == "facenet":
                os.environ.setdefault("TORCH_HOME", str(root / "torch"))
                try:
                    from facenet_pytorch import InceptionResnetV1
                    InceptionResnetV1(pretrained="vggface2")
                    print("  ✓ FaceNet VGGFace2")
                    lock.append({"group": g, "repo": "facenet-pytorch/vggface2"})
                except Exception as e:  # noqa: BLE001
                    print(f"  ✗ FaceNet: {e}  (Quality Gate sẽ báo 'không đo được nhận dạng' cho tới khi sửa)")
    (root / "MODELS.lock.json").write_text(json.dumps(lock, indent=1, ensure_ascii=False), encoding="utf8")
    print(f"Xong. Ghi {root / 'MODELS.lock.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
