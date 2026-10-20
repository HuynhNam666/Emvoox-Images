"""Đo độ giống khuôn mặt bằng FaceNet (facenet-pytorch: MTCNN + InceptionResnetV1/VGGFace2).

Chỉ là CÔNG CỤ HỖ TRỢ: embedding huấn luyện trên mặt thật, ảnh AI có thể lệch; không có ngưỡng "cùng người"
đã hiệu chuẩn cho dự án này. Dùng để so sánh tương đối (LoRA vs không LoRA) và gợi ý ảnh ngoại lai.
"""
import io
import json
import sys
import urllib.request

import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

import os
_dev = os.environ.get("FACE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
_mtcnn = MTCNN(image_size=160, margin=20, device=_dev, post_process=True, select_largest=True)
_net = InceptionResnetV1(pretrained="vggface2").eval().to(_dev)


def fetch(api: str, asset_id: str) -> Image.Image:
    with urllib.request.urlopen(f"{api}/api/v1/assets/{asset_id}/content", timeout=60) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


@torch.no_grad()
def embed(img: Image.Image):
    """Trả vector chuẩn hóa hoặc None nếu không thấy mặt."""
    face = _mtcnn(img)
    if face is None:
        return None
    e = _net(face.unsqueeze(0).to(_dev))[0].cpu().numpy()
    return e / np.linalg.norm(e)


def cos(a, b) -> float:
    return float(np.dot(a, b))


if __name__ == "__main__":
    api = sys.argv[1]
    plan = json.load(open(sys.argv[2], encoding="utf-8"))
    rows = []
    for j in plan["jobs"]:
        e = embed(fetch(api, j["asset_id"]))
        rows.append({**j, "emb": e})
    anchor = next(r["emb"] for r in rows if r["caption"].startswith("front view close-up") and r["emb"] is not None)
    have = [r["emb"] for r in rows if r["emb"] is not None]
    mean = np.mean(have, axis=0)
    mean /= np.linalg.norm(mean)
    out = []
    for r in rows:
        out.append({"group": r["group"], "caption": r["caption"], "asset_id": r["asset_id"],
                    "face": r["emb"] is not None,
                    "sim_anchor": None if r["emb"] is None else round(cos(r["emb"], anchor), 3),
                    "sim_mean": None if r["emb"] is None else round(cos(r["emb"], mean), 3)})
    json.dump(out, open(sys.argv[3], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for o in out:
        print(f"{o['group']:11} {o['caption'][:42]:42} face={int(o['face'])} anchor={o['sim_anchor']} mean={o['sim_mean']}")
