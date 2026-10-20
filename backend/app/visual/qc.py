"""Quality Gate (báo cáo mục 3.4): kiểm tra nhận dạng, đúng cảnh, lỗi hình thể và quy chuẩn ảnh.

Phần TỰ ĐỘNG chỉ bắt được lỗi kỹ thuật + so khớp khuôn mặt; "đúng hành động", "hình thể", "thẩm mỹ" vẫn cần người duyệt
(báo cáo: QC hỗ trợ, người phụ trách duyệt bản cuối). Kết quả mỗi check: pass | warn | fail | skipped (kèm lý do).
Verdict: fail nếu có check fail; review nếu có check bắt buộc bị skipped/warn; pass nếu mọi check đều đạt.
Ngưỡng nhận dạng là ngưỡng THAM KHẢO của FaceNet/VGGFace2 (chưa hiệu chuẩn cho A01/A02): chỉnh bằng QC_IDENTITY_PASS/FAIL.
"""
import hashlib
import io
import logging

import numpy as np
from PIL import Image

from app.core.config import settings

log = logging.getLogger("visual.qc")
_engine = None
_engine_err: str | None = None
_embed_cache: dict[str, np.ndarray | None] = {}


def _chk(name, status, value=None, threshold=None, msg=""):
    return {"name": name, "status": status, "value": value, "threshold": threshold, "message": msg}


class FaceEngine:
    def __init__(self):
        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1
        import os
        dev = os.environ.get("FACE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch, self.dev = torch, dev
        self.mtcnn = MTCNN(image_size=160, margin=20, device=dev, post_process=True, keep_all=True)
        self.net = InceptionResnetV1(pretrained="vggface2").eval().to(dev)

    def embeds(self, img: Image.Image) -> list[np.ndarray]:
        """Embedding chuẩn hoá của MỌI khuôn mặt tìm thấy (lớn trước)."""
        boxes, probs = self.mtcnn.detect(img)
        if boxes is None:
            return []
        order = sorted(range(len(boxes)), key=lambda i: -(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
        faces = self.mtcnn.extract(img, boxes[order], None)
        if faces is None:
            return []
        with self.torch.no_grad():
            e = self.net(faces.to(self.dev)).cpu().numpy()
        return [v / np.linalg.norm(v) for v in e]


def face_engine():
    """None nếu không dùng được (kèm lý do trong _engine_err). Không bao giờ ném lỗi."""
    global _engine, _engine_err
    if _engine is not None or _engine_err is not None:
        return _engine
    try:
        _engine = FaceEngine()
    except Exception as e:  # noqa: BLE001
        _engine_err = f"không nạp được FaceNet/MTCNN ({type(e).__name__}: {e})"
        log.warning(_engine_err)
    return _engine


def engine_error() -> str | None:
    return _engine_err


def embed_reference(data: bytes) -> np.ndarray | None:
    """Embedding mặt lớn nhất của ảnh tham chiếu (cache theo sha256)."""
    key = hashlib.sha256(data).hexdigest()
    if key in _embed_cache:
        return _embed_cache[key]
    eng = face_engine()
    out = None
    if eng:
        es = eng.embeds(Image.open(io.BytesIO(data)).convert("RGB"))
        out = es[0] if es else None
    _embed_cache[key] = out
    return out


def evaluate(png: bytes, *, target_aspect: tuple[int, int], expected_faces: int, ref_embeds: list[list[np.ndarray]],
             prior_embeds: list[np.ndarray] | None = None) -> dict:
    """ref_embeds: với mỗi nhân vật kỳ vọng, danh sách embedding ảnh chuẩn (có thể rỗng nếu không đọc được mặt).
    Trả {verdict, score, checks[]}."""
    checks: list[dict] = []
    try:
        img = Image.open(io.BytesIO(png))
        img.load()
        img = img.convert("RGB")
    except Exception as e:  # noqa: BLE001
        return {"verdict": "fail", "score": 0.0, "checks": [_chk("decode", "fail", msg=f"không đọc được ảnh: {e}")]}
    w, h = img.size
    checks.append(_chk("decode", "pass", f"{w}x{h}"))

    short = min(w, h)
    checks.append(_chk("min_resolution", "pass" if short >= 512 else "fail", short, 512))

    ratio, tratio = w / h, target_aspect[0] / target_aspect[1]
    crop = 1 - min(ratio / tratio, tratio / ratio)
    st = "pass" if crop <= 0.05 else ("warn" if crop <= 0.25 else "fail")
    checks.append(_chk("aspect", st, round(crop, 3), 0.25, "tỉ lệ phần phải cắt để vào khung đích (không kéo méo)"))

    g = np.asarray(img.convert("L").resize((512, max(1, int(512 * h / w)))), dtype=np.float32)
    std = float(g.std())
    checks.append(_chk("not_blank", "pass" if std >= 8 else "fail", round(std, 1), 8))
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    sharp = float(lap.var())
    checks.append(_chk("sharpness", "pass" if sharp >= 15 else ("warn" if sharp >= 5 else "fail"), round(sharp, 1), 15,
                       "phương sai Laplacian; thấp = mờ"))
    dark, bright = float((g < 5).mean()), float((g > 250).mean())
    checks.append(_chk("exposure", "pass" if max(dark, bright) < 0.4 else "warn", round(max(dark, bright), 3), 0.4))

    id_scores: list[float] = []
    if expected_faces > 0:
        eng = face_engine()
        if eng is None:
            checks.append(_chk("faces", "skipped", msg=engine_error() or "không có FaceNet"))
            checks.append(_chk("identity", "skipped", msg="không đo được nhận dạng tự động: cần người duyệt"))
        else:
            faces = eng.embeds(img)
            if len(faces) == 0:
                checks.append(_chk("faces", "fail", 0, expected_faces, "không thấy khuôn mặt"))
            else:
                checks.append(_chk("faces", "pass" if len(faces) == expected_faces else "warn", len(faces), expected_faces,
                                   "số mặt tìm thấy so với số nhân vật (mặt quay đi/che có thể không phát hiện được)"))
                # ghép tham lam từng nhân vật với mặt khớp nhất chưa dùng
                used: set[int] = set()
                for ci, refs in enumerate(ref_embeds):
                    refs = [r for r in refs if r is not None]
                    if not refs:
                        checks.append(_chk(f"identity_{ci + 1}", "skipped", msg="ảnh chuẩn không đọc được mặt"))
                        continue
                    best, bi = -1.0, -1
                    for fi, f in enumerate(faces):
                        if fi in used:
                            continue
                        s = max(float(np.dot(f, r)) for r in refs)
                        if s > best:
                            best, bi = s, fi
                    if bi < 0:
                        checks.append(_chk(f"identity_{ci + 1}", "fail", None, settings.qc_identity_pass, "thiếu người"))
                        continue
                    used.add(bi)
                    id_scores.append(best)
                    status = "pass" if best >= settings.qc_identity_pass else ("fail" if best < settings.qc_identity_fail else "warn")
                    checks.append(_chk(f"identity_{ci + 1}", status, round(best, 3), settings.qc_identity_pass,
                                       "cosine FaceNet với ảnh chuẩn (ngưỡng tham khảo)"))
                if prior_embeds:
                    s = max(float(np.dot(faces[0], p)) for p in prior_embeds)
                    checks.append(_chk("sequence_identity", "pass" if s >= settings.qc_identity_pass else "warn", round(s, 3),
                                       settings.qc_identity_pass, "so với các shot đã duyệt trước đó trong cùng tập"))
    else:
        checks.append(_chk("faces", "skipped", msg="shot không có nhân vật"))

    if any(c["status"] == "fail" for c in checks):
        verdict = "fail"
    else:
        # skipped chỉ đáng ngại khi shot có nhân vật (không đo được nhận dạng => cần người duyệt)
        unchecked = expected_faces > 0 and any(c["status"] == "skipped" for c in checks)
        verdict = "review" if unchecked or any(c["status"] == "warn" for c in checks) else "pass"
    score = float(np.mean(id_scores)) if id_scores else min(1.0, sharp / 100.0)
    return {"verdict": verdict, "score": round(score, 4), "checks": checks}


def failure_instruction(qc: dict) -> tuple[str, str]:
    """Từ kết quả QC => (chỉ dẫn sửa, bước quay lại): 'compile' (bước 2) hoặc 'render' (bước 4)."""
    fails = [c for c in qc["checks"] if c["status"] == "fail"]
    names = {c["name"] for c in fails}
    if any(n.startswith("identity") or n == "faces" for n in names):
        return ("strictly match the reference faces; keep each character's facial identity, hairstyle and clothing unchanged; "
                "show the face clearly and unobstructed", "compile")
    if "aspect" in names:
        return ("compose the full subject inside the frame with the requested orientation", "compile")
    return ("sharp focus, clean details, correct exposure, no blur", "render")
