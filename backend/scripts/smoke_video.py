"""Tạo MỘT clip video thật qua toàn bộ hệ thống để nghiệm thu server (và ghi lại số liệu thời gian/VRAM/RAM).

  python scripts/smoke_video.py --api http://localhost:8080 --token $API_TOKEN [--image anh.png] [--frames 49] [--out clip.mp4]

Làm gì: (1) dùng ảnh --image (hoặc tự dựng ảnh thử nếu không có) làm ảnh chuẩn của nhân vật SMOKE, nên đã thuộc diện "ảnh đã duyệt";
(2) gửi /video/jobs 2 đoạn chuyển động; (3) chờ render; (4) tải mp4 về --out; (5) in số liệu đo được (thời gian, VRAM, RAM).
Không tự duyệt clip và không đổi trạng thái workflow: sau khi XEM clip thấy ổn, đặt "_meta.status": "verified" trong
backend/app/ml/workflows/ltx23_promptrelay.api.json.
Dùng ảnh nhân vật thật (của A01/A02) cho kết quả có ý nghĩa; ảnh dựng sẵn chỉ để kiểm đường ống chạy được.
"""
import argparse
import io
import json
import sys
import time
import uuid
from urllib import error, request


def call(api, method, path, token, body=None, raw=False, headers=None):
    h = {"Authorization": f"Bearer {token}"} if token else {}
    h.update(headers or {})
    data = None
    if body is not None and not isinstance(body, bytes):
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    elif isinstance(body, bytes):
        data = body
    req = request.Request(api + path, data=data, method=method, headers=h)
    try:
        with request.urlopen(req, timeout=600) as r:
            out = r.read()
            return out if raw else json.loads(out or b"{}")
    except error.HTTPError as e:
        sys.exit(f"HTTP {e.code} {method} {path}: {e.read()[:600].decode('utf8', 'replace')}")


def multipart(name, filename, data, ctype="image/png"):
    b = uuid.uuid4().hex
    body = (f'--{b}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\nContent-Type: {ctype}\r\n\r\n').encode() + data \
        + f"\r\n--{b}--\r\n".encode()
    return body, {"Content-Type": f"multipart/form-data; boundary={b}"}


def synthetic_image() -> bytes:
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (768, 1344), (40, 50, 70))
    d = ImageDraw.Draw(im)
    for i in range(0, 1344, 8):
        d.rectangle([0, i, 768, i + 8], fill=(40 + i // 12, 60 + i // 20, 90))
    d.ellipse([234, 300, 534, 640], fill=(220, 190, 170))
    d.rectangle([184, 640, 584, 1100], fill=(245, 245, 245))
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8080")
    ap.add_argument("--token", default="")
    ap.add_argument("--image", help="ảnh khung đầu (nên là ảnh nhân vật thật, tỉ lệ 9:16)")
    ap.add_argument("--frames", type=int, default=49, help="8n+1; 49 ≈ 2 giây ở 25 fps (nhanh nhất để thử)")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=896)
    ap.add_argument("--out", default="smoke_clip.mp4")
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()

    h = call(a.api, "GET", "/api/v1/video/health", a.token)
    print("ComfyUI:", json.dumps({k: h.get(k) for k in ("ok", "workflow_status", "missing_nodes", "missing_model_files", "warning", "error")}, ensure_ascii=False))
    if h.get("error") or h.get("missing_nodes") or h.get("missing_model_files"):
        return 2

    data = open(a.image, "rb").read() if a.image else synthetic_image()
    body, hd = multipart("file", "smoke.png", data)
    asset = call(a.api, "POST", "/api/v1/assets?kind=reference", a.token, body, headers=hd)
    code = "SMOKE" + uuid.uuid4().hex[:4].upper()
    ch = call(a.api, "POST", "/api/v1/characters", a.token, {"code": code, "name": "smoke test", "identity": {"identity_en": "a person"}})
    call(a.api, "POST", f"/api/v1/characters/{ch['id']}/references", a.token, {"asset_id": asset["id"]})   # ảnh chuẩn => thuộc diện đã duyệt

    req = {"source_keyframe_asset_id": asset["id"], "width": a.width, "height": a.height, "frame_count": a.frames, "fps": 25, "seed": 42,
           "global_prompt": "Cinematic medium shot of the same person shown in the input image. Preserve the facial identity, hairstyle and clothing. A single continuous shot.",
           "motion_segments": [{"start": 0, "end": 50, "prompt": "The person looks forward and breathes naturally."},
                               {"start": 50, "end": 100, "prompt": "The person slowly turns their head to the right. The camera gently moves closer."}]}
    job = call(a.api, "POST", "/api/v1/video/jobs", a.token, req, headers={"Idempotency-Key": "smoke-" + uuid.uuid4().hex})
    print("job:", job["job_id"])
    t0 = time.time()
    while time.time() - t0 < a.timeout:
        j = call(a.api, "GET", f"/api/v1/video/jobs/{job['job_id']}", a.token)
        if j["state"] in ("review_required", "failed", "needs_human"):
            break
        print(f"  {int(time.time() - t0):5d}s  {j['state']}", flush=True)
        time.sleep(10)
    else:
        print("hết thời gian chờ (clip có thể vẫn đang render; xem tab Video)")
        return 3
    att = j["attempts"][-1]
    if j["state"] != "review_required":
        print("THẤT BẠI:", j["error"])
        return 1
    mp4 = call(a.api, "GET", f"/api/v1/assets/{att['output_asset_id']}/content", a.token, raw=True)
    open(a.out, "wb").write(mp4)
    m = att.get("metrics") or {}
    print(f"\nXONG: {a.out} ({len(mp4) / 1e6:.1f} MB) · tổng {att['total_seconds']:.0f}s (chờ render {m.get('wait_s')}s, tải {m.get('download_s')}s)")
    print(f"GPU đỉnh {m.get('gpu_peak_mb')} MB / {m.get('gpu_total_mb')} MB · RAM đỉnh {m.get('ram_peak_mb')} MB / {m.get('ram_total_mb')} MB · {m.get('gpu_name')}")
    print("Xem clip, nếu ổn: đặt _meta.status = verified trong ltx23_promptrelay.api.json, rồi duyệt ở tab Video. Số liệu: /reports")
    return 0


if __name__ == "__main__":
    sys.exit(main())
