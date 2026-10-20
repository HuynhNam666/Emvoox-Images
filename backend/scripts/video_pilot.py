"""Tạo lô pilot video: 6 shot đại diện × 2 seed × 2 cấu hình = 24 clip (hướng dẫn mục 6).

Dùng cùng một ảnh nguồn đã duyệt cho mọi clip để so sánh cấu hình. Script chỉ GỬI job qua API (idempotency key cố định
theo tên shot/seed/cấu hình nên chạy lại không tạo trùng); duyệt từng clip ở tab Video, rồi xem GET /api/v1/video/metrics.
Prompt dưới đây là mẫu minh họa, hãy thay bằng nội dung thật của phim.

  cd backend && PYTHONPATH=. ..\\.venv\\Scripts\\python scripts/video_pilot.py --keyframe <asset_id> [--api http://127.0.0.1:8000] [--dry-run]
"""
import argparse
import json
import sys
from urllib import request

GLOBAL = ("Cinematic shot of the same character shown in the input image. Preserve facial identity, hairstyle, clothing "
          "and background layout. A single continuous shot.")

SHOTS = {
    "close_expression": ["Calm neutral face, subtle breathing.", "His expression slowly changes to a soft smile.", "He holds the smile, eyes steady."],
    "head_turn": ["Looks forward, still.", "Slowly turns his head toward the doorway on the right.", "Pauses, expression becomes concerned."],
    "simple_walk": ["Stands still facing the camera.", "Takes slow steady steps forward.", "Stops, camera gently follows."],
    "hold_prop": ["Holds the object in both hands, looking down at it.", "Lifts the object a little toward the camera.", "Lowers it again, calm."],
    "two_people": ["Two people stand facing each other in silence.", "The person on the left speaks quietly, the other listens.", "The person on the right nods slowly."],
    "two_shot_sequence": ["Wide steady framing, characters in place.", "Camera pushes in slightly.", "Characters exchange a brief glance."],
}
CONFIGS = {  # hai cấu hình để so sánh; thay một biến mỗi lần sau khi chạy được
    "A_97f_512x896": {"width": 512, "height": 896, "frame_count": 97, "fps": 25},
    "B_49f_512x896": {"width": 512, "height": 896, "frame_count": 49, "fps": 25},
}
SEEDS = [101, 202]
CUTS = [(0, 25), (25, 65), (65, 100)]


def post(api: str, body: dict, key: str):
    req = request.Request(f"{api}/api/v1/video/jobs", data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json", "Idempotency-Key": key})
    with request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyframe", required=True, help="asset_id ảnh nguồn ĐÃ DUYỆT")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--budget", type=float, default=0.0, help="trần chi phí mỗi job (0 = không giới hạn)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    n = 0
    for shot, lines in SHOTS.items():
        for cname, cfg in CONFIGS.items():
            for seed in SEEDS:
                body = {"source_keyframe_asset_id": a.keyframe, "global_prompt": GLOBAL, "seed": seed, "budget_limit": a.budget,
                        "motion_segments": [{"start": s, "end": e, "prompt": t} for (s, e), t in zip(CUTS, lines)], **cfg}
                key = f"pilot:{a.keyframe}:{shot}:{cname}:{seed}"
                n += 1
                if a.dry_run:
                    print(n, shot, cname, seed)
                    continue
                try:
                    r = post(a.api, body, key)
                    print(n, shot, cname, seed, "->", r["job_id"], r["state"])
                except Exception as e:  # dừng sớm: lỗi thường là ảnh chưa duyệt hoặc ComfyUI chưa sẵn sàng
                    print(n, shot, cname, seed, "LỖI:", getattr(e, "read", lambda: b"")().decode("utf8", "replace") or e)
                    return 1
    print(f"{n} clip{' (dry-run)' if a.dry_run else ' đã xếp hàng'}. Duyệt ở tab Video; số liệu: {a.api}/api/v1/video/metrics")
    return 0


if __name__ == "__main__":
    sys.exit(main())
