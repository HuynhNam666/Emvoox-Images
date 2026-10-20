"""Kiểm tra workflow API đã ghim với một ComfyUI ĐANG CHẠY (không cần model, không cần GPU).

  python scripts/validate_comfy_workflow.py [--url http://127.0.0.1:8188]

1. Mọi class_type phải có trong /object_info (đã cài đủ node).
2. Mọi input trong graph phải là input hợp lệ của node (bắt tên input sai).
3. Mọi input bắt buộc phải có (trừ input động của KJNodes).
4. Gửi /prompt: lỗi duy nhất được chấp nhận là "file model chưa có trong danh sách" (chưa tải model).
Thoát mã 0 nếu cấu trúc đúng.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

sys.path.insert(0, ".")
from app.ml import comfy_client as C  # noqa: E402

MODEL_INPUTS = ("unet_name", "clip_name1", "clip_name2", "vae_name", "image")   # lỗi "chưa có file model" được chấp nhận ở bước thử /prompt


def get(url):
    return json.load(urllib.request.urlopen(url, timeout=60))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    a = ap.parse_args()
    wf = C.load_workflow()
    oi = get(a.url + "/object_info")
    bad = C.structure_errors(wf, oi)
    graph = C.build_prompt(
        wf, image_name="example.png", width=512, height=896, frame_count=97, fps=25, seed=1, global_prompt="g",
        segments=[{"start": 0, "end": 40, "prompt": "a"}, {"start": 40, "end": 100, "prompt": "b"}],
        audio_mode="generated", filename_prefix="validate")
    req = urllib.request.Request(a.url + "/prompt", data=json.dumps({"prompt": graph}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=60))
        print("ComfyUI nhận prompt:", r.get("prompt_id"), "(model đã có đủ)")
    except urllib.error.HTTPError as e:
        res = json.loads(e.read())
        errs = res.get("node_errors", {})
        for nid, ne in errs.items():
            for er in ne.get("errors", []):
                info = er.get("extra_info", {})
                if er.get("type") == "value_not_in_list" and info.get("input_name") in MODEL_INPUTS:
                    print(f"  (chấp nhận) node {nid}: chưa có file {info['input_name']} = {info.get('received_value')}")
                else:
                    bad.append(f"node {nid} ({graph[nid]['class_type']}): {er.get('type')}: {er.get('message')} {er.get('details')}")
        if not errs:
            bad.append("lỗi /prompt: " + json.dumps(res)[:400])
    print("\n".join(bad) or "OK: cấu trúc workflow khớp schema ComfyUI")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
