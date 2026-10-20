"""Client ComfyUI (HTTP, stdlib) + dựng prompt API từ workflow đã ghim.

ComfyUI chạy như dịch vụ riêng; EMVOOX chỉ gọi /upload/image, /prompt, /history, /view, /free, /object_info.
Client không bao giờ nhận workflow tùy ý từ người dùng: chỉ điền các tham số khai báo trong file mapping.
"""
import copy
import hashlib
import json
import mimetypes
import time
import uuid
from pathlib import Path
from urllib import error as urlerror, parse, request

from app.core.config import settings

WORKFLOW_DIR_DEFAULT = Path(__file__).parent / "workflows"
MAX_SEGMENTS = 6
MAX_FRAMES = 257


class ComfyError(Exception):
    """Lỗi giao tiếp/thực thi ComfyUI. `code` dùng làm tiền tố thông báo lỗi của job."""

    def __init__(self, code: str, msg: str):
        super().__init__(msg)
        self.code = code


# ---------- kiểm tra tham số ----------
def validate_video_params(width: int, height: int, frame_count: int, fps: int) -> None:
    if width % 32 or height % 32:
        raise ValueError("width/height phải chia hết cho 32")
    if (frame_count - 1) % 8 or frame_count < 9:
        raise ValueError("frame_count phải có dạng 8n+1 (vd 49, 97, 121)")
    if frame_count > MAX_FRAMES:
        raise ValueError(f"frame_count tối đa {MAX_FRAMES}")
    if not 8 <= fps <= 60:
        raise ValueError("fps phải trong khoảng 8–60")


def validate_segments(segments: list[dict]) -> list[dict]:
    """Đoạn chuyển động theo TỶ LỆ % (0–100), liền nhau, phủ đủ 0→100. Không phải frame hay giây."""
    if not 1 <= len(segments) <= MAX_SEGMENTS:
        raise ValueError(f"cần 1–{MAX_SEGMENTS} đoạn chuyển động")
    out, cursor = [], 0.0
    for i, s in enumerate(segments):
        a, b, text = float(s["start"]), float(s["end"]), str(s["prompt"]).strip()
        if not text or "|" in text or "[" in text or "]" in text:
            raise ValueError(f"đoạn {i + 1}: prompt rỗng hoặc chứa ký tự dành riêng | [ ]")
        if abs(a - cursor) > 1e-6 or b <= a or b > 100:
            raise ValueError(f"đoạn {i + 1}: các đoạn phải liền nhau, tăng dần và kết thúc ở 100 "
                             f"(nhận {a}-{b}, mong đợi bắt đầu {cursor})")
        out.append({"start": a, "end": b, "prompt": text})
        cursor = b
    if abs(cursor - 100) > 1e-6:
        raise ValueError("đoạn cuối phải kết thúc ở 100")
    return out


def segment_frame_lengths(segments: list[dict], frame_count: int) -> list[int]:
    """Đổi các đoạn theo % thành số frame (pixel-space) có tổng đúng bằng frame_count (largest remainder).
    PromptRelayEncode nhận "segment_lengths" dạng '31, 39, 27' và tự quy đổi sang latent."""
    exact = [(s["end"] - s["start"]) * frame_count / 100 for s in segments]
    out = [int(e) for e in exact]
    for i in sorted(range(len(exact)), key=lambda i: -(exact[i] - out[i]))[: frame_count - sum(out)]:
        out[i] += 1
    for i in range(len(out)):  # mỗi đoạn tối thiểu 1 frame
        if out[i] < 1:
            j = max(range(len(out)), key=lambda k: out[k])
            out[j] -= 1
            out[i] = 1
    return out


def _num(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def compile_local_prompts(segments: list[dict], fmt: str = "pipe_labeled") -> str:
    """Chuỗi cho Prompt Relay Encode Smart: các đoạn cách nhau bằng '|'.
    pipe_labeled: '[0-25] text | [25-65] text'  ·  pipe_plain: 'text | text'."""
    if fmt == "pipe_plain":
        return " | ".join(s["prompt"] for s in segments)
    return " | ".join(f"[{_num(s['start'])}-{_num(s['end'])}] {s['prompt']}" for s in segments)


# ---------- workflow ----------
def workflow_dir() -> Path:
    return Path(settings.video_workflow_dir) if settings.video_workflow_dir else WORKFLOW_DIR_DEFAULT


def load_workflow(name: str | None = None) -> dict:
    name = name or settings.video_workflow_name
    base = workflow_dir()
    graph_path, map_path = base / f"{name}.api.json", base / f"{name}.mapping.json"
    if not graph_path.exists() or not map_path.exists():
        raise ComfyError("workflow_missing", f"không tìm thấy {graph_path.name}/{map_path.name} trong {base}")
    raw = graph_path.read_bytes() + map_path.read_bytes()
    graph = json.loads(graph_path.read_text(encoding="utf8"))
    mapping = json.loads(map_path.read_text(encoding="utf8"))
    meta = graph.pop("_meta", {})
    for key, targets in mapping["params"].items():
        for node_id, inp in targets:
            if node_id not in graph or inp not in graph[node_id].get("inputs", {}):
                raise ComfyError("workflow_mapping_invalid",
                                 f"mapping '{key}' trỏ tới node {node_id}.{inp} không có trong workflow")
    if mapping.get("output_node") not in graph:
        raise ComfyError("workflow_mapping_invalid", "output_node không có trong workflow")
    return {
        "graph": graph, "mapping": mapping, "meta": meta,
        "version": f"{mapping['version']}+{hashlib.sha256(raw).hexdigest()[:12]}",
        "model_version": mapping.get("model_version"),
        "status": meta.get("status", "unverified_draft"),
        "verified": meta.get("status") in ("schema_validated", "verified"),   # schema_validated: khớp /object_info; verified: đã render clip thật
    }


def build_prompt(wf: dict, *, image_name: str, width: int, height: int, frame_count: int, fps: int, seed: int,
                 global_prompt: str, segments: list[dict], audio_mode: str, filename_prefix: str) -> dict:
    """Điền tham số khai báo vào bản sao của graph. Trả về prompt đúng định dạng API của ComfyUI."""
    graph = copy.deepcopy(wf["graph"])
    mp = wf["mapping"]
    values = {
        "image": image_name, "width": width, "height": height, "frame_count": frame_count, "fps": fps, "seed": seed,
        "global_prompt": global_prompt,
        "local_prompts": compile_local_prompts(segments, mp.get("segment_format", "pipe_labeled")),
        "filename_prefix": filename_prefix,
        "segment_lengths": ", ".join(str(n) for n in segment_frame_lengths(segments, frame_count)),
    }
    for key, v in values.items():
        for node_id, inp in mp["params"].get(key, []):
            graph[node_id]["inputs"][inp] = v
    if audio_mode == "none":
        for node_id, inp in mp.get("audio_inputs_dropped_when_no_audio", []):
            graph[node_id]["inputs"].pop(inp, None)
    return graph


# ---------- HTTP ----------
def _url(path: str) -> str:
    return settings.comfyui_url.rstrip("/") + path


def _call(req: request.Request, timeout: float = 60) -> bytes:
    try:
        with request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urlerror.HTTPError as e:
        body = e.read().decode("utf8", "replace")[:1500]
        raise ComfyError("comfy_http_error", f"{e.code} {req.full_url}: {body}")
    except (urlerror.URLError, TimeoutError, ConnectionError, OSError) as e:
        raise ComfyError("comfy_unreachable", f"không kết nối được ComfyUI tại {settings.comfyui_url}: {e}")


def _json(path: str, payload: dict | None = None, timeout: float = 60):
    data = json.dumps(payload).encode() if payload is not None else None
    req = request.Request(_url(path), data=data, headers={"Content-Type": "application/json"} if data else {})
    return json.loads(_call(req, timeout) or b"null")


def upload_image(data: bytes, filename: str) -> str:
    boundary = uuid.uuid4().hex
    ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    parts = []
    for name, val in (("type", "input"), ("overwrite", "true")):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{val}\r\n'.encode())
    parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{filename}"\r\n'
                  f"Content-Type: {ctype}\r\n\r\n").encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    req = request.Request(_url("/upload/image"), data=b"".join(parts),
                          headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    res = json.loads(_call(req))
    name = res.get("name")
    if not name:
        raise ComfyError("comfy_upload_failed", f"ComfyUI không trả tên file: {res}")
    return f"{res['subfolder']}/{name}" if res.get("subfolder") else name


def missing_nodes(graph: dict) -> list[str]:
    """Các class_type chưa cài trong ComfyUI (đối chiếu /object_info)."""
    info = _json("/object_info", timeout=120)
    return sorted({n["class_type"] for n in graph.values() if n["class_type"] not in info})


def queue_prompt(graph: dict, client_id: str) -> str:
    res = _json("/prompt", {"prompt": graph, "client_id": client_id})
    if res.get("node_errors"):
        raise ComfyError("comfy_node_errors", json.dumps(res["node_errors"])[:1500])
    pid = res.get("prompt_id")
    if not pid:
        raise ComfyError("comfy_queue_failed", f"ComfyUI không trả prompt_id: {res}")
    return pid


def history(prompt_id: str) -> dict | None:
    return (_json(f"/history/{parse.quote(prompt_id)}") or {}).get(prompt_id)


def wait_for(prompt_id: str, timeout_s: float | None = None, poll_s: float | None = None, on_tick=None) -> dict:
    """Chờ tới khi /history/{id} có kết quả. Timeout KHÔNG hủy prompt trên ComfyUI: attempt vẫn giữ prompt_id để đối chiếu."""
    limit = timeout_s or settings.comfyui_timeout_s
    deadline = time.monotonic() + limit
    poll = poll_s if poll_s is not None else settings.comfyui_poll_s
    while time.monotonic() < deadline:
        h = history(prompt_id)
        if h:
            st = h.get("status", {})
            if st.get("status_str") == "error":
                msgs = [m for m in st.get("messages", []) if m and m[0] == "execution_error"]
                raise ComfyError("comfy_execution_error", json.dumps(msgs[-1][1])[:1200] if msgs else "execution_error")
            if st.get("completed") or h.get("outputs"):
                return h
        if on_tick:
            on_tick()
        time.sleep(poll)
    raise ComfyError("comfy_timeout", f"quá {limit}s chờ prompt_id={prompt_id} (prompt có thể vẫn đang chạy)")


def find_output_file(hist: dict, node_id: str) -> dict:
    out = hist.get("outputs", {}).get(node_id, {})
    for key in ("gifs", "videos", "images"):
        for f in out.get(key, []):
            if str(f.get("filename", "")).lower().endswith((".mp4", ".webm", ".mov")):
                return f
    raise ComfyError("comfy_no_output", f"node {node_id} không trả file video: {json.dumps(out)[:500]}")


def fetch_file(f: dict, max_bytes: int | None = None) -> bytes:
    q = parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")})
    data = _call(request.Request(_url(f"/view?{q}")), timeout=300)
    if max_bytes and len(data) > max_bytes:
        raise ComfyError("video_too_large", f"video {len(data)} bytes vượt giới hạn {max_bytes}")
    head = data[:64]
    if b"ftyp" not in head and not head.startswith(b"\x1a\x45\xdf\xa3"):
        raise ComfyError("comfy_bad_output", "file trả về không giống MP4/WebM")
    return data


def mime_for(filename: str) -> str:
    return "video/webm" if filename.lower().endswith(".webm") else "video/mp4"


def free_memory() -> None:
    try:
        _json("/free", {"unload_models": True, "free_memory": True}, timeout=15)
    except ComfyError:
        pass  # nhả VRAM là best-effort


# ---------- kiểm tra workflow với ComfyUI đang chạy (dùng bởi scripts/validate_comfy_workflow.py và server_selfcheck.py) ----------
MODEL_INPUTS = {"UNETLoader": ["unet_name"], "DualCLIPLoader": ["clip_name1", "clip_name2"], "VAELoaderKJ": ["vae_name"]}


def structure_errors(wf: dict, oi: dict) -> list[str]:
    """Lỗi cấu trúc: thiếu node, input lạ, thiếu input bắt buộc. oi = /object_info. Không xét file model."""
    bad = []
    for nid, n in wf["graph"].items():
        info = oi.get(n["class_type"])
        if not info:
            bad.append(f"node {nid}: ComfyUI chưa cài node {n['class_type']}")
            continue
        req, opt = info.get("input", {}).get("required", {}), info.get("input", {}).get("optional", {})
        allowed = {**req, **opt}
        for k in n["inputs"]:
            # VHS_VideoCombine nhận thêm widget phụ thuộc định dạng; LTXVImgToVideoInplaceKJ có input động num_images.*
            if k not in allowed and not k.startswith("num_images.") and not (n["class_type"] == "VHS_VideoCombine" and k in
                                                                              ("pix_fmt", "crf", "save_metadata", "trim_to_audio")):
                bad.append(f"node {nid} {n['class_type']}: input lạ '{k}'")
        for k in req:
            if k not in n["inputs"] and k != "num_images":
                bad.append(f"node {nid} {n['class_type']}: thiếu input bắt buộc '{k}'")
        if n["class_type"] == "VHS_VideoCombine" and "video/h264-mp4" not in str((allowed.get("format") or [[]])[0]):
            bad.append("VHS chưa có định dạng video/h264-mp4: ComfyUI cần ffmpeg")
    return bad


def missing_model_files(wf: dict, oi: dict) -> list[str]:
    """File model mà workflow gọi nhưng loader của ComfyUI chưa thấy (chưa tải hoặc đặt sai thư mục)."""
    out = []
    for nid, n in wf["graph"].items():
        for inp in MODEL_INPUTS.get(n["class_type"], []):
            opts = ((oi.get(n["class_type"], {}).get("input", {}).get("required", {}).get(inp) or [[]])[0]) or []
            if n["inputs"][inp] not in opts:
                out.append(f"{n['class_type']}.{inp} = {n['inputs'][inp]}")
    return out
