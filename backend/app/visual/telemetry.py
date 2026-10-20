"""Đo tài nguyên thật: thời gian, VRAM, RAM, tải GPU (báo cáo cần "GPU, VRAM, số bước và batch").

- ResourceSampler: lấy mẫu mỗi `interval` giây trong lúc job chạy (nvidia-smi + psutil). Không có nvidia-smi => trường GPU = None
  (không bịa số).
- ComfySampler: với video, ComfyUI chạy ở tiến trình/container khác nên hỏi /system_stats của chính nó.
- system_snapshot(): ảnh chụp RAM/đĩa/GPU hiện tại cho trang báo cáo.
"""
import json
import shutil
import subprocess
import threading
import time
from urllib import request

try:
    import psutil
except ImportError:  # pragma: no cover - psutil nằm trong requirements, nhánh này chỉ để không sập khi thiếu
    psutil = None

_SMI = shutil.which("nvidia-smi")


def gpu_query() -> list[dict]:
    """[{name,total_mb,used_mb,util}] cho từng GPU; rỗng nếu không có nvidia-smi."""
    if not _SMI:
        return []
    try:
        out = subprocess.run(
            [_SMI, "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            try:
                gpus.append({"name": parts[0], "total_mb": float(parts[1]), "used_mb": float(parts[2]), "util": float(parts[3])})
            except ValueError:
                pass
    return gpus


def _proc_rss_mb() -> float | None:
    if not psutil:
        return None
    p = psutil.Process()
    total = p.memory_info().rss
    for c in p.children(recursive=True):
        try:
            total += c.memory_info().rss
        except Exception:  # noqa: BLE001
            pass
    return total / 2**20


class ResourceSampler:
    """with ResourceSampler() as rs: ...; rs.summary() -> dict. Luôn có wall_s; phần còn lại có khi đo được."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._stop = threading.Event()
        self._t: threading.Thread | None = None
        self.t0 = 0.0
        self.t1 = 0.0
        self.gpu_name = None
        self.gpu_total_mb = None
        self.gpu_peak_mb = None
        self.gpu_base_mb = None
        self._util: list[float] = []
        self.ram_peak_mb = None
        self.ram_total_mb = psutil.virtual_memory().total / 2**20 if psutil else None
        self.rss_peak_mb = None
        self.samples = 0

    def _sample(self) -> None:
        g = gpu_query()
        if g:
            g0 = g[0]
            self.gpu_name, self.gpu_total_mb = g0["name"], g0["total_mb"]
            self.gpu_peak_mb = max(self.gpu_peak_mb or 0, g0["used_mb"])
            if self.gpu_base_mb is None:
                self.gpu_base_mb = g0["used_mb"]
            self._util.append(g0["util"])
        if psutil:
            used = psutil.virtual_memory().used / 2**20
            self.ram_peak_mb = max(self.ram_peak_mb or 0, used)
            rss = _proc_rss_mb()
            if rss is not None:
                self.rss_peak_mb = max(self.rss_peak_mb or 0, rss)
        self.samples += 1

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def __enter__(self):
        self.t0 = time.perf_counter()
        self._sample()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)
        self._sample()
        self.t1 = time.perf_counter()
        return False

    def summary(self) -> dict:
        r = lambda x: None if x is None else round(x, 1)  # noqa: E731
        return {
            "wall_s": round((self.t1 or time.perf_counter()) - self.t0, 3),
            "gpu_name": self.gpu_name, "gpu_total_mb": r(self.gpu_total_mb), "gpu_peak_mb": r(self.gpu_peak_mb),
            "gpu_base_mb": r(self.gpu_base_mb),
            "gpu_util_avg": r(sum(self._util) / len(self._util)) if self._util else None,
            "ram_peak_mb": r(self.ram_peak_mb), "ram_total_mb": r(self.ram_total_mb), "rss_peak_mb": r(self.rss_peak_mb),
            "samples": self.samples,
        }


class ComfySampler:
    """Hỏi /system_stats của ComfyUI trong lúc render để biết VRAM/RAM của máy chạy ComfyUI."""

    def __init__(self, base_url: str, interval: float = 1.0):
        self.url = base_url.rstrip("/") + "/system_stats"
        self.interval = interval
        self._stop = threading.Event()
        self._t: threading.Thread | None = None
        self.t0 = self.t1 = 0.0
        self.vram_total = self.vram_min_free = self.ram_total = self.ram_min_free = None
        self.gpu_name = None
        self.samples = 0

    def _sample(self) -> None:
        try:
            with request.urlopen(self.url, timeout=3) as r:
                d = json.loads(r.read())
        except Exception:  # noqa: BLE001
            return
        dev = (d.get("devices") or [{}])[0]
        sysd = d.get("system", {})
        self.gpu_name = dev.get("name", self.gpu_name)
        if dev.get("vram_total"):
            self.vram_total = dev["vram_total"] / 2**20
            free = dev.get("vram_free", dev["vram_total"]) / 2**20
            self.vram_min_free = free if self.vram_min_free is None else min(self.vram_min_free, free)
        if sysd.get("ram_total"):
            self.ram_total = sysd["ram_total"] / 2**20
            free = sysd.get("ram_free", sysd["ram_total"]) / 2**20
            self.ram_min_free = free if self.ram_min_free is None else min(self.ram_min_free, free)
        self.samples += 1

    def __enter__(self):
        self.t0 = time.perf_counter()
        self._sample()
        self._t = threading.Thread(target=lambda: [self._sample() for _ in iter(lambda: self._stop.wait(self.interval), True)], daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join(timeout=3)
        self._sample()
        self.t1 = time.perf_counter()
        return False

    def summary(self) -> dict:
        r = lambda x: None if x is None else round(x, 1)  # noqa: E731
        peak = lambda total, free: None if total is None or free is None else total - free  # noqa: E731
        return {
            "wall_s": round((self.t1 or time.perf_counter()) - self.t0, 3), "gpu_name": self.gpu_name,
            "gpu_total_mb": r(self.vram_total), "gpu_peak_mb": r(peak(self.vram_total, self.vram_min_free)),
            "ram_total_mb": r(self.ram_total), "ram_peak_mb": r(peak(self.ram_total, self.ram_min_free)),
            "samples": self.samples, "source": "comfyui /system_stats",
        }


def system_snapshot(paths: list[str] | None = None) -> dict:
    snap: dict = {"gpus": gpu_query()}
    if psutil:
        vm = psutil.virtual_memory()
        snap["ram"] = {"total_mb": round(vm.total / 2**20), "used_mb": round(vm.used / 2**20), "available_mb": round(vm.available / 2**20)}
        snap["cpu_percent"] = psutil.cpu_percent(interval=0.2)
        snap["cpu_count"] = psutil.cpu_count()
    disks = {}
    for p in paths or ["."]:
        try:
            u = shutil.disk_usage(p)
            disks[p] = {"total_gb": round(u.total / 2**30, 1), "free_gb": round(u.free / 2**30, 1)}
        except OSError:
            disks[p] = None
    snap["disks"] = disks
    return snap
