"""Redis lease so only one heavy GPU job runs at a time (renewed by a heartbeat thread)."""
import threading
import uuid

import redis

from app.config import get_settings

KEY = "emvoox:gpu-lease"
_RELEASE = "if redis.call('get',KEYS[1])==ARGV[1] then return redis.call('del',KEYS[1]) else return 0 end"
_RENEW = "if redis.call('get',KEYS[1])==ARGV[1] then return redis.call('expire',KEYS[1],ARGV[2]) else return 0 end"


class GpuBusy(Exception):
    pass


class GpuLease:
    def __init__(self, owner: str):
        s = get_settings()
        self.r = redis.Redis.from_url(s.redis_url)
        self.ttl = s.gpu_lock_ttl_seconds
        self.token = f"{owner}:{uuid.uuid4()}"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        if not self.r.set(KEY, self.token, nx=True, ex=self.ttl):
            raise GpuBusy("GPU đang được job khác sử dụng.")
        self._thread = threading.Thread(target=self._beat, daemon=True)
        self._thread.start()
        return self

    def _beat(self):
        while not self._stop.wait(self.ttl / 3):
            try:
                self.r.eval(_RENEW, 1, KEY, self.token, self.ttl)
            except redis.RedisError:
                pass  # lease will expire; another job may take over safely

    def __exit__(self, *exc):
        self._stop.set()
        try:
            self.r.eval(_RELEASE, 1, KEY, self.token)
        except redis.RedisError:
            pass
