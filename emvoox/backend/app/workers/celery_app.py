import socket
import threading

import redis
from celery import Celery
from celery.signals import worker_ready, worker_shutdown

from app.config import get_settings

WORKER_KEY_PREFIX = "emvoox:worker:"
_stop = threading.Event()


def _beat(r: redis.Redis, key: str) -> None:
    # Independent of the task pool: a --pool=solo worker cannot answer `inspect ping` while it trains.
    while not _stop.is_set():
        try:
            r.set(key, get_settings().image_provider, ex=30)
        except redis.RedisError:
            pass
        _stop.wait(10)


@worker_ready.connect
def _start_presence(**_):
    r = redis.Redis.from_url(get_settings().redis_url)
    threading.Thread(target=_beat, args=(r, WORKER_KEY_PREFIX + socket.gethostname()), daemon=True).start()


@worker_shutdown.connect
def _stop_presence(**_):
    _stop.set()
    try:
        redis.Redis.from_url(get_settings().redis_url).delete(WORKER_KEY_PREFIX + socket.gethostname())
    except redis.RedisError:
        pass

s = get_settings()
celery_app = Celery("emvoox", broker=s.redis_url, backend=None, include=["app.workers.tasks"])
celery_app.conf.update(
    task_acks_late=True,  # redelivered if a worker dies; task bodies are idempotent
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_default_queue="generate",
    task_ignore_result=True,
    beat_schedule={"sweep": {"task": "app.workers.tasks.sweep", "schedule": 60.0}},
)
