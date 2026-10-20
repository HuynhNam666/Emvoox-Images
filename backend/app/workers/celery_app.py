from celery import Celery

from app.core.config import settings

celery_app = Celery("emvoox", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.task_acks_late = True  # at-least-once; task phải idempotent
celery_app.conf.task_routes = {
    "app.workers.tasks.generate": {"queue": "generation"},
    "app.workers.tasks.train": {"queue": "training"},
    "app.workers.tasks.generate_video": {"queue": "video"},
    "app.workers.tasks.visual_run": {"queue": "visual"},
}
celery_app.autodiscover_tasks(["app.workers"])


from celery.signals import worker_ready  # noqa: E402


@worker_ready.connect
def _on_ready(**_):
    try:
        from app.workers.tasks import report_worker_status
        report_worker_status()
    except Exception:  # noqa: BLE001 - DB chưa sẵn sàng lúc khởi động: sẽ báo lại ở job đầu tiên
        pass
