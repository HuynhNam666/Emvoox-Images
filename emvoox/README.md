# EMVOOX Character Studio

Web app: Vietnamese character prompt → SDXL image → PostgreSQL (BYTEA) → dataset → Character LoRA → generation with the approved LoRA.

Read **[docs/STATUS.md](docs/STATUS.md)** first: it says exactly what is implemented, tested, and not yet verified.

## Architecture

```
React/Vite ── nginx ──► FastAPI (/api/v1) ──► PostgreSQL  (metadata + image bytes in asset_blobs.data BYTEA)
                            │
                            └─► Redis ──► Celery worker-gpu (queues: generate, train; concurrency 1)
                                            ├─ SDXLProvider (Diffusers, loaded once per process)
                                            └─ trainer adapter → pinned train_dreambooth_lora_sdxl.py (subprocess, argv list)
```

- Images are stored **in PostgreSQL** (`ASSET_STORAGE_BACKEND=postgres`, the only backend implemented in v1).
- Model weights, HF cache, training runs and LoRA `.safetensors` live on the `appdata` volume (`/data`); the DB keeps path, SHA-256 and metadata.
- One GPU job at a time: queue concurrency 1 plus a Redis lease with heartbeat (`app/ml/gpu_lock.py`).
- At-least-once delivery + idempotent task bodies; durable job row before enqueue; `sweep` task re-enqueues rows that never reached the broker and fails running rows whose heartbeat went silent.

## Run (CPU/web only, no GPU)

```powershell
copy .env.example .env        # then set POSTGRES_PASSWORD
docker compose up -d --build postgres redis api beat web
docker compose run --rm migrate          # explicit migration step; workers never migrate
```

Open http://localhost:8080. The sidebar shows **Worker AI: unavailable** until a worker is up; jobs wait in the queue.

## Run with the GPU worker

Needs Linux or WSL2 with NVIDIA Container Toolkit.

```bash
docker compose --profile gpu up -d --build worker-gpu
```

First generation downloads SDXL (~7 GB) into `appdata`. Accept the model license on Hugging Face if required and set `HF_TOKEN`. Pin `SDXL_REVISION` to a commit hash once you have verified the checkpoint.

CI / no-GPU demo only: `docker compose --profile mock up -d worker-mock` (images are stamped `MOCK` and the UI shows a warning banner).

## First end-to-end pass

1. **Tạo ảnh nhân vật** → pick preset fields / enter English prompt → *Biên dịch prompt* → *Tạo ảnh*.
2. *Lưu làm ảnh chuẩn* with a character code (`A01`).
3. **Dataset** → create dataset (trigger token e.g. `a01char`) → upload JPG/PNG/WebP (+ same-name `.txt` captions, or a ZIP) with a license statement → fix captions → *Duyệt* → *Tạo dataset version*.
4. **Huấn luyện LoRA** → pick the version → *Preflight* → *Bắt đầu*. Watch loss and log. When state is `awaiting_review`, inspect validation samples (assets of jobs with `purpose=validation`), then *Duyệt LoRA*.
5. Back in **Tạo ảnh**, choose the same base model; the approved LoRA appears in the LoRA list.

Verify bytes are in PostgreSQL:

```bash
docker compose exec postgres psql -U emvoox emvoox_images -c \
  "select a.id, a.storage_backend, a.byte_size, length(b.data) as blob_len from assets a join asset_blobs b on b.asset_id=a.id order by a.created_at desc limit 5;"
```

## Tests

```bash
cd backend
pip install -r requirements.txt fakeredis lupa
export DATABASE_URL=postgresql+psycopg://emvoox:<pw>@localhost:5432/emvoox_test   # DISPOSABLE db: the suite drops schema public
pytest -q
```

Tests run against real PostgreSQL with the **mock** provider and an in-memory Redis fake. They validate persistence, idempotency, state handling, dataset integrity and safety checks, not model quality. GPU smoke steps are in `docs/STATUS.md`.

## Backup / restore

```bash
docker compose exec -T postgres pg_dump -U emvoox -Fc emvoox_images > emvoox_$(date +%F).dump
docker run --rm -v emvoox-studio_appdata:/data -v "$PWD":/b alpine tar czf /b/appdata_$(date +%F).tgz -C /data .
# restore
docker compose exec -T postgres pg_restore -U emvoox -d emvoox_images --clean --if-exists < emvoox_YYYY-MM-DD.dump
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `provider_unavailable` on a job | Worker image has no CUDA or torch: use the `gpu` profile on a host with NVIDIA toolkit. |
| `cuda_oom` | Lower resolution/steps, keep `LOW_VRAM_OFFLOAD=true`, use the 8 GB preset, or a bigger GPU. Same config will not be retried automatically. |
| `connection refused` / auth error to DB | Inside Compose the host is `postgres`; on the host use `localhost:5432`. Check `POSTGRES_PASSWORD`. |
| Job stays `queued` | Worker offline or Redis down. `GET /api/v1/health/ready` reports each part. |
| Preflight `trainer_flags` fails | The pinned script does not accept a flag the adapter emits; fix `app/ml/trainer.py` against the script's `--help`. |
| Model download 401/403 | Accept the license on Hugging Face and set `HF_TOKEN`. |

## Known limits

No authentication/authorization (single-user, localhost only: do not expose). No IP-Adapter/ControlNet. Training resume returns 501. Prompt compiler is template-only. Upscaling is Lanczos crop/resize, not AI upscaling.
