# EMVOOX Character Studio

Ứng dụng web để **tạo ảnh nhân vật bằng SDXL**, quản lý hồ sơ nhân vật + ảnh chuẩn, quản lý **dataset 8 nhóm** (upload, caption, duyệt, version bất biến), **train Character LoRA**, và **giữ diện mạo bằng IP-Adapter**. Ảnh và toàn bộ lịch sử lưu trong **PostgreSQL** (bytes ảnh trong cột BYTEA).

> Phạm vi hiện tại: nghiên cứu/nội bộ, một máy, một GPU. **Chưa có đăng nhập** — chỉ mở trên localhost, không đưa ra Internet.
> Thư mục con `emvoox/` trong repo là một mã nguồn khác có sẵn từ trước; dự án này không dùng và không sửa nó.

Đặc tả gốc: `EMVOOX_Prompt_Claude.md`. Danh mục dataset công khai: [docs/DATASETS.md](docs/DATASETS.md).

## Mới (20/10/2026): chạy trên server, pipeline ảnh theo báo cáo EMVOOX1, video thật
- **Triển khai server:** [docs/SERVER_DEPLOY.md](docs/SERVER_DEPLOY.md) — `.env.server.example`, Compose có `comfyui` (đã ghim commit), `models-init` (tải model), `caddy` (HTTPS), đăng nhập bằng `API_TOKEN`, `scripts/server_selfcheck.py`, `scripts/smoke_video.py`.
- **Đối chiếu từng yêu cầu của báo cáo:** [docs/DOI_CHIEU_BAO_CAO.md](docs/DOI_CHIEU_BAO_CAO.md) (có/chưa/đã kiểm chứng đến đâu).
- **Pipeline ảnh** (`/api/v1/visual/jobs`): Character/Scene/Costume Library → Prompt Compiler → **Model Router** (13 model: SDXL, FLUX.2 Klein/Pro/Klein-API, "FLUX 3 Image", Gemini Nano Banana, Seedream, GPT Image, Qwen-Image(-Edit), Z-Image, Midjourney thủ công) → Render Worker → **Quality Gate** (FaceNet) → sửa tự động ≤2 lượt trong ngân sách → người duyệt → xuất bản có **nhãn AI + C2PA**.
- **Báo cáo chi phí, thời gian, VRAM, RAM:** tab Báo cáo, `/api/v1/reports/summary`, `.md`, `.csv`.
- **Benchmark chấm mù** 30 cảnh × 4 lần × các model; **rà quyền dataset** (`COMMERCIAL_USE`); `scripts/expand_library.py` mở rộng thư viện lên 200–300 ảnh/nhân vật.
- Trạng thái trung thực: workflow video khớp schema ComfyUI thật nhưng **chưa render clip**; image Docker, model tự chạy mới (Klein/Qwen/Z-Image) và các API ngoài **chưa chạy thật** — xem mục 7 của SERVER_DEPLOY.

## Đã làm được gì (và đã kiểm chứng đến đâu)

| Hạng mục | Trạng thái |
|---|---|
| Tạo ảnh SDXL 1.0 qua job bất đồng bộ (idempotency, lưu BYTEA, 1:1/9:16/16:9) | Chạy thật trên RTX 4060 8GB (~30 s/ảnh 768²) |
| Thư viện nhân vật (version bất biến, ảnh chuẩn; A01/A02 là hồ sơ mẫu chỉ có mô tả) | Chạy thật qua UI + Postgres |
| Dataset Manager (8 nhóm, ZIP an toàn, trùng/gần trùng, ảnh thấp phân giải, version + manifest) | pytest + chạy thật qua UI |
| Training Center (preflight GPU qua worker, SDXL DreamBooth-LoRA diffusers v0.41.0, loss thật, hủy/resume, registry, duyệt LoRA) | Chạy thật: 30 bước và 300 bước; hủy + resume từ checkpoint đã thử 1 lần |
| IP-Adapter Plus Face: dùng ảnh chuẩn làm tham chiếu | Chạy thật; xem số đo bên dưới |
| **Video** ảnh→clip qua ComfyUI + LTX-2.3 + PromptRelay: bảng `video_jobs/attempts`, API `/api/v1/video/*`, task worker, tab Video, duyệt/loại có lý do, tối đa 2 lượt sửa + trần ngân sách, metrics pilot | pytest với **ComfyUI giả**; **chưa render bằng LTX-2.3 thật**, workflow là bản nháp, 8GB VRAM chưa xác nhận — xem [docs/VIDEO_WORKFLOW.md](docs/VIDEO_WORKFLOW.md) |
| Docker Compose (postgres, redis, api, web, worker GPU) | Đã build và chạy thật trên Docker Desktop + RTX 4060 (xem "Kiểm tra Compose") |

Kết quả đo nhận dạng A02 (FaceNet, 6 cảnh chưa học, mẫu nhỏ): không LoRA + mô tả chữ 0,69 · LoRA 300 bước + mô tả chữ 0,80 · **IP-Adapter + mô tả chữ 0,87**. LoRA chỉ-trigger chưa đủ (0,51). Chi tiết và các hạn chế: [docs/EVAL_A02.md](docs/EVAL_A02.md), [docs/IP_ADAPTER.md](docs/IP_ADAPTER.md).

## Kiến trúc

```
Trình duyệt ──► web (nginx) ──/api──► api (FastAPI) ──► PostgreSQL  (nhân vật, job, dataset, ảnh BYTEA, registry)
                                         │
                                         └──► Redis (hàng đợi Celery) ──► worker-gpu (SDXL, LoRA, IP-Adapter, trainer)
                                                                               └─ volume: weights (/data/models), dataset + run (/data/datasets)
```
- Worker chạy pool `solo`, một job nặng mỗi lúc (một GPU một job). Trạng thái nghiệp vụ nằm ở PostgreSQL; Celery chỉ giao việc (at-least-once, task idempotent).
- Weights, checkpoint, LoRA nằm ở volume; PostgreSQL giữ metadata + checksum. Ảnh sinh ra/ảnh dataset: bytes trong `asset_blobs`.

## Yêu cầu
- Docker Desktop (Windows: bật WSL2) hoặc Docker Engine; cho GPU: NVIDIA driver + NVIDIA Container Toolkit (Windows: Docker Desktop + WSL2 GPU).
- GPU NVIDIA: **8GB VRAM là mức tối thiểu đã thử** (SDXL fp16 sinh ảnh ≤1024², train LoRA 512px, batch 1 + gradient checkpointing). Train 1024px trên 8GB **không được cam kết**.
- ~12GB đĩa cho weights (SDXL ~7GB, IP-Adapter + encoder ~3,3GB, VAE fp16-fix nhỏ), thêm ~0,7GB mỗi run train (checkpoint).
- RAM trống ≥ 6GB khi chạy worker (có ảnh tham chiếu cần thêm ~1,3GB tạm thời).

## Chạy bằng Docker Compose

```bash
cp .env.example .env              # đổi POSTGRES_PASSWORD; đổi API_PORT/WEB_PORT nếu bị trùng cổng
docker compose up -d --build      # postgres, redis, api, web (không cần GPU)
docker compose --profile gpu up -d --build   # thêm worker GPU
```
- Giao diện: http://localhost:8080 (đổi bằng `WEB_PORT`). API/Swagger: http://localhost:8000/docs (`API_PORT`).
- Lần đầu tạo ảnh: worker tải SDXL (~7GB) từ Hugging Face vào volume `models`; lần đầu dùng IP-Adapter tải thêm ~3,3GB. Muốn dùng weights đã có trên máy (hoặc để weights nằm ổ khác cho đỡ chật ổ C:): `MODEL_CACHE_HOST=D:/emvoox-data/models` trong `.env` (thư mục theo bố cục cache của Hugging Face).
- Nếu chỉ chạy không GPU: giao diện/API vẫn lên; job tạo ảnh thất bại rõ ràng (`provider_unavailable`), không giả ảnh.
- Dừng: `docker compose --profile gpu down` (giữ dữ liệu); thêm `-v` để **xóa** volume (DB, dataset, weights).
- Sao lưu: `docker compose exec postgres pg_dump -U emvoox emvoox_images > backup.sql`; sao lưu thêm volume `datasets` và `models`. Khôi phục: `psql` nạp lại file SQL vào DB trống.

## Chạy trực tiếp trên máy (phát triển)

```powershell
# 1) Postgres + Redis (Docker, cổng khác để khỏi trùng)
docker run -d --name dev-pg -e POSTGRES_USER=emvoox -e POSTGRES_PASSWORD=change-me -e POSTGRES_DB=emvoox_images -p 127.0.0.1:55432:5432 postgres:16
docker run -d --name dev-redis -p 127.0.0.1:56379:6379 redis:7
# 2) .env ở thư mục gốc (dùng 127.0.0.1, không dùng "localhost" — trên Windows có thể treo kết nối)
#    DATABASE_URL=postgresql+psycopg://emvoox:change-me@127.0.0.1:55432/emvoox_images
#    REDIS_URL=redis://127.0.0.1:56379/0
# 3) Python 3.11
py -3.11 -m venv .venv
.\.venv\Scripts\pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\pip install -r backend\requirements-gpu.txt
# 4) Ba tiến trình (mỗi cái một terminal), trong thư mục backend:
$env:PYTHONPATH="."
..\.venv\Scripts\python -m uvicorn app.main:app --port 8000
..\.venv\Scripts\python -m celery -A app.workers.celery_app worker -Q generation,training -P solo -c 1 --loglevel=info
# trong thư mục frontend: npm install; npm run dev   (đặt API_URL=http://localhost:8000 nếu API không ở 8000)
```
**Không** đặt `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` trên máy 8GB Windows: đo được chậm ~3,5× (xem [docs/IP_ADAPTER.md](docs/IP_ADAPTER.md)).

## Quy trình sử dụng
1. **Thư viện nhân vật**: tạo/sửa hồ sơ (mô tả tiếng Việt + đoạn nhận dạng tiếng Anh gửi model). Mỗi thay đổi tạo version mới.
2. **Tạo ảnh**: chọn nhân vật, preset (cảm xúc, tư thế, trang phục, bối cảnh…), xem/sửa prompt đã biên dịch, bấm tạo. Bấm "Đặt làm ảnh chuẩn" trên ảnh ưng ý.
   Bật **tham chiếu diện mạo (IP-Adapter)** để dùng ảnh chuẩn giữ khuôn mặt (scale mặc định 0,6).
3. **Lịch sử**: xem lại mọi job đã tạo (lọc theo trạng thái/nhân vật, xem ảnh, prompt, tham số), bấm “Dùng lại tham số” để điền lại form Tạo ảnh.
4. **Dataset**: tạo dataset theo nhóm (Identity/Body/Expression/Pose/Outfit/Appearance/Style/Scene), upload ảnh hoặc ZIP (`.txt` cùng tên làm caption), sửa caption, duyệt/loại, chia train/val/test, **khóa version**. Ảnh gần trùng phải cùng split.
5. **Training**: chọn dataset version (có thể gộp nhiều nhóm qua API `extra_dataset_version_ids`), base model, trigger token → Preflight → Bắt đầu. Theo dõi loss thật, hủy/resume, xem ảnh validation.
6. **Registry**: duyệt/loại LoRA. LoRA chưa duyệt chỉ dùng để đánh giá (`evaluation=true`); LoRA của nhân vật này bị chặn dùng cho nhân vật khác.
7. **Video**: duyệt ảnh ở Lịch sử (“Duyệt ảnh”) → tab Video: chọn ảnh đã duyệt, viết prompt + các đoạn chuyển động (%), tạo; xem clip, duyệt/loại (kèm lý do) hoặc sửa tự động. Cần ComfyUI chạy sẵn: [docs/VIDEO_WORKFLOW.md](docs/VIDEO_WORKFLOW.md).

Nhân vật **A01** cũng có đủ 8 dataset (46 ảnh duyệt, tạo bằng ảnh neo + IP-Adapter, xem [docs/A01_DATASET.md](docs/A01_DATASET.md)).

Tái lập thí nghiệm A02 (cần API + worker đang chạy; chạy trong `backend`, đặt `PYTHONPATH=.`):
`scripts/gen_character_dataset.py` → `scripts/curate_a02.py` → train → `scripts/eval_identity.py`. Bằng chứng: [docs/evidence](docs/evidence).

## Kiểm thử

```bash
cd backend && PYTHONPATH=. python -m pytest -q      # 55 test (video, pipeline ảnh, xuất bản C2PA, benchmark, báo cáo, auth, quyền dataset), SQLite tạm, không cần Postgres/Redis/GPU
```
Test dùng SQLite, thay hàng đợi bằng stub và (ở một test) thay SDXL bằng provider giả **chỉ để kiểm tra logic**; chúng không chứng minh pipeline AI chạy. Kiểm chứng GPU làm thủ công: `backend/scripts/smoke_gpu.py` (sinh 1 ảnh → lưu Postgres → tải lại → so SHA-256) và hướng dẫn trong [docs/SMOKE_TEST.md](docs/SMOKE_TEST.md).

## Giới hạn đã biết / chưa làm
- Chưa có đăng nhập/phân quyền, giới hạn tốc độ; chưa có Alembic (bảng tạo bằng `create_all` — đổi cột cần migration hoặc tạo lại bảng).
- Một GPU/một worker; chưa có lease/heartbeat đa worker, chưa có outbox (job đã ghi nhưng enqueue lỗi → báo lỗi 503, đánh dấu failed).
- Resume train mới thử 1 lần; validation tự động sau train từng lỗi cuDNN 1 lần (chạy lại bằng `POST /api/v1/training-runs/{id}/validate` thì đạt) — chưa tìm ra nguyên nhân.
- Chất lượng LoRA mới thử trên một nhân vật tổng hợp (25 ảnh train); IP-Adapter chưa kiểm tra toàn thân/góc nghiêng. Ảnh chuẩn **không** tự đi vào LoRA; chỉ IP-Adapter dùng chúng.
- Chưa có ControlNet, Style LoRA riêng, import Kaggle/HF, xuất 1080×1920, C2PA/watermark.
- Video: ComfyUI + model LTX-2.3 + node PromptRelay/KJNodes/VHS phải cài tay ngoài hệ thống (không nằm trong Compose); workflow đi kèm chưa kiểm chứng; Image+Audio→Video chưa hỗ trợ.
- Phát hiện gần trùng chỉ trong từng dataset version, không chéo giữa các version được gộp khi train.

## Khắc phục sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| `provider_unavailable` | Worker GPU chưa chạy / thiếu CUDA. Docker: `--profile gpu`, kiểm tra `docker compose logs worker-gpu`, `nvidia-smi` trong container. |
| `cuda_oom` khi sinh ảnh (đã sửa 17/10/2026) | Nguyên nhân từng gặp: giải mã VAE fp32 ở 1024² + LoRA trên 8GB, rồi lỗi CUDA làm job kẹt. Nay dùng VAE fp16-fix, giữ text encoder ở CPU khi VRAM thấp, tự thử lại 1 lần và tự khởi động lại worker khi context CUDA hỏng. Chạy worker bằng `powershell -File scripts/run_worker.ps1` (tự bật lại khi thoát). RAM/commit gần đầy vẫn làm chậm hoặc lỗi: đóng Chrome/Overwolf, tạm dừng container khác. |
| `cuda_oom` khi sinh/train | Hạ resolution (train 512), dùng `1:1-small` (768²), `SDXL_CPU_OFFLOAD=true` (cần RAM), đóng app dùng GPU (Chrome/Overwolf/Ollama). |
| Job chạy rất chậm (>2 phút/ảnh) | VRAM tràn sang bộ nhớ chia sẻ. Bỏ `expandable_segments`; giảm kích thước; kiểm tra tiến trình khác dùng GPU. |
| Kết nối DB treo (Windows) | Dùng `127.0.0.1` thay `localhost` trong `DATABASE_URL`/`REDIS_URL`. |
| Cổng 8000 bị chiếm | Đổi `API_PORT` trong `.env` (và `API_URL` cho Vite khi chạy dev). |
| Validation sau train báo `validation_failed` | Bấm "Chạy lại ảnh validation" (hoặc POST `/training-runs/{id}/validate`). |
| Sai `DATABASE_URL` | Trong Compose host là `postgres`; chạy trực tiếp là `127.0.0.1:<cổng>`. |
| Thiếu model | Lần đầu cần Internet tới huggingface.co; một số model cần `HF_TOKEN`. |

## Cấu trúc

```
backend/   FastAPI + Celery + SQLAlchemy; app/ml (SDXL, IP-Adapter, trainer), scripts/, tests/,
           training/vendor/ (script trainer diffusers v0.41.0 đã ghim), Dockerfile, Dockerfile.gpu
frontend/  React + TypeScript + Vite (Tạo ảnh, Thư viện, Dataset, Training), Dockerfile, nginx.conf
datasets/  registry dataset công khai (không tải tự động)
docs/      DATASETS, EVAL_A02, IP_ADAPTER, SMOKE_TEST, VIDEO_WORKFLOW, VIDEO_LTX_INTEGRATION, evidence/
compose.yaml  .env.example
```

## Kiểm tra Compose
Đã chạy thật (17/10/2026, Windows 11, Docker Desktop 29.8 + WSL2, RTX 4060 8GB):
- `docker compose --profile gpu build`: image `api` 371MB, `web` 74MB, `worker-gpu` **10,5GB** (nền `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`).
- `up -d`: 5 dịch vụ lên; `api` healthy; nginx proxy `/api` hoạt động; hồ sơ mẫu A01/A02 và registry (base, ip_adapter) được seed; worker kết nối Redis và thấy GPU (`nvidia-smi` trong container).
- `POST /training-runs/preflight`: worker trong container trả VRAM thật (trống 6,93GB/8GB), trainer + base model (đọc cache gắn ngoài) hợp lệ.
- Job sinh ảnh SDXL trong container: thành công (119 s lần đầu, gồm nạp model từ cache gắn ngoài `MODEL_CACHE_HOST` trỏ thư mục weights trên máy); ảnh nằm trong PostgreSQL (BYTEA 814.142 bytes, SHA-256 khớp).
- `down` (giữ volume) rồi `up`: ảnh vẫn tải lại được, cùng SHA-256.

**Chưa kiểm chứng trong container:** train LoRA, IP-Adapter, validation, và tải weights từ Hugging Face vào volume `models` (lần thử dùng cache có sẵn trên máy). Các phần này đã chạy thật ngoài Docker. Lưu ý vận hành: build context mỗi image là thư mục con (`backend/`, `frontend/`) — đừng dùng thư mục gốc làm context vì nó chứa `data/` (hàng chục GB); Docker Desktop từng treo khi gửi context lớn.

## Dữ liệu, license, quyền
- Dataset công khai: nhiều bộ chỉ cho nghiên cứu phi thương mại (FFHQ, CelebA, DeepFashion, AffectNet…) — xem [docs/DATASETS.md](docs/DATASETS.md); chưa tải bộ nào. Ảnh huấn luyện hiện có là ảnh AI tự sinh (nội bộ).
- SDXL 1.0: CreativeML Open RAIL++-M. IP-Adapter (`h94/IP-Adapter`): Apache-2.0 theo model card. FaceNet (`facenet-pytorch`, trọng số VGGFace2) chỉ dùng cho script đánh giá. Kiểm tra lại license trước khi phát hành thương mại.
