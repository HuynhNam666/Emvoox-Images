# Trạng thái triển khai (2026-10-16)

Môi trường đã dùng để kiểm chứng: Windows 11, Python 3.14 (venv), Node 24, Docker Desktop, PostgreSQL 16 trong container, RTX 4060 Laptop 8 GB. **Không có lần chạy GPU nào được thực hiện.**

| Hạng mục | Implemented | Tested | Ghi chú |
|---|---|---|---|
| Schema PostgreSQL + Alembic 0001 | ✓ | ✓ (tạo qua `alembic upgrade head` trong test) | Revision đầu dùng `create_all`; thay đổi sau phải là migration tường minh |
| Lưu ảnh BYTEA, `/content`, `/thumbnail`, list không tải bytes | ✓ | ✓ | SHA-256 khớp, bytes đọc lại từ `asset_blobs` |
| Generation job: idempotency, 202, state machine có điều kiện, cancel, review | ✓ | ✓ (mock provider) | Redelivery không tạo output trùng |
| Outbox/sweeper, heartbeat, GPU lease Redis | ✓ | ✗ | Lease chỉ test với fakeredis; sweeper chưa có test |
| Provider SDXL (Diffusers) + LoRA load/unload | ✓ | ✗ **blocked** | Cần worker CUDA; chưa chạy |
| Capability gate LoRA (family/base/approval) | ✓ | ✓ | |
| Prompt compiler template | ✓ | ✓ | Không dịch tiếng Việt tự do; cảnh báo thay vì bịa |
| Export 1080×1920 (resize + crop, giữ tỉ lệ) | ✓ | ✓ | Không phải AI upscale |
| Dataset: upload, ZIP an toàn, SHA-256 trùng, pHash gần trùng, caption, duyệt | ✓ | ✓ | |
| Dataset version bất biến + split theo nhóm gần trùng | ✓ | ✓ | |
| Training adapter, preflight, materialize, parser loss | ✓ | một phần (argv + parser) | Chưa chạy trainer thật; preflight kiểm cờ qua `--help` của script ghim |
| Validation samples sau training bằng LoRA vừa tạo | ✓ | ✗ **blocked** | |
| Review LoRA → approved mới dùng được ở sản xuất | ✓ | ✓ (phần chặn LoRA chưa duyệt) | |
| Resume training | ✗ (501) | ✗ | Chưa kiểm chứng checkpoint resume của script |
| Frontend (6 trang) | ✓ | typecheck + build ✓; E2E trình duyệt ✗ | Chưa chạy UI với API thật |
| Docker Compose + Dockerfile (api, beat, web, worker-gpu, worker-mock) | ✓ | ✗ | Chưa `docker compose build` |
| Authentication / quyền theo project | ✗ | ✗ | Chỉ dùng localhost; không đưa ra Internet |
| IP-Adapter / ControlNet | ✗ | ✗ | Cố ý để sau tuyến chính |
| Hồ sơ mẫu A01/A02 | ✗ | – | Chưa seed; tạo qua UI |

## Việc người dùng cần làm để đóng các mục blocked

1. Máy có NVIDIA Container Toolkit (WSL2/Linux): `docker compose --profile gpu up -d --build worker-gpu`.
2. Smoke test sinh ảnh: tạo 1 job 1:1, 20 steps từ UI; xác nhận `assets` + `asset_blobs.length(data)` > 0.
3. Smoke test LoRA: dataset ~20 ảnh có quyền sử dụng, preset 8 GB, ghi đè `max_train_steps=20`; kiểm tra có `pytorch_lora_weights.safetensors`, validation samples sinh được, LoRA nạp lại.
4. Nếu OOM ở 8 GB với 512px, giảm `resolution` hoặc dùng máy nhiều VRAM hơn. Chưa có số đo thực để cam kết.
5. Chạy quality pilot A01/A02 và ghi tỉ lệ đạt thủ công; chưa có số liệu nào.
