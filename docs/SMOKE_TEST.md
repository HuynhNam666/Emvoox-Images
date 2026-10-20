# Smoke test SDXL thật (RTX 4060 8GB, RAM 16GB)

Đã chuẩn bị sẵn: `.venv` (Python 3.11 + torch cu124), weights SDXL trong `MODEL_CACHE_DIR` (6,7GB; hiện ở `D:\emvoox-data\models`), Postgres riêng `ecs-smoke-pg` (cổng 55432), `.env` trỏ DB này.
Chưa từng chạy thành công: lần thử trước thrash vì RAM trống ~0,2GB.

## 1. Giải phóng RAM (không tắt container nào)
- Mở Task Manager → cột Memory. Cần **≥ 9 GB RAM trống** trước khi chạy (SDXL fp16 ≈ 7GB trong RAM khi dùng CPU offload).
- Đóng trình duyệt nhiều tab, Overwolf (đang giữ GPU), game/app overlay.
- `vmmem`/`VmmemWSL` thường chiếm nhiều nhất (Docker). Giới hạn mà không tắt container: tạo `%UserProfile%\.wslconfig`:
  ```
  [wsl2]
  memory=6GB
  swap=8GB
  ```
  rồi `wsl --shutdown` — **lệnh này khởi động lại Docker Desktop và dừng mọi container**, nên chỉ làm khi bạn sẵn sàng. Bỏ qua nếu không muốn.

## 2. Chạy (PowerShell, tại thư mục dự án)
```powershell
cd C:\Users\User\Downloads\emvoox\backend
$env:PYTHONPATH="."; $env:PYTHONUTF8="1"
$env:SDXL_CPU_OFFLOAD="true"
# MODEL_CACHE_DIR đọc từ .env (D:/emvoox-data/models); chỉ đặt tay nếu muốn ghi đè
..\.venv\Scripts\python.exe -W ignore scripts\smoke_gpu.py --steps 15 --aspect 1:1
```
Kết quả đạt: dòng `OK asset <id> ... bytes, sha256 ...` và file `backend\smoke_output.png`.

## 3. Giảm tải nếu vẫn chậm/OOM
| Triệu chứng | Cách chỉnh |
|---|---|
| CUDA out of memory | giữ `SDXL_CPU_OFFLOAD=true`; dùng `--aspect 1:1` (1024²) thay vì 9:16; hạ `--steps 12` |
| RAM đầy, ổ đĩa chạy liên tục | thêm RAM trống (mục 1); không chạy cùng lúc worker GPU của stack khác |
| Quá chậm | đây là chi phí của offload; chỉ cần chứng minh plumbing, không đo tốc độ |

## 4. Xác minh ảnh nằm trong PostgreSQL
```powershell
docker exec ecs-smoke-pg psql -U emvoox -d emvoox_images -c "select a.id, a.byte_size, a.sha256, length(b.data) as blob_bytes from assets a join asset_blobs b on b.asset_id=a.id"
```
`blob_bytes` phải bằng `byte_size`.

## 5. Dọn dẹp (tùy chọn, chỉ container/volume của test này)
```powershell
docker rm -f ecs-smoke-pg; docker volume rm ecs_smoke_pgdata
```
