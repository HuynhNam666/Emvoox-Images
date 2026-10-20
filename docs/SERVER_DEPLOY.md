# Triển khai trên server (GPU NVIDIA, Docker Compose)

Tài liệu này là quy trình để bạn chạy thử trên server. **Phần đã kiểm chứng ở máy phát triển** và **phần chỉ có thể kiểm chứng trên server của bạn** được tách riêng ở mục 7. Trước hết đọc mục 7.

## 1. Yêu cầu máy chủ

| Hạng mục | Tối thiểu để chạy pilot (ảnh SDXL/Klein + video LTX) | Khuyến nghị |
|---|---|---|
| GPU NVIDIA | 24 GB VRAM (LTX-2.3 22B fp8 chạy với ComfyUI offload) | 48 GB (A6000/L40S/…): Qwen-Image không cần offload |
| RAM | 64 GB (offload sang RAM) | 128 GB |
| Đĩa NVMe trống | 120 GB (`core` ≈ 87 GB + dữ liệu) | 300 GB (thêm `zimage`, `qwen` ≈ 150 GB) |
| Driver | NVIDIA ≥ 570 cho image ComfyUI cu128 (mặc định); driver cũ: build ComfyUI với `--build-arg TORCH_INDEX=…/cu126` | |
| Phần mềm | Docker Engine ≥ 24 + Compose v2 + NVIDIA Container Toolkit (`docker run --gpus all … nvidia-smi` chạy được) | |

Số VRAM/RAM ở trên là **ước tính từ kích thước file**, chưa đo trên máy bạn. Đo thật bằng `scripts/smoke_video.py` rồi xem tab **Báo cáo** (VRAM/RAM đỉnh).
Worker build mặc định với torch **cu128** (chạy GPU Ada/Hopper và Blackwell RTX 50xx/B200). GPU/driver cũ: đặt `WORKER_BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` trong `.env`.

Model theo nhóm (`scripts/download_models.py --list`):

| Nhóm | Dung lượng | Dùng cho | Giấy phép |
|---|---|---|---|
| `video` | ≈ 42 GB | LTX-2.3 22B distilled fp8, VAE video/audio, Gemma 3 12B fp8, text projection | theo từng repo (kiểm tra trước khi thương mại) |
| `sdxl` | ≈ 12 GB | SDXL 1.0 fp16 + VAE fp16-fix + IP-Adapter plus-face + CLIP ViT-H | RAIL++-M · Apache-2.0 |
| `klein` | ≈ 32 GB | FLUX.2 Klein 4B (distilled) + Klein Base 4B (LoRA) | Apache-2.0 |
| `zimage` | ≈ 33 GB | Z-Image-Turbo | Apache-2.0 |
| `qwen` | ≈ 115 GB | Qwen-Image + Qwen-Image-Edit-2511 | Apache-2.0 |
| `face` | ≈ 0,1 GB | FaceNet VGGFace2 cho Quality Gate | MIT (facenet-pytorch); dữ liệu VGGFace2 cần xem điều khoản |

`core` = `video + sdxl + klein + face`. Model **API** (FLUX.2 Pro/Klein API, "FLUX 3 Image", Gemini Nano Banana, Seedream, GPT Image) không có trọng số: chỉ cần khóa trong `.env`. **Midjourney** không có API chính thức: tạo ngoài rồi nhập ảnh (tab Ảnh theo shot → API `/visual/jobs/{id}/manual-import`).

## 2. Cài đặt (lần đầu)

```bash
git clone <repo> emvoox && cd emvoox          # hoặc chép thư mục dự án lên server
cp .env.server.example .env
nano .env                                     # đổi POSTGRES_PASSWORD, API_TOKEN; điền khóa API muốn dùng; đặt GPU_COST_PER_HOUR nếu muốn quy ra tiền
docker compose --profile gpu build            # api, web, worker-gpu, comfyui (lần đầu 15–30 phút)
docker compose --profile init run --rm models-init     # tải model nhóm MODEL_GROUPS (mặc định core ≈ 87 GB); chạy lại được, tải tiếp khi bị ngắt
docker compose --profile gpu up -d
docker compose exec api python scripts/server_selfcheck.py
```

`models-init` ghi `MODELS.lock.json` (repo, revision, kích thước, sha256 theo Hugging Face) vào volume model để truy vết phiên bản.
Thêm nhóm sau này: `MODEL_GROUPS=zimage,qwen docker compose --profile init run --rm models-init`.
Model nằm ổ khác: đặt `MODEL_CACHE_HOST=/mnt/nvme/emvoox-models` trong `.env` (cả worker, ComfyUI và models-init dùng chung thư mục này).

### HTTPS và truy cập từ ngoài
- Chỉ trong mạng nội bộ/VPN: để mặc định (`WEB_BIND=127.0.0.1`) rồi dùng SSH tunnel: `ssh -L 8080:127.0.0.1:8080 server`.
- Công khai: trỏ DNS về server, đặt `DOMAIN=studio.example.com`, `COOKIE_SECURE=true`, rồi `docker compose --profile gpu --profile tls up -d` (Caddy tự cấp chứng chỉ HTTPS). **Bắt buộc đặt `API_TOKEN`** — không có thì ai vào được cổng web đều dùng được hệ thống và tốn tiền API của bạn.
- ComfyUI **không có xác thực** nên compose không publish cổng của nó; đừng mở cổng 8188 ra ngoài.
- Đăng nhập hiện là **một token dùng chung** (chưa có người dùng/phân quyền riêng). Tên người duyệt do người dùng tự nhập và được lưu vào lịch sử duyệt.

## 3. Kiểm nghiệm trên server (theo thứ tự)

1. `docker compose exec api python scripts/server_selfcheck.py` — xem mỗi dòng ✓/!/✗ và cách sửa. Cần 0 lỗi (✗) trước khi tiếp tục.
2. **Video thật, một lệnh** (cần ảnh nhân vật thật 9:16; không có `--image` sẽ dùng ảnh dựng sẵn chỉ để kiểm đường ống):
   ```bash
   docker compose exec api python scripts/smoke_video.py --api http://localhost:8000 --token "$API_TOKEN" --image /data/exports/a01.png --frames 49 --out /data/exports/smoke.mp4
   ```
   Script in thời gian, VRAM đỉnh, RAM đỉnh. Xem clip; nếu đạt, mở `backend/app/ml/workflows/ltx23_promptrelay.api.json` và đổi `"status": "schema_validated"` thành `"verified"` (rồi `docker compose restart api worker-gpu`).
3. Mở giao diện → tab **Model**: kiểm danh sách khả dụng (GPU của worker, các API đã có khóa).
4. Tab **Ảnh theo shot**: tạo một shot A01 → xem Model Router chọn tuyến, QC, duyệt, **Xuất bản** (kiểm nhãn AI + C2PA).
5. Tab **Video**: chọn ảnh vừa duyệt → tạo clip → duyệt → Xuất bản.
6. Tab **Báo cáo**: kiểm chi phí/thời gian/VRAM/RAM; tải `.md` và `.csv`.

## 4. Mở rộng thư viện nhân vật và dataset

```bash
# 30 ảnh × 8 nhóm = 240 ảnh/nhân vật qua pipeline mới (Model Router + Quality Gate), vào dataset ở trạng thái CHỜ DUYỆT
docker compose exec api python scripts/expand_library.py --api http://localhost:8000 --token "$API_TOKEN" --code A01 --per-group 30 --budget 0.5
```
Điều kiện: nhân vật phải có ảnh chuẩn (nếu không Quality Gate không đo được nhận dạng). Duyệt ở tab Dataset rồi khóa version trước khi train.
`COMMERCIAL_USE=true` chặn train bằng dataset phi thương mại hoặc chưa ghi license; ảnh AI tự sinh được xem là hợp lệ.
Dataset công khai khác: `docs/DATASETS.md`, `datasets/registry.json` (mỗi dòng có `commercial` và `training_use` — nhiều bộ ảnh quá nhỏ để train).

## 5. Vận hành

- **Một GPU, một job nặng mỗi lúc.** Worker nhả model của mình trước khi video chạy và gọi `/free` của ComfyUI trước khi nạp model ảnh, nên ảnh và video không tranh VRAM; đổi lại mỗi lần đổi loại việc phải nạp lại model (chậm vài chục giây–vài phút). Muốn song song: dùng hai GPU và ghim `NVIDIA_VISIBLE_DEVICES` riêng cho từng container.
- **Sao lưu:** `docker compose exec postgres pg_dump -U emvoox emvoox_images > backup.sql`; sao lưu thêm volume `datasets`, `exports`; model tải lại được (`MODELS.lock.json`).
- **Cập nhật:** `git pull && docker compose --profile gpu build && docker compose --profile gpu up -d`. Cột mới được thêm tự động khi API khởi động (`app/db/migrate.py`); chưa có Alembic nên đổi kiểu cột cần xử lý thủ công.
- **ComfyUI/node:** commit đã ghim trong `deploy/comfyui/Dockerfile` (đúng bộ đã kiểm schema). Đổi commit ⇒ chạy lại `python scripts/validate_comfy_workflow.py --url http://comfyui:8188` trong container api.
- **Log:** `docker compose logs -f worker-gpu comfyui api`.
- **Chi phí GPU:** đặt `GPU_COST_PER_HOUR` (thuê hoặc khấu hao) để báo cáo quy ra tiền; bỏ trống thì báo cáo chỉ hiện giờ GPU.

## 6. Sự cố thường gặp

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| selfcheck: Worker GPU "không thấy CUDA" | Thiếu NVIDIA Container Toolkit hoặc driver. Thử `docker compose exec worker-gpu nvidia-smi`. |
| Video lỗi `comfy_unreachable` | ComfyUI chưa lên (nạp lâu) hoặc `COMFYUI_URL` sai: `docker compose logs comfyui`. |
| Video lỗi `comfy_execution_error` + CUDA OOM | Thêm `COMFY_ARGS=--lowvram`, giảm `frame_count`/độ phân giải, đảm bảo không có job ảnh đang giữ VRAM. |
| `comfy_timeout` | Tăng `COMFYUI_TIMEOUT_S`. Prompt vẫn chạy ở ComfyUI; attempt giữ `prompt_id` để đối chiếu. |
| Ảnh luôn "QC: cần xem" | FaceNet không nạp được hoặc nhân vật thiếu ảnh chuẩn có mặt rõ (xem Tab Model và selfcheck). |
| Model API bị loại khỏi Router | Thiếu khóa, hoặc (FLUX 3 Image) chưa đặt `BFL_FLUX3_ENDPOINT`. |
| `cuda_oom` khi tạo ảnh với Qwen/Klein | GPU nhỏ hơn mức `min_vram_gb` của model: Router tự dùng offload CPU (chậm) hoặc loại model; xem cột "khả dụng". |

## 7. Điều gì đã kiểm chứng, điều gì chưa

**Đã kiểm chứng (máy phát triển Windows, không có GPU đủ lớn):**
- 53+ test pytest: Model Router, Prompt Compiler, Quality Gate (với FaceNet giả), vòng sửa tối đa 2 lượt, failover, chống tính phí trùng, ngân sách, duyệt, benchmark chấm mù, báo cáo, đăng nhập, rà quyền dataset.
- **Xuất bản thật:** nhãn AI, đổi khung 1080×1920/1920×1080 không kéo méo, ký + đọc lại xác minh C2PA (ảnh PNG và video MP4 qua ffmpeg thật; chứng thư thử nghiệm ⇒ trạng thái *Valid*, chưa *Trusted*).
- **Workflow video khớp ComfyUI thật:** ComfyUI 0.42.0 + PromptRelay + KJNodes + VideoHelperSuite được dựng lên, `/object_info` đối chiếu từng node/input, `/prompt` chấp nhận toàn bộ graph (chỉ báo thiếu file model). Schema đó được lưu thành fixture test (`backend/tests/fixtures/comfy_object_info.json`).
- Tải model: `download_models.py` đã tải thử một file thật và khớp sha256 của Hugging Face.
- Giao diện biên dịch (`tsc`, `vite build`); `docker compose config` hợp lệ.

**Chưa kiểm chứng — chỉ chạy được trên server của bạn:**
1. **Chưa có clip LTX-2.3 nào được render** (không có GPU/model ở máy phát triển). Workflow ở trạng thái `schema_validated`, chưa `verified`.
2. **Chưa build image Docker** (daemon Docker ở máy phát triển không chạy). `deploy/comfyui/Dockerfile`, `Dockerfile.gpu` và `compose.yaml` chưa được build/chạy; lỗi phiên bản torch/CUDA nếu có sẽ lộ ở bước `build` hoặc `selfcheck`.
3. **Các model tự chạy** (Klein, Qwen-Image, Z-Image) mới được viết theo chữ ký pipeline diffusers 0.41 (đã kiểm chữ ký, chưa chạy thật); SDXL là đường đã chạy thật trước đó.
4. **Các bộ kết nối API** (BFL, Gemini, Seedream, OpenAI) viết theo tài liệu công khai, chưa gọi với khóa thật. Tên "FLUX 3 Image" và "Nano Banana 2.1" trong báo cáo chưa xác minh được endpoint/id công khai ⇒ cấu hình bằng biến môi trường.
5. **Ngưỡng nhận dạng** (`QC_IDENTITY_PASS=0.60`, `QC_IDENTITY_FAIL=0.45`) là ngưỡng tham khảo của FaceNet, chưa hiệu chuẩn cho A01/A02.
6. **Số liệu thời gian/VRAM/RAM/chi phí thực tế** chỉ có sau khi chạy trên server. Mọi con số trong báo cáo đều lấy từ lượt chạy thật; chưa chạy thì "—".
7. **Chứng thư C2PA tin cậy** cho phát hành thật phải do bạn mua/cấp (script `make_dev_c2pa_cert.py` chỉ tạo chứng thư thử nghiệm). Watermark ẩn **không** được tự nhúng (cần bộ mã hóa + bộ phát hiện); chỉ ghi nhận dấu của nhà cung cấp nếu có (vd SynthID của Gemini).
8. **Pháp lý:** căn cứ Điều 11 Luật AI 134/2025/QH15 và Điều 17–18 NĐ 142/2026/NĐ-CP được lấy từ báo cáo, **không xác minh lại**; kích thước chữ/lề nhãn là thông số nội bộ đề xuất, không phải quy định.
