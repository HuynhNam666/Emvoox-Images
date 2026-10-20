# Luồng hoạt động và các giá trị trong hệ thống

Tài liệu mô tả đúng như mã nguồn hiện tại (backend/app, frontend/src). Giá trị mặc định lấy từ code; con số hiệu năng lấy từ lần đo trên RTX 4060 8GB.

## 1. Thành phần

```
Trình duyệt ──► web / Vite ──/api──► API (FastAPI) ──► PostgreSQL   (trạng thái nghiệp vụ + ảnh BYTEA)
                                        │
                                        └──► Redis (hàng đợi Celery) ──► Worker GPU ──► SDXL / LoRA / IP-Adapter / trainer
                                                                              └── ổ đĩa: weights (MODEL_CACHE_DIR), dataset + run train (DATASET_DIR)
```
- **PostgreSQL** là nguồn sự thật: nhân vật, job, dataset, ảnh, registry. Redis chỉ giao việc.
- **Worker** chạy pool `solo`: một job nặng mỗi lúc (một GPU một job). Job khác xếp hàng.
- Weights, checkpoint, LoRA nằm ổ đĩa; DB chỉ giữ đường dẫn + checksum.

## 2. Luồng tạo ảnh

1. **Form (UI)**: chọn nhân vật, preset, tham số. *Prompt compiler* (chạy ở trình duyệt) ghép chuỗi tiếng Anh theo thứ tự: `nhận dạng nhân vật` → góc máy → cảm xúc → tư thế → trang phục → bối cảnh → ánh sáng → phong cách → ghi chú thêm. Mô tả tiếng Việt chỉ lưu làm bản gốc (`prompt_original`), không dịch.
2. **`POST /api/v1/generation-jobs`** (kèm header `Idempotency-Key`): API kiểm tra (LoRA hợp lệ/đã duyệt, đúng nhân vật, ảnh chuẩn có đủ...), ghi job `queued` vào DB, rồi gửi task `generate` vào queue `generation`. Trả HTTP 202 + `job_id`. Cùng key + cùng payload → trả lại job cũ; cùng key khác payload → 409.
3. **Worker** nhận task: đổi trạng thái có điều kiện `queued → running` (task giao lại sẽ bị bỏ qua).
4. **Provider SDXL**: nạp pipeline (một lần, giữ trong VRAM) → (nếu có) nạp LoRA → (nếu có tham chiếu) tính embedding ảnh chuẩn bằng CLIP ViT-H, nạp IP-Adapter → khử nhiễu `steps` bước → PNG. LoRA/IP-Adapter được gỡ sau mỗi job để không rò sang job khác.
5. **Lưu**: mỗi ảnh = một dòng `assets` (metadata + sha256 + provenance) + một dòng `asset_blobs` (bytes PNG, cột BYTEA) + `generation_outputs` (thứ tự, seed thật). Chỉ sau khi ghi xong mới đặt job `review_required`.
6. **Giao diện** hỏi `GET /generation-jobs/{id}` mỗi 2 giây, hiển thị ảnh qua `/assets/{id}/content`.
7. **Lịch sử**: `GET /api/v1/generation-jobs?limit&offset&state&character_id` trả job mới nhất trước kèm ảnh đầu ra, nhân vật và toàn bộ tham số; giao diện tab Lịch sử dùng nó (tự làm mới 4 giây khi còn job đang chạy) và có nút “Dùng lại tham số”.
8. **Đặt làm ảnh chuẩn**: tạo *character version* mới có thêm ảnh đó (job cũ vẫn trỏ về version đã dùng).

## 3. Luồng dataset

1. **Tạo dataset**: tên, nhóm (1 trong 8), nhân vật (tùy chọn), nguồn, license.
2. **Upload** ảnh JPG/PNG/WebP hoặc ZIP (ảnh + file `.txt` cùng tên = caption). Kiểm tra: đọc ảnh thật (không tin đuôi file), giới hạn dung lượng/pixel, ZIP chống `../`, symlink, bom giải nén. Báo cáo import trả về: nhập, trùng, gần trùng, độ phân giải thấp, bị từ chối.
3. **Phát hiện trùng**: sha256 (trùng hoàn toàn → bỏ), dHash 64-bit khoảng cách ≤ 4 (gần trùng → gắn cờ `near_duplicate:<id>`), cạnh ngắn < 512px → tự loại (`low_resolution`).
4. **Duyệt**: mỗi ảnh có `status` (pending/approved/rejected), `split` (train/val/test), caption, cờ `synthetic` (ảnh AI). Thao tác hàng loạt: duyệt/loại, đổi split, thêm trigger vào đầu caption.
5. **Khóa version**: chỉ lấy ảnh *approved*; ảnh gần trùng phải cùng split (hoặc dùng "gộp cùng split"); tính `manifest_hash` (sha256 của danh sách sha256+caption+split). Version **bất biến**: sửa ảnh/caption sau đó không đổi version cũ, chỉ tạo version mới. Xuất `manifest.jsonl`.

## 4. Luồng train LoRA

1. **Preflight** (`/training-runs/preflight`): API hỏi *worker* (task `gpu_info`) để lấy VRAM thật; kiểm tra dataset, base model, số bước, trainer, đĩa, job đang chạy. Kết quả ok/warn/fail; fail thì không cho bắt đầu.
2. **Tạo run** (`POST /training-runs`): trạng thái `queued`, lưu cấu hình đã resolve.
3. **Worker** `preparing`: giải phóng SDXL khỏi VRAM → xuất ảnh *train* + `metadata.jsonl` từ PostgreSQL ra đĩa (caption được đảm bảo chứa trigger) → ảnh val dùng để lấy prompt validation, ảnh test không dùng khi train.
4. **`running`**: chạy script DreamBooth-LoRA SDXL (diffusers v0.41.0, ghim) bằng tiến trình con, tham số dựng từ cấu hình đã validate (không qua shell). Worker đọc tiến độ thật → bảng `training_metrics` (bước, loss, lr), heartbeat; kiểm tra cờ hủy mỗi vài giây (hủy = kill cả cây tiến trình). Checkpoint ghi mỗi `checkpointing_steps`; có thể **resume** từ checkpoint mới nhất.
5. **`validating`**: sao chép `pytorch_lora_weights.safetensors` ra `MODEL_CACHE_DIR/loras/<id>.safetensors`, tính sha256, tạo bản ghi registry `awaiting_review`; nạp lại chính LoRA để sinh ảnh validation.
6. **`awaiting_review`** → người duyệt `approved`/`rejected`. LoRA chưa duyệt chỉ dùng được ở chế độ đánh giá.

## 5. Các giá trị trạng thái

| Đối tượng | Giá trị | Ý nghĩa |
|---|---|---|
| Job tạo ảnh | `queued` | đã ghi DB, chờ worker |
| | `running` | worker đang tạo |
| | `review_required` | ảnh đã lưu xong, chờ người duyệt (**không** có nghĩa "đã duyệt") |
| | `approved` / `rejected` | quyết định của người |
| | `failed` / `cancelled` | lỗi (`error` ghi rõ: `cuda_oom`, `provider_unavailable`, `generation_error`...) / đã hủy |
| Run train | `queued` → `preparing` → `running` → `validating` → `awaiting_review` → `approved`/`rejected`; `failed`, `cancelled` | như trên |
| Ảnh dataset | `pending` / `approved` / `rejected` | chờ duyệt / dùng được / loại |
| Split | `train` / `val` / `test` | học / chọn checkpoint / đánh giá cuối |
| Cờ ảnh | `low_resolution`, `near_duplicate:<id>` | cạnh ngắn dưới ngưỡng / gần trùng ảnh đã có |
| Preflight | `ok` / `warn` / `fail` | đạt / cảnh báo (vẫn chạy được) / chặn |
| Registry model | `verified`, `experimental`, `unavailable` | base/adapter: đã kiểm chứng / thử nghiệm / không dùng được |
| | `awaiting_review`, `approved`, `rejected` | LoRA: chờ duyệt / dùng sản xuất / loại |
| Loại model | `base`, `character_lora`, `style_lora`, `ip_adapter` | nền SDXL / LoRA một nhân vật / LoRA phong cách / adapter tham chiếu |

## 6. Tham số tạo ảnh

| Tham số | Phạm vi (mặc định) | Ý nghĩa |
|---|---|---|
| `aspect_ratio` | `9:16` (768×1344), `1:1` (1024²), `16:9` (1344×768), `1:1-small` (768²) | khung hình gốc của SDXL; `1:1-small` đỡ VRAM nhất |
| `num_images` | 1–4 (1) | số ảnh mỗi job; ảnh thứ i dùng seed `seed+i` |
| `seed` | 0 … 2³¹−1 (42) | số khởi tạo nhiễu; cùng seed + cùng tham số cho kết quả gần giống để truy vết (không đảm bảo từng bit giữa máy khác) |
| `steps` | 1–100 (28) | số bước khử nhiễu; nhiều bước = chi tiết hơn nhưng chậm tỉ lệ thuận |
| `guidance_scale` | 0–20 (6) | độ bám prompt (CFG); cao quá ảnh gắt/cứng, thấp quá lệch prompt |
| `negative_prompt` | chuỗi | những thứ **không** muốn (mờ, tay thừa, watermark...) |
| `prompt_compiled` | chuỗi | prompt thật gửi model (có thể sửa tay); `prompt_original` là mô tả gốc tiếng Việt |
| `lora_version_id` | id | LoRA dùng cho job; trigger token tự thêm vào đầu prompt nếu thiếu |
| `lora_scale` | 0–2 (1) | cường độ LoRA (0 = tắt, >1 = mạnh hơn mức đã train) |
| `evaluation` | true/false | cho phép dùng LoRA `awaiting_review` để thử; sản xuất chỉ dùng LoRA `approved` |
| `use_references` | true/false | bật IP-Adapter dùng ảnh chuẩn của nhân vật làm tham chiếu diện mạo |
| `reference_asset_ids` | ≤ 4 ảnh | ghi đè danh sách ảnh tham chiếu (mặc định: ảnh chuẩn của version) |
| `reference_scale` | 0–1 (0,6) | độ mạnh tham chiếu; cao → giống ảnh chuẩn hơn nhưng kéo theo áo/khung hình, ít nghe prompt |
| `Idempotency-Key` (header) | chuỗi | chống tạo trùng job khi gửi lại cùng yêu cầu |

Hiệu năng đã đo (768², 25 bước): ~30 s/ảnh không IP-Adapter, ~49 s có IP-Adapter (nóng); 9:16 + IP-Adapter ~28 s/bước khi VRAM/RAM đầy.

## 7. Tham số train

| Tham số | Phạm vi (mặc định) | Ý nghĩa |
|---|---|---|
| `objective` | `character` / `style` | học một nhân vật / học phong cách-khái niệm chung |
| `trigger_token` | 3–64 ký tự `[A-Za-z0-9_]` | từ kích hoạt gắn vào caption (vd `emv_a02`); dùng lại khi tạo ảnh |
| `resolution` | 512 / 768 / 1024 (512) | cạnh ảnh khi train; cao hơn = nặng VRAM hơn rất nhiều (1024 trên 8GB không cam kết) |
| `rank` | 4–64 (16) | "dung lượng" của LoRA; lớn = học được nhiều hơn nhưng file to, dễ overfit |
| `learning_rate` | 1e-6 … 1e-3 (1e-4) | bước cập nhật; quá cao → hỏng/nhiễu, quá thấp → học chậm |
| `max_train_steps` **hoặc** `num_train_epochs` | 1–5000 / 1–200 | độ dài train (chỉ một trong hai). Số bước tối ưu = `ceil(ceil(N/batch)/accum) × epochs` |
| `train_batch_size` | 1–4 (1) | số ảnh mỗi lần đưa vào GPU |
| `gradient_accumulation_steps` | 1–16 (4) | gộp nhiều lượt trước khi cập nhật; batch hiệu dụng = batch × accum |
| `checkpointing_steps` | 5–2000 (100) | lưu checkpoint mỗi N bước (để resume); chỉ giữ 2 bản gần nhất |
| `mixed_precision` | `fp16` / `bf16` / `no` (fp16) | độ chính xác số học; fp16 tiết kiệm VRAM |
| `gradient_checkpointing` | true (true) | đổi thời gian lấy VRAM (bắt buộc thực tế trên 8GB) |
| `use_vae_fp16_fix` | true (true) | dùng VAE `madebyollin/sdxl-vae-fp16-fix` để tránh lỗi số học khi fp16 |
| `seed` | 42 | số ngẫu nhiên của lần train |
| `extra_dataset_version_ids` | ≤ 16 | gộp thêm dataset version (vd 8 nhóm của một nhân vật) vào một lần train |

Cố định bởi hệ thống: scheduler `constant`, warmup 0, `center_crop`, `dataloader_num_workers=0`, báo cáo qua tensorboard, caption dùng cột `prompt`.

**Loss** là sai số khử nhiễu trên từng bước, **rất nhiễu** (dao động 0,00x–0,3 dù train tốt). Không dùng loss để kết luận chất lượng nhận dạng: phải xem ảnh validation và đo trên cảnh chưa học.

**Ngưỡng preflight**: VRAM trống tối thiểu gợi ý (heuristic, chưa benchmark): 512px → 6,5 GB, 768px → 8,5 GB, 1024px → 12 GB; đĩa trống ≥ 15 GB; cảnh báo khi < 10 ảnh train; cảnh báo khi dataset thiếu license; thiếu val/test hoặc < 30 ảnh được ghi ở cảnh báo của version.

## 8. 8 nhóm dataset

`identity` khuôn mặt (chính diện, nghiêng, ¾, ngẩng/cúi) · `body` toàn thân/bán thân nhiều mặt · `expression` biểu cảm · `pose` tư thế · `outfit` trang phục · `appearance` tóc/phụ kiện · `style` phong cách (anime, manhua, ảnh thật...) · `scene` bối cảnh. Mỗi dataset thuộc đúng một nhóm; một lần train có thể gộp nhiều dataset version.

## 9. Cấu hình hệ thống (biến môi trường)

| Biến | Mặc định / ví dụ | Ý nghĩa |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg://…@127.0.0.1:55432/emvoox_images` | kết nối PostgreSQL (Compose: host `postgres`) |
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | hàng đợi Celery |
| `ASSET_STORAGE_BACKEND` | `postgres` | nơi lưu bytes ảnh (hiện chỉ `postgres` được dùng) |
| `MODEL_CACHE_DIR` | `./data/models` (đang đặt `D:/emvoox-data/models`) | weights SDXL/IP-Adapter, thư mục `loras/` |
| `DATASET_DIR` | `./data/datasets` (đang `D:/emvoox-data/datasets`) | dữ liệu xuất để train + checkpoint (`runs/<id>`) |
| `SDXL_MODEL_ID` | `stabilityai/stable-diffusion-xl-base-1.0` | base model |
| `SDXL_CPU_OFFLOAD` | `false` | `true` = đẩy bớt model sang CPU (đỡ VRAM, chậm, cần ~8GB RAM trống) |
| `max_upload_bytes` | 20 MB | tối đa mỗi file ảnh |
| `max_zip_entries` / `max_zip_uncompressed_bytes` | 500 / 500 MB | giới hạn ZIP import |
| `max_image_pixels` | 50 triệu | chống ảnh khổng lồ |
| `min_image_side` | 512 px | dưới ngưỡng này ảnh bị tự loại (`MIN_IMAGE_SIDE` để đổi) |
| `near_dup_hamming` | 4 | khoảng cách dHash coi là gần trùng |
| Compose: `API_PORT`, `WEB_PORT`, `MODEL_CACHE_HOST`, `HF_TOKEN` | 8000, 8080, volume `models`, trống | cổng, thư mục weights máy chủ, token Hugging Face |

## 10. Đánh giá nhận dạng (script `eval_identity.py`)

Độ giống = **cosine** giữa embedding khuôn mặt FaceNet của ảnh sinh ra và vector trung bình ảnh tham chiếu (1 = giống hệt, ~0 = không liên quan). Mốc đối chiếu: *leave-one-out* của chính ảnh tham chiếu (median 0,94; p10 0,72). "≥ p10" = tỉ lệ ảnh đạt ít nhất mức giống của ảnh yếu nhất trong nhóm tham chiếu. Đây là chỉ số hỗ trợ, không phải ngưỡng "cùng một người".
Các nhánh: A không LoRA + mô tả chữ · B LoRA chỉ trigger · C "a man" (đối chứng) · D LoRA + chữ · E IP-Adapter + chữ · F IP-Adapter + "a man" · G LoRA + IP + chữ (kết quả: `docs/IP_ADAPTER.md`, `docs/EVAL_A02.md`).

## 6. Luồng tạo video (ảnh → clip)

1. **Điều kiện**: ảnh nguồn đã duyệt — đầu ra của job ảnh `approved` (`POST /generation-jobs/{id}/review`) hoặc ảnh chuẩn nhân vật. Chưa duyệt → 409.
2. **`POST /api/v1/video/jobs`** (+ `Idempotency-Key`): kiểm tra khung (chia hết 32), `frame_count` = 8n+1, các đoạn chuyển động liền nhau phủ 0→100 (%), tải workflow ghim; ghi `video_jobs` + `video_attempts #1` rồi gửi task `generate_video` vào queue `video`. HTTP 202.
3. **Worker** (`app/workers/tasks.py::generate_video`): nhận attempt `queued` (hoặc nối lại attempt `running` đã có `prompt_id`), nhả VRAM SDXL, kiểm tra node ComfyUI (`/object_info`), upload ảnh, dựng prompt API từ mapping, `/prompt`, **ghi `prompt_id`**, chờ `/history/{id}`, tải mp4 (`/view`), lưu `assets(kind=generated_video)` + `asset_blobs`, gọi `/free`. Ghi thời gian + chi phí (kể cả khi lỗi).
4. **`review_required`** → người xem cả chuyển động và âm thanh: duyệt (`approved`) hoặc loại kèm lý do. `keyframe_identity` → `returned_to_image`; lý do khác → `rejected` rồi `POST /video/jobs/{id}/retry` (attempt `repair`). Tối đa `VIDEO_MAX_REPAIRS`=2 lượt sửa hoặc hết `budget_limit` → `needs_human`.
5. **Số liệu**: `GET /api/v1/video/metrics` (tỷ lệ đạt, p50/p95, chi phí / clip đạt, lý do loại).

| Đối tượng | Giá trị | Ý nghĩa |
|---|---|---|
| Job video | `queued` / `running` | chờ / đang render |
| | `review_required` | đã render, chờ người duyệt (không phải đã duyệt) |
| | `approved` / `rejected` | người duyệt chấp nhận / loại (còn lượt sửa) |
| | `returned_to_image` | lỗi ở ảnh nguồn, quay về công đoạn ảnh |
| | `failed` | lỗi thực thi (`comfy_unreachable`, `comfy_nodes_missing`, `comfy_execution_error`, `comfy_timeout`…) |
| | `needs_human` | hết lượt sửa hoặc ngân sách |
| Attempt | `queued` → `running` → `rendered` / `failed` | một lần render; `kind` = `initial` / `repair` |

## 7. Luồng pipeline ảnh theo shot (báo cáo EMVOOX1)

1. **`POST /api/v1/visual/jobs`** (+ `Idempotency-Key`): nhận project/tập/shot, danh sách `character_version_id`, `scene_code`, `scene_spec` (hành động, cảm xúc, cỡ cảnh, góc, ánh sáng, phong cách), khung hình, số phương án, trần chi phí, tùy chọn ghim model.
2. **Resolve + ghim phiên bản:** nhân vật (mô tả `identity_en` + ảnh chuẩn), trang phục, bối cảnh (version), tham chiếu bổ sung. Mã A01/B02 chỉ là mã quản trị — model chỉ nhận mô tả + ảnh thật.
3. **Model Router:** loại model tắt/thiếu khóa/thiếu VRAM/không đủ năng lực (giữ nhân vật, đa tham chiếu, chữ trong ảnh); xếp theo tác vụ đề xuất rồi chi phí/ảnh đạt (benchmark) hoặc chi phí ước tính; dự phòng đầu tiên khác loại/nhà cung cấp. Kết quả ghi vào `route_plan`.
4. **Attempt (task `visual_run`, hàng đợi `visual`):** kiểm ngân sách → biên dịch prompt theo model → provider tạo `num_variants` ảnh (đo thời gian/VRAM/RAM) → lưu ảnh + ứng viên trong một giao dịch.
5. **Quality Gate** từng ảnh: kích thước, tỉ lệ, trống, mờ, phơi sáng, số mặt, nhận dạng với ảnh chuẩn, nhận dạng với các shot đã duyệt. Verdict `pass` / `review` (không đo được hoặc cảnh báo) / `fail`.
6. **Quyết định:** có ảnh `pass`/`review` → `review_required`; toàn bộ `fail` → sửa tự động (≤ `VISUAL_MAX_REPAIRS`=2, còn ngân sách; lượt cuối đổi tuyến) hoặc `needs_human`. Lỗi nhà cung cấp → chuyển tuyến kế tiếp (không tốn lượt sửa); API đồng bộ bị ngắt → coi là có thể đã tính phí, đổi nhà cung cấp.
7. **Người duyệt** (`/visual/jobs/{id}/review`): `approve` (ảnh QC lỗi cần `override_qc`), `reject` (kèm lý do), `repair` (chỉ dẫn; không tính vào 2 lượt tự động). Lưu tên + thời điểm.
8. **Xuất bản** (`/visual/jobs/{id}/export`): sRGB → khung 1080×1920/1920×1080 → nhãn AI → ký C2PA → đọc lại xác minh → lưu; thiếu dấu bắt buộc ⇒ `blocked`.
9. Ảnh được duyệt (ứng viên được chọn) là điều kiện để tạo **video** (`/video/jobs`, mục 6).

| Đối tượng | Giá trị |
|---|---|
| Job ảnh | `queued` · `running` · `review_required` · `approved` · `rejected` · `failed` · `needs_human` |
| Attempt ảnh | `queued` → `running` → `done` / `failed`; `kind` = `initial` · `repair` · `human_repair` · `manual` |
| Ứng viên (QC) | `pass` · `review` · `fail` · `pending` |
| Bản xuất | `released` · `blocked` |
