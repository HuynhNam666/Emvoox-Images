# Workflow video: ảnh đã duyệt → clip (ComfyUI + LTX-2.3 + PromptRelay)

Triển khai theo `EMVOOX_Huong_dan_workflow_anh_video.docx` (bản gốc cùng thư mục). Tài liệu này ghi **cái gì đã có trong mã, cái gì đã kiểm chứng đến đâu, và việc phải làm trên máy**.
Nghiên cứu nền: [VIDEO_LTX_INTEGRATION.md](VIDEO_LTX_INTEGRATION.md).

> **Trạng thái trung thực:** toàn bộ phía EMVOOX (bảng, API, worker, UI, kiểm soát duyệt/ngân sách) đã có và được pytest kiểm tra **với ComfyUI giả**.
> **Cập nhật 20/10/2026:** workflow API đã được **dựng lại từ workflow mẫu của kijai và khớp schema ComfyUI thật** (ComfyUI 0.42.0 + PromptRelay + KJNodes + VideoHelperSuite; `scripts/validate_comfy_workflow.py`). **Chưa** có clip nào được render bằng LTX-2.3 (cần GPU lớn + model). Hướng dẫn chạy trên server: [SERVER_DEPLOY.md](SERVER_DEPLOY.md). Trạng thái workflow: `schema_validated` → đổi thành `verified` sau khi `scripts/smoke_video.py` ra clip đạt.

## 1. Điều kiện phần cứng (xác nhận trước)
Hướng dẫn: không có cấu hình GPU/RAM trong tệp nghiên cứu để xác nhận khả năng chạy → thử tải model và render một clip ngắn, ghi VRAM đỉnh, RAM, thời gian, lỗi.
Máy dự án: RTX 4060 8GB, RAM ~15,7GB. LTX-2.3 22B fp8 chỉ riêng trọng số ≈ 22GB nên rất có thể cần offload nặng hoặc GPU khác; nếu không chạy được, đặt `COMFYUI_URL` trỏ tới ComfyUI trên máy/worker khác (chỉ sau khi xác minh tuyến đó hỗ trợ ảnh→video + âm thanh + quyền sử dụng).

## 2. Cài ComfyUI và tài nguyên (làm tay, ngoài EMVOOX)
1. Cập nhật ComfyUI. Trong Manager cài rồi khởi động lại, đọc log khởi động:

   | Gói | Mục đích |
   |---|---|
   | `kijai/ComfyUI-PromptRelay` | prompt theo đoạn/timeline |
   | `kijai/ComfyUI-KJNodes` | VAE loader KJ, Set/Get, tiện ích |
   | `Kosinkadink/ComfyUI-VideoHelperSuite` | ghép khung hình + âm thanh → MP4 |
2. Nhập workflow mẫu: kho PromptRelay → `example_workflows/prompt_relay_ltx23_test_02.json` (Download raw) → kéo vào canvas, chọn lại model/ảnh. (Mẫu công khai gần với video, chưa xác nhận là file gốc của người hướng dẫn.)
3. Đặt model (nguồn `Kijai/LTX2.3_comfy`):

   | File | Thư mục trong ComfyUI |
   |---|---|
   | `ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors` | `models/diffusion_models/` |
   | `LTX23_video_vae_bf16.safetensors`, `LTX23_audio_vae_bf16.safetensors` | `models/vae/` |
   | Gemma 3 12B + `LTX23_text_projection_bf16.safetensors` | `models/text_encoders/` |
   | LoRA tương thích LTX (nếu workflow dùng) | `models/loras/` |

   Không trộn cách nạp checkpoint đầy đủ của template khác với cách nạp transformer tách rời (`UNETLoader`). LoRA ảnh SDXL/FLUX **không** nạp được vào LTX-2.3.
4. Chạy thử một clip ngắn **trong giao diện ComfyUI** trước. Chỉ khi chạy được mới sang bước 3.

## 3. Nối vào EMVOOX
```
Tab Video / POST /api/v1/video/jobs ─► queue "video" ─► worker (cùng pool solo với ảnh/train)
   release_pipeline() → kiểm tra node (/object_info) → upload ảnh (/upload/image) → /prompt
   → ghi prompt_id → chờ /history/{id} → /view lấy mp4 → asset_blobs (video/mp4) → /free
```
- **Workflow bản ghim:** `backend/app/ml/workflows/ltx23_promptrelay.api.json` + `…mapping.json`. Backend chỉ điền các tham số khai báo trong mapping (ảnh, kích thước, frame, fps, seed, prompt, tiền tố file); **không** nhận workflow từ client.
- **Workflow đã ghim:** `_meta.status = "schema_validated"`: graph dùng `PromptRelayEncode` (đầu vào `global_prompt`, `local_prompts` cách nhau `|`, `segment_lengths` là số frame, `epsilon`), `LTXVImgToVideoInplaceKJ` (input động `num_images.image_1`), `BasicScheduler linear_quadratic 8 bước`, `SamplerCustom cfg 1.0`, `VHS_VideoCombine h264-mp4`. Muốn thay bằng workflow riêng đã chạy được: ComfyUI → bật Dev mode → **Save (API Format)** → ghi đè `ltx23_promptrelay.api.json`, sửa `mapping.json` cho khớp id node, rồi chạy `scripts/validate_comfy_workflow.py`.
- **Đoạn chuyển động:** UI/API nhập đoạn theo **% thời lượng** (0–100); backend đổi thành số **frame** (tổng đúng bằng `frame_count`, phương pháp phần dư lớn nhất) và gửi qua `segment_lengths`; PromptRelay tự quy đổi sang latent. `segment_format` trong mapping (`pipe_plain`) chỉ dùng cho node Smart (nhãn `[0-25]`), không dùng ở workflow này.
- **Cấu hình:** `COMFYUI_URL` (mặc định `http://127.0.0.1:8188`; Docker: `http://host.docker.internal:8188`), `COMFYUI_TIMEOUT_S`, `VIDEO_COST_PER_GPU_SECOND`, `VIDEO_MAX_REPAIRS` (xem `.env.example`).
- **Worker:** hàng đợi `video` đã thêm vào lệnh worker (`scripts/run_worker.ps1`, `Dockerfile.gpu`). Nếu chạy tay, thêm `video` vào `-Q`.
- **Kiểm tra sẵn sàng:** `GET /api/v1/video/health` (hoặc dải cảnh báo ở tab Video) cho biết ComfyUI có kết nối được không, node nào thiếu, workflow đã kiểm chứng chưa.

## 4. Quy trình và luật chặn
1. Ảnh nguồn **phải được duyệt**: job ảnh `approved` (`POST /generation-jobs/{id}/review`, nút “Duyệt ảnh” trong Lịch sử) hoặc ảnh chuẩn của nhân vật. Chưa duyệt → HTTP 409.
2. `POST /api/v1/video/jobs` (kèm `Idempotency-Key`): `source_keyframe_asset_id`, `global_prompt`, `motion_segments` (%), `width/height` (chia hết 32), `frame_count` (8n+1, ≤257), `fps`, `seed`, `audio_mode` (`generated|none`), `budget_limit`. Phiên bản nhân vật được ghim từ job ảnh nguồn.
   `audio_mode=asset` (Image+Audio→Video) không được nhận (HTTP 422): tuyến này cần đánh giá riêng về phát âm tiếng Việt, khớp môi, giữ giọng.
3. Render xong → `review_required` (**không** có nghĩa đã duyệt). Người duyệt xem cả chuyển động và âm thanh: `POST /video/jobs/{id}/review`.
4. Loại clip phải chọn lý do, và lý do quyết định hướng đi:

   | Lý do | Hướng | Kết quả |
   |---|---|---|
   | `keyframe_identity` | sửa ảnh | job → `returned_to_image`, không retry video |
   | `motion_identity`, `wrong_action`, `visual_artifact`, `audio_issue`, `other` | sửa tuyến video | `rejected` → `POST /video/jobs/{id}/retry` (giảm chuyển động, đổi seed/prompt) |
   | `interaction_error` | tách shot / giảm số hành động | như trên |
5. **Giới hạn sửa:** tối đa 2 lượt sửa tự động (mỗi retry kể cả sau lỗi thực thi tính 1 lượt) và trần `budget_limit`; vượt → `needs_human`. Chi phí lượt hỏng/sửa vẫn được cộng.
6. **Timeout/crash:** `prompt_id` được ghi trước khi chờ. Timeout không hủy prompt trên ComfyUI; nếu worker chết khi attempt đang `running` có `prompt_id`, task giao lại sẽ nối vào `/history/{prompt_id}` thay vì gửi trùng. Hết thời gian thì attempt `failed` (`comfy_timeout`), có thể retry.
7. Trạng thái job: `queued → running → review_required → approved`; nhánh `rejected`, `failed`, `returned_to_image`, `needs_human`. Mọi attempt được lưu (`video_attempts`) cùng tham số, thời gian, chi phí, quyết định duyệt.
8. **Nguồn gốc:** asset video (`kind=generated_video`) mang provenance: ảnh nguồn + sha256, job ảnh nguồn, phiên bản nhân vật/cảnh, workflow + hash, model, seed, fps, frame, `prompt_id`, cờ `workflow_verified`. Video được phục vụ qua `/assets/{id}/content` với HTTP Range.

## 5. Cấu hình thử đề xuất (điểm xuất phát, chưa đo)
Một clip ~3–5 s, 2–3 đoạn đơn giản; 97 frame @ 25 fps ≈ 3,88 s; 512×896 (gần 9:16); giữ preset sampler của mẫu, đổi **một** biến mỗi lần; cố định seed khi so sánh, dùng nhiều seed trong benchmark. Lần đầu: một nhân vật, một hành động, camera nhẹ. Với hai nhân vật thử riêng lỗi trộn mặt/đổi áo/sai vị trí/sai người nói. Kiểm tra crop/pad khi xuất 1080×1920 (xuất 1080×1920 **chưa** có trong hệ thống).
Prompt ví dụ (minh họa, chưa phải nội dung phim) đã nạp sẵn làm mặc định trong form tab Video.

## 6. Pilot nghiệm thu
- Giữ benchmark ảnh 30 cảnh × 4 lần × 5 lựa chọn như báo cáo, **không** trộn số liệu video vào mẫu 600 ảnh.
- Khi đã có ảnh đạt: 6 shot × 2 seed × 2 cấu hình = **24 clip**: `backend/scripts/video_pilot.py --keyframe <asset_id>` (có `--dry-run`). 6 shot: cận mặt đổi biểu cảm, quay đầu, đi bộ, cầm đạo cụ, hai người tương tác, chuỗi hai shot.
- Số liệu: `GET /api/v1/video/metrics` và thẻ “Số liệu pilot”: tỷ lệ đạt (clip duyệt / clip đã xem), p50/p95 độ trễ, chi phí / clip đạt (gồm lượt hỏng/sửa), lý do loại.
- Ngưỡng đề xuất để thảo luận: không clip phát hành nào có lỗi loại về nhân vật/hành động; tỷ lệ đạt ≥ 80%. p95 và ngân sách do chủ dự án chốt **trước** khi chạy. Chỉ mở rộng khi cả chất lượng và chi phí nằm trong ngưỡng.
- Chi phí hiện đo bằng giây GPU × `VIDEO_COST_PER_GPU_SECOND` (mặc định 1 → đơn vị = giây). Chưa tính QA, lưu trữ, phân bổ.

## 7. Chưa làm / hạn chế
- Chưa render LTX-2.3 thật (workflow đã khớp schema, chưa verified); cần GPU ≥ 24 GB (ước tính từ kích thước file, chưa đo).
- Chưa có xuất 1080×1920, C2PA/watermark (lưu provenance nội bộ **không** đồng nghĩa đã ký C2PA).
- Chưa có Image+Audio→Video, giữ giọng, lời thoại tiếng Việt chính xác.
- Theo dõi tiến độ bằng polling `/history` (chưa dùng WebSocket); không đo VRAM đỉnh tự động (ghi tay khi thử).
- Giấy phép model/node (LTX-2.3, PromptRelay, KJNodes, VHS, Gemma) phải kiểm tra trước khi dùng thương mại. Chỉ cài node từ repo gốc, ghim commit.

## 8. Ràng buộc logic đã siết (rà soát 20/10/2026)
- `Idempotency-Key` được kiểm **trước** mọi kiểm tra khác: phát lại cùng khóa + cùng payload luôn trả job cũ (kể cả khi ảnh nguồn bị loại sau đó); hai request đua cùng khóa không gây lỗi 500.
- Duyệt và sửa dùng `UPDATE … WHERE state = …`: hai thao tác đồng thời chỉ một cái được ghi nhận (cái kia nhận 409). Đã duyệt thì không sửa được.
- Workflow được ghim theo phiên bản (tên + hash file) lúc tạo/sửa job. Worker thấy file trên đĩa đã đổi → `workflow_changed`, không chạy ngầm bản khác; bấm sửa lại để ghim bản mới.
- Không xếp hàng được lượt sửa (Redis lỗi) → hoàn lại lượt sửa đã trừ.
- Ảnh nguồn phải là ảnh (mime `image/*`); job ảnh bị loại sau khi duyệt thì không tạo được video mới từ ảnh đó.
