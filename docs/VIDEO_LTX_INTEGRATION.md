# Tích hợp tạo video (ComfyUI + LTX-2.3 + PromptRelay) — nghiên cứu

> **Cập nhật 20/10/2026:** phần tích hợp đã được cài vào hệ thống theo hướng dẫn `EMVOOX_Huong_dan_workflow_anh_video.docx` — xem [VIDEO_WORKFLOW.md](VIDEO_WORKFLOW.md). Mục 4 và 6 dưới đây là kế hoạch ban đầu; điểm khác: endpoint là `/api/v1/video/jobs` (không phải `/video-jobs`), dùng bảng `video_jobs` + `video_attempts`, ComfyUI chạy ngoài Compose. Các mục 1, 3, 5 (nghiên cứu, rào cản 8GB, rủi ro) vẫn còn hiệu lực.

Nguồn: reel Facebook "Xem Phim Cùng Josa" (caption quảng cáo "công cụ AI tạo ảnh & video không giới hạn", tặng phần mềm qua bình luận) và file video 6,9MB, dài ~413 s, 640×360, phụ đề tiếng Việt.
**Giới hạn của phân tích:** chỉ lấy 12 khung hình đều nhau trong ~7 phút và đọc phụ đề trên khung hình; không có âm thanh/transcript đầy đủ. Mọi chi tiết dưới đây là điều đọc được trên màn hình, chưa chạy thử.

## 1. Video thực chất nói gì
Đây không phải một "phần mềm" riêng. Đó là bài hướng dẫn (bài 12 của một khóa ComfyUI, gốc tiếng Trung) dựng workflow **ComfyUI** (chạy ở `127.0.0.1:8188`) để tạo video AI:

| Thành phần thấy trong video | Vai trò |
|---|---|
| UNet loader: `ltx-2.3-22b-distilled-1.1_transformer_only_fp8…` | model video LTX-2.3 22B bản distilled, fp8 |
| 2 × VAELoader KJ: `LTX23_video_vae_bf16`, `LTX23_audio_vae_bf16` (comfyui-kjnodes) | VAE video và VAE âm thanh (video có tạo âm thanh môi trường) |
| LoRA loader (chỉ model) | gắn LoRA phong cách vào model |
| **Prompt Relay Encode (Timeline)** | node chính của bài: chia timeline thành nhiều đoạn, mỗi đoạn một prompt (đổi hành động theo thời gian) |
| LTXV Conditioning (frame rate 25), custom sampler/K-sampler (cfg 8), LTX2 Sampling Preview Override | điều kiện hóa và lấy mẫu |
| Ảnh đầu vào (con thỏ) + prompt | chế độ ảnh → video |
| Danh sách workflow 001–013: LTX23 sinh video, nhiều ảnh dẫn hướng, nối video, SCAIL2 chuyển động tác, SDPose, SAM3 | các workflow mẫu đi kèm khóa học |

Độ tin cậy: tên model/node đọc từ ảnh chụp màn hình độ phân giải thấp; cần đối chiếu khi cài thật.

## 2. Khớp với EMVOOX thế nào
EMVOOX hiện có ảnh nhân vật chuẩn (`character_references`), IP-Adapter giữ diện mạo, hàng đợi Celery + worker GPU, ảnh lưu BYTEA trong `asset_blobs`. Khoảng trống: README ghi "xuất 1080×1920" chưa làm, và chưa có video. Ảnh → video là bước tự nhiên: lấy **ảnh chuẩn / ảnh vừa sinh làm khung đầu**, thêm prompt theo timeline.

## 3. Rào cản phần cứng — phải quyết trước khi viết mã
- README: máy đích là RTX 4060 **8GB VRAM**, 8GB là mức tối thiểu đã thử cho SDXL.
- Model là **22B tham số**; bản fp8 chỉ riêng trọng số đã cỡ ~22GB, lớn hơn VRAM 8GB nhiều lần. Chạy được hay không phụ thuộc offload sang RAM (chậm) và cấu hình ComfyUI; tôi **chưa kiểm chứng** trên máy này.
- Đề xuất: làm một "phép thử khả thi" thủ công trước (cài ComfyUI riêng, tải workflow, sinh 1 clip 3–5 s độ phân giải thấp), đo thời gian và VRAM, rồi mới tích hợp.

### Kết quả nghiên cứu model nhỏ (18/10/2026, chỉ qua tìm kiếm web, chưa chạy thử)
- Hướng dẫn từ phía LTX cho LTX-2.3 khuyên GPU 16–24GB → **không phù hợp 8GB**; chạy CPU cũng không được (RAM 15,7GB, ổ C: trống ~24GB).
- Ứng viên cho 8GB: **LTX-Video 2B distilled** (bản fp8 hoặc GGUF q4/q6/q8, node GGUF loader của ComfyUI). Nguồn: [LTX-Video Model Card](https://huggingface.co/Lightricks/LTX-Video), [bản GGUF/fp8 của cộng đồng](https://huggingface.co/biali/ltxv-gguf), [bài của LTX về chạy cục bộ](https://ltx.io/blog/run-video-generation-model-locally).
- Có nguồn cộng đồng nói bản Q8 sinh 720×480×121 frame chưa tới 1 phút trên RTX 4060 8GB — số này **chưa kiểm chứng** và không phải bản fp8.
- Quy tắc: kích thước chia hết cho 32, số frame dạng 8n+1; bắt đầu nhỏ (≈512×320, 49–97 frame). Phiên bản 0.9.8 distilled fp8 có báo cáo xuất hiện watermark trong video.
- Việc chưa làm: kiểm tra license của từng bản (model, GGUF, node), xác nhận workflow mẫu của ComfyUI-LTXVideo.

Vị trí cài đặt: toàn bộ phần mới (ComfyUI, model, cache) đặt trên ổ D:. Dự án đã chép sang `D:\emvoox`; cache pip ở `D:\pip-cache`.

## 4. Kiến trúc đề xuất (chưa cài)
ComfyUI chạy như dịch vụ riêng, EMVOOX chỉ gọi API của nó — không nhúng mã ComfyUI vào backend.

```
Frontend (tab "Video") ──► api: POST /api/v1/video-jobs
                              └─► Celery queue "video" ─► worker-video
                                     1. release_pipeline() (nhả VRAM SDXL)
                                     2. nạp ảnh khung đầu từ asset_blobs, upload lên ComfyUI (/upload/image)
                                     3. điền workflow_api.json (ảnh, prompt theo timeline, số frame, fps 25, seed)
                                     4. POST /prompt → theo dõi /history/{id} (hoặc websocket) → /view lấy mp4
                                     5. lưu mp4 vào asset_blobs (mime video/mp4) + provenance
                                     6. POST /free để ComfyUI nhả VRAM
```
- **Tách GPU:** một GPU, một job nặng mỗi lúc. Worker video dùng chung queue/pool `solo` với generation/training, hoặc chặn bằng khóa, để không tranh VRAM với SDXL/LoRA.
- **Dữ liệu:** thêm bảng `video_jobs` (hoặc `GenerationJob.config.type="video"`), asset `kind="generated_video"`; `create_all` không đổi cột cũ nên dùng bảng mới, không sửa bảng hiện có.
- **Workflow:** lưu `workflow_api.json` (xuất từ ComfyUI, định dạng API) trong repo kèm checksum; backend chỉ thay các trường đã khai báo, không cho client gửi workflow tùy ý.
- **Compose:** service `comfyui` trong profile `video`, volume weights riêng (model video rất nặng), cổng chỉ bind `127.0.0.1`.
- **API tối thiểu:** `POST /video-jobs` {character_version_id | start_asset_id, segments:[{start_s,end_s,prompt}], duration_s, seed}, `GET /video-jobs/{id}`, `GET /assets/{id}` phục vụ mp4 (cần hỗ trợ Range).
- **Frontend:** tab "Video": chọn ảnh khung đầu, trình sửa timeline (thanh các đoạn + prompt mỗi đoạn, giống node Prompt Relay), tiến độ, trình phát.

## 5. Rủi ro và việc cần làm trước
1. **Giấy phép:** kiểm tra license của LTX-2.3, các node ComfyUI (PromptRelay, KJNodes) và LoRA trước khi dùng thương mại.
2. **An toàn chuỗi cung ứng:** node ComfyUI tùy biến chạy mã Python tùy ý. Chỉ cài từ repo gốc đã kiểm tra, ghim commit; không cài "phần mềm tặng qua bình luận" từ bài đăng quảng cáo.
3. **Quyền riêng tư/pháp lý:** video có thể ghép mặt nhân vật; gắn provenance (nguồn AI) như ảnh hiện nay; chưa có C2PA/watermark.
4. Ảnh chuẩn của nhân vật phải là ảnh sinh nội bộ; không đưa ảnh người thật vào pipeline nếu chưa có quyền.

## 6. Các bước tiếp theo
1. (Thủ công) Cài ComfyUI + model + node, chạy workflow ảnh→video trên máy này, ghi VRAM/thời gian.
2. Nếu khả thi: xuất `workflow_api.json`, viết `app/ml/comfy_client.py`, task Celery `generate_video`, bảng `video_jobs`, endpoint, rồi tab frontend và test (có ComfyUI giả cho pytest).
3. Nếu không khả thi trên 8GB: dùng model video nhỏ hơn (cần nghiên cứu riêng) hoặc chạy ComfyUI trên GPU khác và cấu hình URL từ xa.
