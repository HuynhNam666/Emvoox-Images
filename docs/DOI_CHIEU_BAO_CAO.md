# Đối chiếu báo cáo EMVOOX1 ↔ hệ thống (20/10/2026)

Đánh giá trung thực từng yêu cầu của báo cáo `EMVOOX1.docx`: nằm ở đâu trong mã, đã kiểm chứng đến mức nào.
Ký hiệu: ✅ có và đã kiểm tra · 🟡 có, chưa chạy thật (cần server) · ⬜ chưa làm / ngoài phạm vi.

## 1. Benchmark công cụ và mô hình

| Yêu cầu | Trạng thái | Ở đâu |
|---|---|---|
| Năm lựa chọn: FLUX 3 Image, FLUX.2 Klein 4B, Gemini Nano Banana 2.1, Midjourney, SDXL 1.0 | 🟡 | `app/visual/registry.py`. Klein 4B và SDXL tự chạy; Gemini/FLUX qua API; Midjourney nhập thủ công. **"FLUX 3 Image" và "Nano Banana 2.1" không xác minh được endpoint/id công khai** → model ở trạng thái `unverified`/`connector_unverified`, cấu hình bằng biến môi trường. |
| Model Registry (tên, nhà cung cấp, chức năng, tham chiếu, training, giấy phép, giá, ngày kiểm tra, benchmark) | ✅ | bảng `provider_models`; tab **Model**; `GET /visual/models` (kèm khả dụng + lý do) |
| Phủ các họ FLUX, Gemini, Seedream, GPT Image, Qwen Image, Z-Image, SDXL, Midjourney | 🟡 | 13 model: sdxl-1.0, flux2-klein-4b, flux2-klein-base-4b, qwen-image, qwen-image-edit, z-image-turbo, flux2-klein-4b-api, flux2-pro, flux3-image, gemini-nano-banana, seedream-4-5, gpt-image, midjourney. Báo cáo ghi "Z-Image FLUTurbo": không xác minh được tên đó, đối chiếu với `Tongyi-MAI/Z-Image-Turbo`. |
| Bộ thử 30 cảnh × 4 lần × 5 lựa chọn = 600 ảnh/vòng; 6 nhóm cảnh | ✅ (bộ cảnh + công cụ) | `app/visual/benchmark_scenes.json` (30 cảnh, 6 nhóm × 5); `POST /benchmark/runs` |
| Vòng 1 prompt nền, vòng 2 tham chiếu/LoRA | ✅ | `round_no` 1/2 (vòng 1 bỏ toàn bộ tham chiếu) |
| Hai người chấm độc lập, ẩn tên công cụ; trọng số 40/30/20/10; lỗi sai nhân vật/hành động là lỗi loại | ✅ | `GET /benchmark/runs/{id}/next` (ẩn model, xáo theo người chấm), `POST /benchmark/ratings` (mỗi ảnh đúng 2 người khác nhau), tab **Benchmark**; ảnh `core_error` luôn không đạt |
| Tỷ lệ đạt, p50/p95, GPU/VRAM/bước/batch | ✅ | `GET /benchmark/runs/{id}` + `attempts.metrics` |
| C đạt = (C tạo + C sửa + C GPU + C QA + C lưu trữ + C huấn luyện phân bổ) / N đạt | ✅ | `app/reports.py` (định nghĩa từng thành phần ở đầu file) ; tab **Báo cáo** |
| Chạy 600 ảnh và điền kết quả benchmark | ⬜ | cần server + khóa API + người chấm. Kết quả `Apply` ghi cost/ảnh đạt vào registry để Router dùng. |
| Router ưu tiên tuyến rẻ khi cùng vượt QC | ✅ | `app/visual/router.py` (rẻ nhất theo chi phí/ảnh đạt nếu có benchmark, không thì chi phí ước tính) |
| Bảng tác vụ → ứng viên (concept, hành động, cận mặt, hai người, poster chữ, series) | ✅ | cột `tasks` của registry + `classify_task` |

## 2–3. Kiến trúc và luồng

| Thành phần báo cáo | Trạng thái | Ở đâu |
|---|---|---|
| Character Library (A01/A02, ảnh chuẩn, phiên bản) | ✅ | đã có từ trước (`api_characters.py`) |
| Scene Library (mã địa điểm, bố cục, cửa, giờ, hướng sáng, bảng màu, đạo cụ, ảnh tham chiếu) + phiên bản | ✅ | `Scene`/`SceneVersion`, `POST /scenes`; B02 được seed cho pilot |
| Trang phục theo cảnh | ✅ | `Costume`, `POST /characters/{id}/costumes` |
| Proprietary dataset + quyền sử dụng | ✅ | rà quyền theo license/synthetic (`app/dataset_rights.py`); `COMMERCIAL_USE` chặn train |
| Prompt Compiler (thứ tự nhận dạng→hành động→trang phục→bối cảnh→cỡ cảnh→ánh sáng→phong cách→giữ; gán vai trò ảnh tham chiếu) | ✅ | `app/visual/compiler.py` (có test kiểm thứ tự) |
| Consistency Engine (tham chiếu, LoRA tương thích, theo dõi liên tục) | ✅ (một phần) | tham chiếu + LoRA SDXL đã duyệt; cảnh báo ánh sáng/trang phục lệch giữa các shot; kiểm nhận dạng chuỗi shot. **Điều khiển pose/depth (ControlNet) chưa làm.** |
| Model Router (chính + dự phòng, trạng thái, khả năng, ngân sách) | ✅ | `app/visual/router.py` |
| Render Worker (ComfyUI cho tự chạy, bộ kết nối API) | 🟡 | `app/visual/providers/*`; video qua ComfyUI. Tự chạy dùng diffusers (không qua ComfyUI) cho ảnh. |
| Quality Gate (nhận dạng, đúng cảnh, hình thể, quy chuẩn) | ✅ (tự động một phần) | `app/visual/qc.py`: kích thước, tỉ lệ, trống, mờ, phơi sáng, số mặt, nhận dạng FaceNet, nhận dạng chuỗi shot. **Đúng hành động, hình thể, thẩm mỹ vẫn do người duyệt** (báo cáo cũng nói vậy). |
| Cost Monitor (phí tạo/sửa, ngân sách trước lượt tiếp theo, dừng khi hết) | ✅ | `VisualJob.budget_limit`, kiểm trước mỗi lượt, `needs_human` khi hết |
| Asset Store + lịch sử xử lý, hồ sơ nguồn gốc | ✅ | `assets.provenance`, `visual_attempts`, `export_records` |
| `POST /visual/jobs` (+ idempotency), `GET /visual/jobs/{id}`, `POST /visual/jobs/{id}/review` | ✅ | `app/api_visual.py`; trạng thái queued/running/review_required/approved/failed (+ rejected, needs_human) |
| Review lưu người duyệt + thời điểm | ✅ | `reviewer`, `reviewed_at`, `review_history` (người dùng tự nhập tên) |
| Giới hạn 2 lượt sửa tự động + trần chi phí/cảnh | ✅ | `VISUAL_MAX_REPAIRS`, `budget_limit` |
| Lỗi quay lại bước 2 (biên dịch) hoặc 4 (tạo) | ✅ | `qc.failure_instruction` → `compile`/`render` |
| API timeout: kiểm job cũ trước khi chuyển tuyến | ✅ | BFL lưu `polling_url` và nối lại; API đồng bộ bị ngắt ⇒ coi là có thể đã tính phí, chuyển nhà cung cấp khác (không gọi lại) |
| Khi dùng model dự phòng không dùng lại LoRA của model chính | ✅ | LoRA chỉ áp cho SDXL; router không coi LoRA là đủ cho model khác |
| Cấu hình thử: SDXL 25–35 bước, CFG 5–7, DPM++ 2M Karras; LoRA rank 16/32, lr 5e-5/1e-4, trọng số 0,6/0,8/1,0 | 🟡 | SDXL mặc định 30 bước, CFG 6, DPM++ 2M Karras (`SDXL_SCHEDULER=default` để quay lại scheduler cũ; **chưa chạy lại trên GPU**); LoRA suy luận 0,8 |
| Tạo 2–4 phương án/cảnh, giữ seed + phiên bản | ✅ | `num_variants` 1–4, seed mỗi ảnh = seed gốc + chỉ số |
| Xuất 1080×1920 / 1920×1080, sRGB, không kéo méo | ✅ | `app/visual/export.py` (cover + cắt giữa, từ chối nếu phải cắt > 25%) |
| ControlNet / điều khiển pose-depth; inpaint ghép nhiều nhân vật | ⬜ | chưa làm (báo cáo: "thêm khi có lỗi cụ thể") |
| Style LoRA | ⬜ | chưa làm (báo cáo: khởi đầu bằng quy chuẩn phong cách) |

## 3.6 Nhãn AI và nguồn gốc

| Yêu cầu | Trạng thái | Ghi chú |
|---|---|---|
| Nhãn hiển thị "Hình ảnh được tạo/chỉnh sửa bằng AI", cỡ 28–36 px, lề ≥ 48 px (rộng 1080), không che mặt | ✅ | tránh khuôn mặt (nếu FaceNet khả dụng); cỡ/lề là thông số nội bộ như báo cáo nói |
| C2PA: ký sau mọi thay đổi điểm ảnh, đọc lại xác minh, ghi nguồn AI + thao tác | ✅ (chứng thư thử nghiệm) | ảnh và video đã ký + xác minh thật (c2pa-python 0.38). Cần chứng thư được tin cậy cho phát hành công khai |
| Watermark ẩn | ⬜ | không tự nhúng (đúng như báo cáo: chỉ khi có bộ mã hóa + phát hiện); ghi nhận dấu nhà cung cấp nếu có |
| Hồ sơ nội bộ không công khai prompt/dữ liệu cá nhân | ✅ | chỉ lưu SHA-256 của prompt trong hồ sơ xuất |
| Chặn phát hành khi thiếu dấu bắt buộc | ✅ | `REQUIRE_LABEL`, `REQUIRE_C2PA` ⇒ bản xuất `blocked`, không tải được |
| Kiểm thử bản nén/đổi kích thước/tải lại từ nền tảng | ⬜ | chưa có; metadata có thể bị nền tảng gỡ, phải giữ bản gốc đã ký |

## Video (từ tài liệu "workflow ảnh và video")

| Yêu cầu | Trạng thái |
|---|---|
| ComfyUI + PromptRelay + KJNodes + VHS, LTX-2.3 distilled fp8 | 🟡 workflow API **khớp schema ComfyUI thật** (kiểm bằng `/object_info` và `/prompt`); chưa render clip (cần GPU + model). `deploy/comfyui/Dockerfile` ghim đúng commit đã kiểm. |
| Ảnh phải duyệt trước khi tạo chuyển động; chuỗi duyệt | ✅ ảnh nguồn phải là ứng viên được chọn của job ảnh đã duyệt (hoặc ảnh chuẩn) |
| Sửa tối đa 2 lượt, ngân sách, quay về công đoạn gây lỗi | ✅ |
| Xuất video có nhãn AI + C2PA | ✅ (ffmpeg + c2pa thật) |
| Image+Audio→Video, giữ giọng, thoại tiếng Việt | ⬜ (cần đánh giá riêng) |

## Dataset

| Nhóm | Trạng thái |
|---|---|
| Nhân vật A01/A02 8 nhóm (46 + 25 ảnh đã có) | có sẵn; mở rộng lên 200–300 ảnh/nhân vật bằng `scripts/expand_library.py` (🟡 cần server) |
| Bối cảnh (Unsplash Lite thương mại được), tư thế (COCO/MPII), khuôn mặt (FFHQ) | có công cụ nạp (`public_datasets.py`), license ghi rõ phi thương mại; chưa tải về |
| Biểu cảm/bối cảnh/trang phục dataset công khai | **không dùng được để train**: ảnh 48–256 px thấp hơn ngưỡng 512 px; ghi trong `datasets/registry.json` (`training_use`) |
| Khuôn mặt Việt Nam mở | không tìm thấy bộ nào đã xác minh; dùng ảnh tự sinh + ảnh chuẩn |
| 30 cảnh benchmark | ✅ |
| Rà quyền trước khi train | ✅ |
