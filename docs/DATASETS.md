# EMVOOX — Danh mục dataset (tra cứu ngày 16/10/2026)

Trạng thái xác minh: **[đã thấy]** = có trích dẫn license trong kết quả tra cứu; **[chưa xác minh]** = cần đọc trang chính thức trước khi tải/train. Kết quả tra cứu phần lớn qua nguồn thứ cấp — luôn đọc license trên trang gốc.

## Nguyên tắc quan trọng

1. **Dataset đại trà (nhiều người) KHÔNG dùng để train Character LoRA.** Character LoRA cần 30–60 ảnh của *đúng một* nhân vật (xem prompt EMVOOX, mục Dataset Manager). Dataset đại trà chỉ dùng cho: pose/ControlNet reference, đánh giá, biểu cảm, bối cảnh, phong cách.
2. **Cách đúng để có dataset nhóm 1–6 của nhân vật A01/A02:** sinh ảnh nhân vật bằng SDXL → duyệt tay → gắn caption → khóa dataset version. Ảnh AI gắn `synthetic` + `source_asset_id`.
3. **Dự án thương mại (phim dọc EMVOOX)** → ưu tiên license cho phép thương mại. Các dataset "non-commercial" chỉ dùng nghiên cứu nội bộ, không đưa vào model phát hành.

## Bảng theo 8 nhóm

| Nhóm | Dataset | License | Dùng cho EMVOOX | Trạng thái |
|---|---|---|---|---|
| 1. Khuôn mặt | [FFHQ](https://github.com/NVlabs/ffhq-dataset) (+ [FFHQ-Aging](https://github.com/royorel/FFHQ-Aging-Dataset) có head pose yaw/pitch/roll) | CC BY-NC-SA 4.0 (non-commercial) | Đánh giá, tham chiếu góc mặt. Không train bản thương mại | [đã thấy] |
| 1. Khuôn mặt | CelebA / CelebA-HQ | Chỉ nghiên cứu học thuật | Chỉ nghiên cứu | [đã thấy] |
| 1. Khuôn mặt (châu Á) | Asian Face Dataset (AFD, arXiv 2004.03074) | Chưa rõ | Cần kiểm tra paper; không thấy dataset khuôn mặt Việt Nam mở nào | [chưa xác minh] |
| 2/5. Toàn thân + trang phục | [DeepFashion](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) | Non-commercial, cấm phân phối lại, cần ký agreement | Nghiên cứu trang phục | [đã thấy] |
| 2. Toàn thân | SHHQ (Shanghai AI Lab, trong StyleGAN-Human) | Chưa rõ | Cần đọc repo gốc | [chưa xác minh] |
| 3. Biểu cảm | FER2013 (6/7 cảm xúc: vui, buồn, giận, ngạc nhiên…) | Nguồn thứ cấp nói CC BY 4.0; ảnh 48×48 xám | Quá nhỏ để train; chỉ làm nhãn tham chiếu | [đã thấy, cần kiểm tra lại] |
| 3. Biểu cảm | AffectNet (~440k, 8 biểu cảm) | Non-commercial | Chỉ nghiên cứu | [đã thấy] |
| 4. Tư thế | COCO / MPII / Human3.6M (keypoints) | COCO: annotations CC BY 4.0, ảnh Flickr license hỗn hợp | Trích skeleton cho ControlNet-OpenPose (extension sau tuyến đầu) | [chưa xác minh] |
| 7. Phong cách (anime) | Danbooru2021 / mirror HF (`picollect/danbooru`, `KBlueLeaf/Danbooru2021-SQLite`) | Ảnh do người dùng upload, bản quyền hỗn hợp; trường license mirror không rõ | Rủi ro bản quyền cao; chỉ dùng tag/metadata, không train thương mại | [đã thấy một phần] |
| 7. Phong cách (nhân vật nhỏ) | [AICU HibikiMei](https://huggingface.co/AICU) — 25 ảnh 1 nhân vật: mặt trước/bên/sau + biểu cảm | CC BY-NC 4.0 | **Mẫu cấu trúc đúng** cho dataset nhân vật; dùng để thử pipeline | [đã thấy] |
| 8. Bối cảnh | [Unsplash Dataset Lite](https://github.com/unsplash/datasets) (25k ảnh) | Cho phép thương mại + train ML nội bộ theo terms của dataset (bản Full: non-commercial) | **Lựa chọn tốt nhất** cho cảnh phòng/đường/văn phòng/ngày-đêm | [đã thấy] |
| 8. Bối cảnh | Places365 (MIT CSAIL) | Chưa xác minh | Phân loại cảnh | [chưa xác minh] |
| 8. Bối cảnh | Open Images (CC BY phần ảnh) | Chưa xác minh | Cần lọc theo license từng ảnh | [chưa xác minh] |

## Đề xuất dùng thực tế (thứ tự)

1. **Nhóm 1–6 (identity/body/expression/pose/outfit/appearance):** tự sinh bằng SDXL cho A01/A02 theo checklist 8 nhóm → duyệt → dataset version. Đây là nguồn duy nhất đúng danh tính.
2. **Nhóm 8 (scene):** Unsplash Lite → dùng làm thư viện tham chiếu bối cảnh/prompt (không train identity).
3. **Nhóm 4 (pose):** trích keypoint từ COCO/MPII hoặc ảnh tự chụp cho ControlNet sau này.
4. **Nhóm 7 (style):** train Style LoRA riêng từ ảnh có quyền sử dụng (tự vẽ/tự sinh/có license), không dùng Danbooru cho bản thương mại.
5. **Thử pipeline sớm:** HibikiMei (CC BY-NC) để kiểm tra upload → caption → train plumbing, không phát hành.

## Việc cần người dùng quyết định / thực hiện

- Dự án có thương mại không? (quyết định dataset nào được dùng)
- Tải thủ công: DeepFashion/CelebA/AffectNet cần đăng ký/ký agreement — tôi không tự đăng ký hay tải thay.
- Chưa tải file nào về máy; thư mục `emvoox` hiện còn trống (chưa có mã nguồn).


## Cập nhật 20/10/2026 — bổ sung và rà quyền
- `datasets/registry.json` có thêm: bản sao Hugging Face của Places365 (val), FER2013, MPII, Anime-Faces, CelebA-HQ 256, DeepFashion In-Shop, bộ **ảnh tự sinh A01/A02** và **30 cảnh benchmark**. Mỗi dòng có `commercial` và `training_use`.
- **Kết luận thực tế:** các bản sao công khai của Places365/FER2013/Anime-Faces/CelebA-HQ có ảnh 48–256 px, thấp hơn ngưỡng 512 px của Dataset Manager (tự loại `low_resolution`) → **không train được**, chỉ làm tham chiếu/đánh giá. Nguồn train đúng danh tính vẫn là ảnh tự sinh.
- `scripts/expand_library.py` sinh ~30 ảnh × 8 nhóm / nhân vật qua pipeline mới (Quality Gate lọc), đưa vào dataset chờ duyệt (`synthetic=true`) — cách thực tế để đạt thư viện 200–300 ảnh/nhân vật.
- `scripts/public_datasets.py prepare mpii` (tư thế, nghiên cứu) + `prepare unsplash|coco|ffhq` như trước.
- **Rà quyền khi train:** `COMMERCIAL_USE=true` ⇒ preflight chặn dataset có license phi thương mại/chỉ nghiên cứu hoặc chưa ghi license; dataset toàn ảnh AI tự sinh được chấp nhận (`app/dataset_rights.py`). Chỉ là kiểm tra chuỗi văn bản, không thay cho đọc điều khoản gốc.
- License trên Hugging Face là thẻ của **bản sao**, không phải điều khoản của bộ dữ liệu gốc.
