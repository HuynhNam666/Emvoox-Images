# IP-Adapter: dùng ảnh chuẩn để giữ diện mạo (17/10/2026)

## Tích hợp
- Adapter: `h94/IP-Adapter` → `sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors` (808MB) + image encoder CLIP ViT-H/14 `models/image_encoder` (2,4GB). License theo model card: **Apache-2.0** (xác minh 17/10/2026; các biến thể FaceID có license khác — không dùng).
- Registry: bản ghi `kind=ip_adapter`, status `experimental` (chưa duyệt chất lượng).
- Cách dùng: job `use_references=true` + `character_version_id` (lấy tối đa 4 ảnh chuẩn của version) hoặc `reference_asset_ids`; `reference_scale` 0–1 (mặc định 0,6). Backend từ chối khi nhân vật chưa có ảnh chuẩn / adapter không khả dụng. Provenance của ảnh lưu adapter, ảnh tham chiếu và scale.
- Cơ chế: embedding `hidden_states[-2]` của ViT-H, **trung bình** các ảnh tham chiếu (cách gộp do dự án chọn, chưa so với cách khác). Encoder nạp tạm trên CPU (fp16) rồi giải phóng; embedding được cache theo sha256 ảnh. Adapter nạp/gỡ theo từng job (không rò sang job khác); text encoder được đưa tạm sang CPU để vừa 8GB VRAM.
- Kết hợp được với LoRA (chạy thật nhánh G).

## Hiệu năng (RTX 4060 8GB, 768², 25 bước)
| | Thời gian/ảnh |
|---|---|
| Không IP | ~30 s (nóng) |
| IP-Adapter, nóng | ~49 s (adapter nạp ~3 s) |
| IP-Adapter, lần đầu sau khi worker khởi động | ~2,5 phút (embed tham chiếu ~15 s + khởi động kernel) |

Lưu ý vận hành: đặt `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` làm job chậm ~3,5× trên máy này (104 s so với 30 s) — **không dùng**. Bộ nhớ `reserved` vẫn vượt 8GB lúc khử nhiễu (tràn sang bộ nhớ chia sẻ của Windows): chạy được nhưng sát giới hạn.

## Đo nhận dạng (cảnh chưa học: 6 cảnh × 2 seed, cùng prompt/seed)
Cosine FaceNet với vector trung bình 18 ảnh A02 đã duyệt (**đã loại 3 ảnh chuẩn dùng làm điều kiện** khỏi mốc đo; leave-one-out median 0,943, p10 0,719). Các nhánh A–D đo lại với cùng mốc.

| Nhánh | Mô tả | Sim TB ± SD | ≥ p10 tham chiếu |
|---|---|---|---|
| C | không LoRA, "a man" (đối chứng) | 0,131 ± 0,173 (4/12 không thấy mặt) | 0/12 |
| B | LoRA chỉ trigger | 0,509 ± 0,186 | 1/12 |
| A | không LoRA, mô tả chữ | 0,685 ± 0,105 | 5/12 |
| D | LoRA + mô tả chữ | 0,801 ± 0,140 | 11/12 |
| F | IP-Adapter, "a man" | 0,807 ± 0,192 | 11/12 |
| G | LoRA + IP + mô tả chữ | 0,864 ± 0,020 | 12/12 |
| **E** | **IP + mô tả chữ** | **0,874 ± 0,019** | **12/12** |

Xem bằng mắt (`docs/evidence/evalIP_sheet.jpg`): E/F/G cho cùng một người đàn ông ở cả 6 cảnh, cảnh vẫn đúng yêu cầu (sân thượng, quán cà phê đọc sách, mưa + ô, bếp, xe đạp, ga tàu).

## Kết luận và hạn chế
- IP-Adapter dùng ảnh chuẩn **vượt cả LoRA 300 bước hiện có** về độ nhất quán và ổn định (SD 0,02 so với 0,14) mà không cần train; thêm LoRA (G) không cải thiện thêm so với E trên mẫu này.
- Tác dụng phụ quan sát được: trang phục (áo thun xám) và khung hình/biểu cảm bị kéo giống ảnh tham chiếu; nhánh F có 1 ảnh mặt cúi xuống (sim 0,178) làm SD lớn. Độ đa dạng tư thế/trang phục thấp hơn khi `reference_scale` cao.
- Số liệu có thiên lệch: mẫu nhỏ (12/nhánh), một bộ ảnh chuẩn (3 ảnh, do tôi chọn), ảnh tham chiếu và đối tượng đều do SDXL sinh từ cùng một mô tả; FaceNet huấn luyện trên mặt thật. Mọi ảnh đo đều là cận/trung cảnh (mặt lớn) — **chưa kiểm tra toàn thân, nghiêng/profile, nhiều ảnh chuẩn khác nhau, scale khác 0,6**.
- Ảnh chuẩn nên đa dạng trang phục/góc để giảm rò rỉ trang phục; thử scale 0,3–0,5 khi cần theo prompt chặt hơn.

Tái lập: `scripts/eval_identity.py --lora <id> --arms EFG --scale 0.6 --reuse docs/evidence/eval.json,docs/evidence/evalD.json`.
