# Đánh giá nhận dạng LoRA A02 (17/10/2026)

**Phạm vi:** thử nghiệm nội bộ, 1 lần train, 1 nhân vật. Không phải benchmark.

## Dữ liệu
- 38 ảnh SDXL sinh bằng `scripts/gen_character_dataset.py` (seed cố định, 8 nhóm) → duyệt bằng mắt + FaceNet (`scripts/curate_a02.py`): **29 duyệt / 9 loại**, 8 dataset version bất biến.
- Train 25 ảnh, val 2, test 2 (split bị hệ thống ép gộp ở nhóm "tư thế" vì ảnh test gần trùng ảnh train).
- Lỗi của khâu sinh ảnh: seed cố định làm các yêu cầu "nghiêng trái/phải", "mặt sau", "buồn/giận/lo lắng" không được tuân theo → caption gốc sai, đã sửa theo nội dung thực; 4 ảnh ra nét vẽ line-art bị loại; thiếu đa dạng góc mặt và biểu cảm; nhiều ảnh đơn sắc (đã ghi "black and white photo" vào caption).

## Train
SDXL 1.0 base (`462165984030`), diffusers v0.41.0 DreamBooth-LoRA, 512px, rank 16, lr 1e-4, batch 1 × accum 4, **300 bước**, fp16, RTX 4060 8GB. Hủy ở bước 101 rồi **resume từ checkpoint-100** (đã kiểm chứng: trainer nạp lại state, loss không trùng bước). Loss thật: TB 10 bước đầu 0,130 → giữa 0,048 → 10 cuối 0,118 (nhiễu theo bước; không chứng minh chất lượng).
LoRA `5636e7b0…` (sha256 `467b966eb8…`), trạng thái **awaiting_review**.

## Đo nhận dạng (cảnh chưa học: 6 cảnh × 2 seed = 12 ảnh/nhánh, cùng prompt/seed)
Chỉ số: cosine FaceNet với vector trung bình 21 ảnh tham chiếu. Mốc: leave-one-out của chính tham chiếu có median 0,927, p10 0,719.

| Nhánh | Mô tả | Mặt phát hiện | Sim TB ± SD | ≥ p10 tham chiếu |
|---|---|---|---|---|
| A | không LoRA, mô tả chữ | 12/12 | 0,693 ± 0,103 | 5/12 |
| B | LoRA, **chỉ trigger** | 12/12 | 0,510 ± 0,188 | 1/12 |
| C | không LoRA, "a man" (đối chứng) | 8/12 | 0,130 ± 0,172 | 0/12 |
| D | LoRA + trigger + mô tả chữ | 12/12 | **0,802 ± 0,141** | 11/12 |

## Kết luận (thận trọng)
- LoRA **một mình** (B) chưa đủ: trên sheet nhiều ảnh ra phụ nữ — caption train không có từ "man" nên trigger chưa gắn chắc giới tính/nhận dạng.
- LoRA **kết hợp mô tả chữ** (D) cho khuôn mặt nhất quán rõ nhất, cao hơn A trên số liệu và bằng mắt (cùng một người qua 6 cảnh).
- Hạn chế khi đọc số: ảnh D chủ yếu cận/trung cảnh (mặt lớn) còn A có cả toàn thân (mặt nhỏ) → thiên lệch có lợi cho D; mẫu nhỏ (12/nhánh), một seed train; ảnh tham chiếu cũng do chính SDXL + mô tả chữ sinh ra nên nhánh A có lợi thế "trong phân phối". FaceNet được huấn luyện trên mặt thật, ảnh AI có thể lệch.
- Chưa kiểm tra: các góc nghiêng/profile/biểu cảm mạnh (dữ liệu không có), LoRA scale khác 1.0, nhiều seed train.

## Cải thiện đề xuất (chưa làm)
1. Thêm từ lớp vào caption và trigger (`emv_a02 man`).
2. Sinh thêm ảnh bằng nhiều seed/IP-Adapter để có góc nghiêng, profile, biểu cảm thật; bỏ ảnh đơn sắc/line-art.
3. Train lại 500–800 bước và so sánh nhiều checkpoint trên cùng bộ prompt đánh giá.
4. Duyệt LoRA chỉ sau khi đạt tiêu chí do người dùng đặt; hiện không khuyến nghị dùng không kèm mô tả chữ.

Tái lập: `scripts/gen_character_dataset.py` → `scripts/face_sim.py` → `scripts/curate_a02.py` → train qua UI/API → `scripts/eval_identity.py --lora <id> [--arms D]`.
