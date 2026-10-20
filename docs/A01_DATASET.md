# Dataset nhân vật A01 (8 nhóm) — 17/10/2026

**Kết quả:** 52 ảnh sinh bằng SDXL → duyệt 46 / loại 6 → 8 dataset version bất biến (`A01 — khuôn mặt … bối cảnh`, mỗi nhóm v1).

| Nhóm | Ảnh duyệt | Split (train/val/test) |
|---|---|---|
| khuôn mặt | 6 | 4 / 1 / 1 |
| toàn thân | 7 | 5 / 1 / 1 |
| biểu cảm | 11 | 9 / 1 / 1 |
| tư thế | 8 | 6 / 1 / 1 |
| trang phục | 3 | 2 / 1 / 0 |
| tóc & phụ kiện | 3 | 2 / 1 / 0 |
| phong cách | 3 | 1 / 1 / 1 |
| bối cảnh | 5 | 3 / 1 / 1 |

## Khác với A02: dùng ảnh neo + IP-Adapter
1. Cập nhật hồ sơ A01 (nhận dạng rõ: tóc đen dài thẳng, mặt trái xoan, nốt ruồi má phải…).
2. Sinh 4 ảnh neo ứng viên, **chọn bằng mắt** ảnh #0 (`docs/evidence/a01_anchor_candidates.jpg`), đặt làm ảnh chuẩn.
3. Sinh 38 ảnh theo kế hoạch 8 nhóm với **IP-Adapter** (`reference_scale` 0,35–0,55 tùy nhóm) và **seed khác nhau** cho mỗi ảnh (A02 dùng seed cố định).
4. Xem contact sheet, thấy các nhóm tư thế/phụ kiện/biểu cảm bị lệch → sinh bổ sung 14 ảnh (`a01_regen.json`, scale 0,3, prompt mạnh hơn, negative chống ảnh ghép).
5. Duyệt có caption theo đúng nội dung nhìn thấy (`a01_decisions.json`), khóa version.

## Kết quả so với A02 (độ giống FaceNet với ảnh neo/ảnh đầu tiên)
- A01: tất cả 38 ảnh đầu ≥ 0,75 (kể cả toàn thân); không có ảnh gần trùng.
- A02 (seed cố định, không IP-Adapter): cận mặt ~0,9 nhưng toàn thân 0,15–0,26 (mặt nhỏ, không đáng tin), nhiều ảnh trùng bố cục.
→ IP-Adapter + seed đa dạng cho bộ ảnh nhất quán hơn hẳn và nhiều tư thế/góc hơn.

## Hạn chế còn lại (đã ghi trung thực trong caption)
- Yêu cầu "nghiêng profile", "mặt sau", "buồn/giận/ngạc nhiên", "đi/chạy/cúi", "kính", "đồng hồ" **không thực hiện được đầy đủ**: ảnh neo đang mỉm cười nên biểu cảm bị kéo theo; góc nghiêng chủ yếu là ¾.
- Có ảnh ghép nhiều khung (6 ảnh bị loại).
- Trang phục và nền hay là tông xám/be (rò rỉ từ ảnh neo); nhóm phụ kiện chỉ có 1 ảnh có phụ kiện thật (mũ).
- Hai nhóm trang phục và phụ kiện chỉ có 3 ảnh; phong cách 3 ảnh → nên bổ sung trước khi train.
- Chưa train LoRA cho A01.

Tái lập: `scripts/gen_character_dataset.py --code A01 --trigger emv_a01 --refs --seed 100` → `scripts/regen_character.py` → `scripts/curate_character.py`.
