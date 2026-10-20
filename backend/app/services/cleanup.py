"""Xóa an toàn: không để lại tham chiếu treo và không xóa ảnh còn được nơi khác dùng.

Quy tắc chung của các API xóa:
- Mọi thao tác xóa kiểm tra tham chiếu trước; còn bị dùng => HTTP 409 kèm danh sách nơi đang dùng (không xóa một phần).
- Ảnh/video (assets) chỉ bị xóa khi KHÔNG còn tham chiếu nào khác (`purge_assets`).
- Nhân vật/bối cảnh/dataset đã được dùng để huấn luyện hoặc tạo job: phải xóa job/run liên quan trước (giữ lịch sử và truy vết).
"""
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    Asset, AssetBlob, CharacterReference, Costume, DatasetItem, DatasetVersionItem, ExportRecord, GenerationOutput, SceneVersion, VideoAttempt, VideoJob,
    VisualCandidate, VisualJob,
)


def asset_usage(db: Session, asset_id: str) -> list[str]:
    """Danh sách nơi còn dùng asset này (rỗng = xóa được)."""
    use = []
    checks = [
        ("dataset", select(DatasetItem.id).where(DatasetItem.asset_id == asset_id)),
        ("version dataset đã khóa", select(DatasetVersionItem.id).where(DatasetVersionItem.asset_id == asset_id)),
        ("ảnh chuẩn nhân vật", select(CharacterReference.id).where(CharacterReference.asset_id == asset_id)),
        ("job ảnh (SDXL)", select(GenerationOutput.id).where(GenerationOutput.asset_id == asset_id)),
        ("ứng viên ảnh theo shot", select(VisualCandidate.id).where(VisualCandidate.asset_id == asset_id)),
        ("video (ảnh nguồn/đầu ra)", select(VideoJob.id).where((VideoJob.source_keyframe_asset_id == asset_id) | (VideoJob.output_asset_id == asset_id))),
        ("lượt render video", select(VideoAttempt.id).where(VideoAttempt.output_asset_id == asset_id)),
        ("bản xuất", select(ExportRecord.id).where((ExportRecord.source_asset_id == asset_id) | (ExportRecord.export_asset_id == asset_id))),
        ("trang phục", select(Costume.id).where(Costume.reference_asset_id == asset_id)),
    ]
    for label, q in checks:
        if db.scalar(q.limit(1)):
            use.append(label)
    for sv in db.scalars(select(SceneVersion)):
        if asset_id in (sv.reference_asset_ids or []):
            use.append("bối cảnh")
            break
    for vj in db.scalars(select(VisualJob).where(VisualJob.reference_asset_ids != [])):
        if asset_id in (vj.reference_asset_ids or []):
            use.append("tham chiếu của job ảnh")
            break
    return use


def purge_assets(db: Session, ids: list[str]) -> int:
    """Xóa các asset không còn ai dùng (gọi SAU khi đã xóa các hàng tham chiếu và flush). Trả số asset đã xóa."""
    n = 0
    for aid in dict.fromkeys(i for i in ids if i):
        a = db.get(Asset, aid)
        if a is None or asset_usage(db, aid):
            continue
        blob = db.get(AssetBlob, aid)
        if blob:
            db.delete(blob)
        db.delete(a)
        n += 1
    return n
