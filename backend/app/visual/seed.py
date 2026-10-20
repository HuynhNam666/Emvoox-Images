"""Dữ liệu khởi đầu cho pilot (hai nhân vật A01/A02, một tập phim): bối cảnh B02 và trang phục mẫu từ ví dụ trong báo cáo."""
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Character, Costume, Scene, SceneVersion


def seed_pilot_library(db: Session) -> None:
    if not db.scalar(select(Scene).where(Scene.code == "B02")):
        s = Scene(code="B02", name="Hành lang B02")
        db.add(s)
        db.flush()
        v = SceneVersion(scene_id=s.id, version=1, reference_asset_ids=[], spec={
            "layout": "a quiet narrow indoor corridor with a door at the far end on the right",
            "openings": "tall window on the left wall, door on the right wall near the end",
            "time_of_day": "late morning", "light_direction": "camera left (window)",
            "palette": "warm off-white walls, grey floor", "props": ["wall lamp", "wooden bench"]})
        db.add(v)
        db.flush()
        s.current_version_id = v.id
    a01 = db.scalar(select(Character).where(Character.code == "A01"))
    if a01 and not db.scalar(select(Costume).where(Costume.character_id == a01.id, Costume.code == "WHITE_SHIRT")):
        db.add(Costume(character_id=a01.id, code="WHITE_SHIRT", description_en="a plain white shirt"))
    db.commit()
