from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import SessionLocal, get_db
from app.models import Asset, Character, CharacterReference, CharacterVersion

router = APIRouter(prefix="/api/v1")


class IdentitySpec(BaseModel):
    description_vi: str = ""          # mô tả gốc tiếng Việt
    identity_en: str = Field("", max_length=1000)  # đoạn nhận dạng gửi model
    negative_prompt: str = ""


class CharacterCreate(BaseModel):
    code: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=200)
    identity: IdentitySpec = IdentitySpec()


class VersionCreate(BaseModel):
    identity: IdentitySpec | None = None   # None = giữ nguyên spec hiện tại
    reference_asset_ids: list[str] | None = None  # None = giữ nguyên ảnh chuẩn hiện tại


class ReferenceAdd(BaseModel):
    asset_id: str


def _version_dict(db: Session, v: CharacterVersion) -> dict:
    refs = db.scalars(select(CharacterReference).where(CharacterReference.character_version_id == v.id))
    return {"id": v.id, "version": v.version, "identity": v.identity_spec,
            "references": [r.asset_id for r in refs], "created_at": v.created_at.isoformat() if v.created_at else None}


def _character_dict(db: Session, c: Character, with_versions: bool = False) -> dict:
    cur = db.get(CharacterVersion, c.current_version_id) if c.current_version_id else None
    d = {"id": c.id, "code": c.code, "name": c.name, "current_version": _version_dict(db, cur) if cur else None}
    if with_versions:
        vs = db.scalars(select(CharacterVersion).where(CharacterVersion.character_id == c.id)
                        .order_by(CharacterVersion.version.desc()))
        d["versions"] = [_version_dict(db, v) for v in vs]
    return d


def _new_version(db: Session, c: Character, identity: dict, refs: list[str]) -> CharacterVersion:
    last = db.scalar(select(CharacterVersion.version).where(CharacterVersion.character_id == c.id)
                     .order_by(CharacterVersion.version.desc()).limit(1)) or 0
    v = CharacterVersion(character_id=c.id, version=last + 1, identity_spec=identity, trigger_token=None)
    db.add(v)
    db.flush()
    for aid in dict.fromkeys(refs):  # bỏ trùng, giữ thứ tự
        db.add(CharacterReference(character_version_id=v.id, asset_id=aid))
    c.current_version_id = v.id
    return v


def _check_assets(db: Session, ids: list[str]) -> None:
    for aid in ids:
        if not db.get(Asset, aid):
            raise HTTPException(422, f"asset không tồn tại: {aid}")


@router.post("/characters", status_code=201)
def create_character(body: CharacterCreate, db: Session = Depends(get_db)):
    if db.scalar(select(Character).where(Character.code == body.code)):
        raise HTTPException(409, "character code đã tồn tại")
    c = Character(code=body.code, name=body.name)
    db.add(c)
    db.flush()
    _new_version(db, c, body.identity.model_dump(), [])
    db.commit()
    return _character_dict(db, c, True)


@router.get("/characters")
def list_characters(db: Session = Depends(get_db)):
    return [_character_dict(db, c) for c in db.scalars(select(Character).order_by(Character.code))]


@router.get("/characters/{character_id}")
def get_character(character_id: str, db: Session = Depends(get_db)):
    c = db.get(Character, character_id)
    if not c:
        raise HTTPException(404, "nhân vật không tồn tại")
    return _character_dict(db, c, True)


@router.post("/characters/{character_id}/versions", status_code=201)
def create_version(character_id: str, body: VersionCreate, db: Session = Depends(get_db)):
    c = db.get(Character, character_id)
    if not c:
        raise HTTPException(404, "nhân vật không tồn tại")
    cur = db.get(CharacterVersion, c.current_version_id)
    identity = body.identity.model_dump() if body.identity else cur.identity_spec
    if body.reference_asset_ids is None:
        refs = [r.asset_id for r in db.scalars(select(CharacterReference)
                                               .where(CharacterReference.character_version_id == cur.id))]
    else:
        refs = body.reference_asset_ids
        _check_assets(db, refs)
    _new_version(db, c, identity, refs)
    db.commit()
    return _character_dict(db, c, True)


@router.post("/characters/{character_id}/references", status_code=201)
def add_reference(character_id: str, body: ReferenceAdd, db: Session = Depends(get_db)):
    """Thêm ảnh chuẩn => tạo character version mới (job cũ vẫn trỏ về version đã dùng)."""
    c = db.get(Character, character_id)
    if not c:
        raise HTTPException(404, "nhân vật không tồn tại")
    _check_assets(db, [body.asset_id])
    cur = db.get(CharacterVersion, c.current_version_id)
    refs = [r.asset_id for r in db.scalars(select(CharacterReference)
                                           .where(CharacterReference.character_version_id == cur.id))]
    if body.asset_id in refs:
        raise HTTPException(409, "ảnh đã là ảnh chuẩn của version hiện tại")
    _new_version(db, c, cur.identity_spec, refs + [body.asset_id])
    db.commit()
    return _character_dict(db, c, True)


SAMPLES = [
    ("A01", "Nhân vật A01 (mẫu)", IdentitySpec(
        description_vi="Nữ Việt Nam khoảng 25 tuổi, tóc đen dài ngang lưng, mắt nâu, dáng mảnh.",
        identity_en="young Vietnamese woman, around 25 years old, long straight black hair, brown eyes, slim build",
        negative_prompt="blurry, low quality, deformed hands, extra fingers, watermark, text")),
    ("A02", "Nhân vật A02 (mẫu)", IdentitySpec(
        description_vi="Nam Việt Nam khoảng 30 tuổi, tóc đen ngắn, mắt nâu, vóc dáng cân đối.",
        identity_en="Vietnamese man, around 30 years old, short black hair, brown eyes, athletic build",
        negative_prompt="blurry, low quality, deformed hands, extra fingers, watermark, text")),
]


def seed_sample_characters() -> None:
    """Hồ sơ mẫu A01/A02 — chỉ có mô tả, KHÔNG có ảnh. Ảnh chuẩn phải do model tạo rồi người dùng duyệt."""
    with SessionLocal() as db:
        for code, name, spec in SAMPLES:
            if db.scalar(select(Character).where(Character.code == code)):
                continue
            c = Character(code=code, name=name)
            db.add(c)
            db.flush()
            _new_version(db, c, spec.model_dump(), [])
        db.commit()
