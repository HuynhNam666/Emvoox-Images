import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.common import default_project, err, get_or_404
from app.db import get_db
from app.models import Asset, AuditEvent, Character, CharacterReferenceAsset, CharacterVersion

router = APIRouter(prefix="/api/v1/characters", tags=["characters"])


class CharacterCreate(BaseModel):
    code: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=200)
    identity_spec: dict = {}
    trigger_token: str | None = Field(None, max_length=64)
    reference_asset_ids: list[uuid.UUID] = []
    canonical_asset_id: uuid.UUID | None = None


class VersionCreate(BaseModel):
    identity_spec: dict = {}
    trigger_token: str | None = None
    reference_asset_ids: list[uuid.UUID] = []
    canonical_asset_id: uuid.UUID | None = None


def _version_out(db: Session, v: CharacterVersion) -> dict:
    refs = db.query(CharacterReferenceAsset).filter_by(character_version_id=v.id).all()
    return {"id": v.id, "version": v.version, "identity_spec": v.identity_spec, "trigger_token": v.trigger_token,
            "created_at": v.created_at,
            "references": [{"asset_id": r.asset_id, "is_canonical": r.is_canonical} for r in refs]}


def _char_out(db: Session, c: Character) -> dict:
    cur = db.get(CharacterVersion, c.current_version_id) if c.current_version_id else None
    return {"id": c.id, "code": c.code, "name": c.name, "created_at": c.created_at,
            "current_version": _version_out(db, cur) if cur else None}


def _add_version(db: Session, c: Character, body: VersionCreate | CharacterCreate) -> CharacterVersion:
    last = db.query(CharacterVersion).filter_by(character_id=c.id).order_by(CharacterVersion.version.desc()).first()
    v = CharacterVersion(character_id=c.id, version=(last.version + 1 if last else 1),
                         identity_spec=body.identity_spec, trigger_token=body.trigger_token)
    db.add(v)
    db.flush()
    ids = set(body.reference_asset_ids)
    if body.canonical_asset_id:
        ids.add(body.canonical_asset_id)
    for aid in ids:
        if db.get(Asset, aid) is None:
            raise err(404, "not_found", f"Không tìm thấy asset {aid}.")
        db.add(CharacterReferenceAsset(character_version_id=v.id, asset_id=aid,
                                       is_canonical=(aid == body.canonical_asset_id)))
    c.current_version_id = v.id
    db.add(AuditEvent(action="character_version_created", subject_type="character", subject_id=c.id,
                      detail={"version": v.version}))
    return v


@router.post("", status_code=201)
def create_character(body: CharacterCreate, db: Session = Depends(get_db)) -> dict:
    p = default_project(db)
    c = Character(project_id=p.id, code=body.code, name=body.name)
    db.add(c)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise err(409, "duplicate_code", f"Mã nhân vật '{body.code}' đã tồn tại.")
    _add_version(db, c, body)
    db.commit()
    return _char_out(db, c)


@router.get("")
def list_characters(q: str | None = None, db: Session = Depends(get_db)) -> list[dict]:
    qs = db.query(Character)
    if q:
        like = f"%{q}%"
        qs = qs.filter(Character.name.ilike(like) | Character.code.ilike(like))
    return [_char_out(db, c) for c in qs.order_by(Character.created_at.desc()).limit(200)]


@router.get("/{character_id}")
def get_character(character_id: uuid.UUID, db: Session = Depends(get_db)) -> dict:
    return _char_out(db, get_or_404(db, Character, character_id, "nhân vật"))


@router.get("/{character_id}/versions")
def list_versions(character_id: uuid.UUID, db: Session = Depends(get_db)) -> list[dict]:
    get_or_404(db, Character, character_id, "nhân vật")
    vs = db.query(CharacterVersion).filter_by(character_id=character_id).order_by(CharacterVersion.version.desc())
    return [_version_out(db, v) for v in vs]


@router.post("/{character_id}/versions", status_code=201)
def new_version(character_id: uuid.UUID, body: VersionCreate, db: Session = Depends(get_db)) -> dict:
    """Changing the canonical image creates a new version; old jobs keep pointing at the old one."""
    c = get_or_404(db, Character, character_id, "nhân vật")
    v = _add_version(db, c, body)
    db.commit()
    return _version_out(db, v)
