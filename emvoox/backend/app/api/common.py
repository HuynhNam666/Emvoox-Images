import uuid

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models import Project


def err(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def get_or_404(db: Session, model, id_: uuid.UUID, what: str):
    row = db.get(model, id_)
    if row is None:
        raise err(404, "not_found", f"Không tìm thấy {what}.")
    return row


def default_project(db: Session) -> Project:
    p = db.query(Project).filter_by(name="default").first()
    if p is None:
        p = Project(name="default")
        db.add(p)
        db.commit()
    return p
