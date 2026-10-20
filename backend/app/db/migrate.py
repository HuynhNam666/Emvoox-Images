"""Đồng bộ cột còn thiếu (chưa dùng Alembic): thêm cột NULL-able/có default mới vào bảng đã tồn tại.
Chỉ THÊM cột; không đổi kiểu, không xoá, không đụng dữ liệu. Cột bắt buộc (NOT NULL không default) bị bỏ qua và báo cảnh báo."""
import logging

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from app.db.session import Base

log = logging.getLogger("migrate")


def sync_columns(engine: Engine) -> list[str]:
    insp = inspect(engine)
    added = []
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            have = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in have:
                    continue
                if not col.nullable and col.default is None and col.server_default is None:
                    log.warning("bỏ qua cột bắt buộc %s.%s: cần migration thủ công", table.name, col.name)
                    continue
                ddl = f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {col.type.compile(engine.dialect)}'
                conn.execute(text(ddl))
                added.append(f"{table.name}.{col.name}")
    if added:
        log.warning("đã thêm cột: %s", ", ".join(added))
    return added
