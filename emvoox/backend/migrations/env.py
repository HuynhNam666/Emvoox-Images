from alembic import context
from sqlalchemy import create_engine

from app.config import resolved_database_url
from app.db import Base
import app.models  # noqa: F401  registers tables

target_metadata = Base.metadata
url = resolved_database_url()


def run_migrations_online() -> None:
    engine = create_engine(url)
    with engine.connect() as conn:
        context.configure(connection=conn, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    run_migrations_online()
