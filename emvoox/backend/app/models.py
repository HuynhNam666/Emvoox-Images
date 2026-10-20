"""Relational schema. Image bytes live only in asset_blobs (BYTEA)."""
import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger, Boolean, CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, LargeBinary,
    String, Text, UniqueConstraint, func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _id() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _fk(target: str, nullable: bool = False, ondelete: str = "RESTRICT"):
    return mapped_column(UUID(as_uuid=True), ForeignKey(target, ondelete=ondelete), nullable=nullable)


def _ts() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Project(Base):
    __tablename__ = "projects"
    id = _id()
    name: Mapped[str] = mapped_column(String(200), unique=True)
    created_at = _ts()


class Asset(Base):
    __tablename__ = "assets"
    id = _id()
    project_id = _fk("projects.id")
    kind: Mapped[str] = mapped_column(String(32))  # generated | dataset_image | reference | derived
    mime_type: Mapped[str] = mapped_column(String(64))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    byte_size: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    storage_backend: Mapped[str] = mapped_column(String(16), default="postgres")
    storage_key: Mapped[str | None] = mapped_column(String(512))  # only for local/s3 backends
    provenance: Mapped[dict] = mapped_column(JSONB, default=dict)  # ai_generated, provider, source...
    created_at = _ts()
    __table_args__ = (
        CheckConstraint("storage_backend in ('postgres','local','s3')", name="ck_assets_backend"),
        CheckConstraint(
            "(storage_backend = 'postgres' and storage_key is null) or "
            "(storage_backend <> 'postgres' and storage_key is not null)",
            name="ck_assets_key_matches_backend",
        ),
    )


class AssetBlob(Base):
    __tablename__ = "asset_blobs"
    asset_id = mapped_column(UUID(as_uuid=True), ForeignKey("assets.id", ondelete="CASCADE"), primary_key=True)
    data: Mapped[bytes] = mapped_column(LargeBinary)
    thumbnail: Mapped[bytes | None] = mapped_column(LargeBinary)  # small JPEG for galleries


class Character(Base):
    __tablename__ = "characters"
    id = _id()
    project_id = _fk("projects.id")
    code: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(200))
    current_version_id = mapped_column(UUID(as_uuid=True), nullable=True)  # soft pointer, avoids FK cycle
    created_at = _ts()
    __table_args__ = (UniqueConstraint("project_id", "code", name="uq_character_code"),)


class CharacterVersion(Base):
    __tablename__ = "character_versions"
    id = _id()
    character_id = _fk("characters.id")
    version: Mapped[int] = mapped_column(Integer)
    identity_spec: Mapped[dict] = mapped_column(JSONB, default=dict)
    trigger_token: Mapped[str | None] = mapped_column(String(64))
    created_at = _ts()
    __table_args__ = (UniqueConstraint("character_id", "version", name="uq_character_version"),)


class CharacterReferenceAsset(Base):
    __tablename__ = "character_reference_assets"
    character_version_id = mapped_column(UUID(as_uuid=True), ForeignKey("character_versions.id"), primary_key=True)
    asset_id = mapped_column(UUID(as_uuid=True), ForeignKey("assets.id"), primary_key=True)
    is_canonical: Mapped[bool] = mapped_column(Boolean, default=False)


class Dataset(Base):
    __tablename__ = "datasets"
    id = _id()
    project_id = _fk("projects.id")
    character_id = _fk("characters.id", nullable=True)
    name: Mapped[str] = mapped_column(String(200))
    objective: Mapped[str] = mapped_column(String(16), default="character")  # character | style
    trigger_token: Mapped[str | None] = mapped_column(String(64))
    created_at = _ts()


class DatasetItem(Base):
    __tablename__ = "dataset_items"
    id = _id()
    dataset_id = _fk("datasets.id")
    asset_id = _fk("assets.id")
    caption: Mapped[str] = mapped_column(Text, default="")
    tags: Mapped[list] = mapped_column(JSONB, default=list)
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|approved|rejected
    split: Mapped[str | None] = mapped_column(String(8))  # train|val|test, null = auto
    sha256: Mapped[str] = mapped_column(String(64))
    phash: Mapped[str] = mapped_column(String(16))
    source: Mapped[str | None] = mapped_column(String(300))
    license: Mapped[str | None] = mapped_column(String(200))
    synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    source_asset_id = _fk("assets.id", nullable=True)
    quality_flags: Mapped[list] = mapped_column(JSONB, default=list)
    created_at = _ts()
    __table_args__ = (
        UniqueConstraint("dataset_id", "sha256", name="uq_dataset_item_sha"),
        CheckConstraint("status in ('pending','approved','rejected')", name="ck_item_status"),
    )


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    id = _id()
    dataset_id = _fk("datasets.id")
    version: Mapped[int] = mapped_column(Integer)
    manifest_sha256: Mapped[str] = mapped_column(String(64))
    item_count: Mapped[int] = mapped_column(Integer)
    warnings: Mapped[list] = mapped_column(JSONB, default=list)
    created_at = _ts()
    __table_args__ = (UniqueConstraint("dataset_id", "version", name="uq_dataset_version"),)


class DatasetVersionItem(Base):
    """Immutable snapshot: caption/split are copied, later edits do not touch it."""
    __tablename__ = "dataset_version_items"
    dataset_version_id = mapped_column(UUID(as_uuid=True), ForeignKey("dataset_versions.id"), primary_key=True)
    asset_id = mapped_column(UUID(as_uuid=True), ForeignKey("assets.id"), primary_key=True)
    caption: Mapped[str] = mapped_column(Text)
    split: Mapped[str] = mapped_column(String(8))
    sha256: Mapped[str] = mapped_column(String(64))


class ModelVersion(Base):
    """Registry for base checkpoints and LoRA adapters."""
    __tablename__ = "model_versions"
    id = _id()
    kind: Mapped[str] = mapped_column(String(24))  # base | character_lora | style_lora
    name: Mapped[str] = mapped_column(String(200))
    family: Mapped[str] = mapped_column(String(32))  # sdxl
    hf_id: Mapped[str | None] = mapped_column(String(300))
    revision: Mapped[str | None] = mapped_column(String(64))
    license: Mapped[str | None] = mapped_column(String(100))
    capabilities: Mapped[dict] = mapped_column(JSONB, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="experimental")  # verified|experimental|unavailable
    parent_base_id = _fk("model_versions.id", nullable=True)
    trigger_token: Mapped[str | None] = mapped_column(String(64))
    artifact_key: Mapped[str | None] = mapped_column(String(512))  # path relative to data_dir
    artifact_sha256: Mapped[str | None] = mapped_column(String(64))
    training_run_id = mapped_column(UUID(as_uuid=True), nullable=True)
    review_status: Mapped[str] = mapped_column(String(20), default="approved")  # awaiting_review|approved|rejected
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at = _ts()


class TrainingRun(Base):
    __tablename__ = "training_runs"
    id = _id()
    project_id = _fk("projects.id")
    dataset_version_id = _fk("dataset_versions.id")
    base_model_id = _fk("model_versions.id")
    objective: Mapped[str] = mapped_column(String(16), default="character")
    trigger_token: Mapped[str] = mapped_column(String(64))
    config: Mapped[dict] = mapped_column(JSONB)  # resolved config
    state: Mapped[str] = mapped_column(String(20), default="queued")
    current_step: Mapped[int] = mapped_column(Integer, default=0)
    total_steps: Mapped[int | None] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(String(40))
    error_message: Mapped[str | None] = mapped_column(Text)
    trainer_revision: Mapped[str | None] = mapped_column(String(100))
    output_model_version_id = mapped_column(UUID(as_uuid=True), nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    enqueued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at = _ts()
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class TrainingMetric(Base):
    __tablename__ = "training_metrics"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id = mapped_column(UUID(as_uuid=True), ForeignKey("training_runs.id", ondelete="CASCADE"))
    step: Mapped[int] = mapped_column(Integer)
    loss: Mapped[float | None] = mapped_column(Float)
    lr: Mapped[float | None] = mapped_column(Float)
    recorded_at = _ts()
    __table_args__ = (Index("ix_training_metrics_run_step", "run_id", "step"),)


class GenerationJob(Base):
    __tablename__ = "generation_jobs"
    id = _id()
    project_id = _fk("projects.id")
    purpose: Mapped[str] = mapped_column(String(16), default="production")  # production | validation
    character_version_id = _fk("character_versions.id", nullable=True)
    model_version_id = _fk("model_versions.id")
    lora_version_id = _fk("model_versions.id", nullable=True)
    training_run_id = mapped_column(UUID(as_uuid=True), nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(128))
    request_hash: Mapped[str] = mapped_column(String(64))
    prompt_original: Mapped[str] = mapped_column(Text)
    prompt_compiled: Mapped[str] = mapped_column(Text)
    negative_prompt: Mapped[str | None] = mapped_column(Text)
    config: Mapped[dict] = mapped_column(JSONB)  # effective parameters
    state: Mapped[str] = mapped_column(String(20), default="queued")
    progress_step: Mapped[int | None] = mapped_column(Integer)
    progress_total: Mapped[int | None] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(String(40))
    error_message: Mapped[str | None] = mapped_column(Text)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    enqueued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at = _ts()
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    outputs = relationship("GenerationOutput", order_by="GenerationOutput.output_index", lazy="selectin")
    __table_args__ = (
        UniqueConstraint("project_id", "idempotency_key", name="uq_job_idempotency"),
        Index("ix_jobs_state", "state"),
    )


class JobAttempt(Base):
    __tablename__ = "job_attempts"
    id = _id()
    job_id = _fk("generation_jobs.id")
    attempt: Mapped[int] = mapped_column(Integer)
    worker: Mapped[str | None] = mapped_column(String(100))
    started_at = _ts()
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    outcome: Mapped[str | None] = mapped_column(String(20))
    error: Mapped[str | None] = mapped_column(Text)


class GenerationOutput(Base):
    __tablename__ = "generation_outputs"
    id = _id()
    job_id = _fk("generation_jobs.id")
    asset_id = _fk("assets.id")
    output_index: Mapped[int] = mapped_column(Integer)
    seed: Mapped[int] = mapped_column(BigInteger)
    __table_args__ = (UniqueConstraint("job_id", "output_index", name="uq_output_index"),)


class Review(Base):
    __tablename__ = "reviews"
    id = _id()
    subject_type: Mapped[str] = mapped_column(String(24))  # generation_job | model_version | asset
    subject_id = mapped_column(UUID(as_uuid=True))
    decision: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(Text)
    reviewer: Mapped[str] = mapped_column(String(100), default="local")
    created_at = _ts()


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id = _id()
    actor: Mapped[str] = mapped_column(String(100), default="local")
    action: Mapped[str] = mapped_column(String(64))
    subject_type: Mapped[str | None] = mapped_column(String(32))
    subject_id = mapped_column(UUID(as_uuid=True), nullable=True)
    detail: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at = _ts()
