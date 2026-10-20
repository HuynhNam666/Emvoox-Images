"""Bảng lõi. Ảnh: metadata ở `assets`, bytes ở `asset_blobs` (BYTEA)."""
import uuid
from datetime import datetime

from sqlalchemy import (
    JSON, DateTime, ForeignKey, Integer, LargeBinary, String, Text, UniqueConstraint, func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Character(TimestampMixin, Base):
    __tablename__ = "characters"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    code: Mapped[str] = mapped_column(String(32), unique=True)  # vd A01
    name: Mapped[str] = mapped_column(String(200))
    current_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    versions: Mapped[list["CharacterVersion"]] = relationship(back_populates="character")


class CharacterVersion(TimestampMixin, Base):
    __tablename__ = "character_versions"
    __table_args__ = (UniqueConstraint("character_id", "version"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    character_id: Mapped[str] = mapped_column(ForeignKey("characters.id"))
    version: Mapped[int] = mapped_column(Integer)
    identity_spec: Mapped[dict] = mapped_column(JSON, default=dict)
    trigger_token: Mapped[str | None] = mapped_column(String(64))
    character: Mapped[Character] = relationship(back_populates="versions")


class CharacterReference(TimestampMixin, Base):
    """Ảnh chuẩn gắn với một character version (đổi ảnh chuẩn => version mới)."""
    __tablename__ = "character_references"
    __table_args__ = (UniqueConstraint("character_version_id", "asset_id"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    character_version_id: Mapped[str] = mapped_column(ForeignKey("character_versions.id"))
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id"))
    role: Mapped[str] = mapped_column(String(32), default="canonical")


class Asset(TimestampMixin, Base):
    __tablename__ = "assets"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    kind: Mapped[str] = mapped_column(String(32))  # generated | dataset | reference
    mime_type: Mapped[str] = mapped_column(String(64))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    byte_size: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    storage_backend: Mapped[str] = mapped_column(String(16), default="postgres")
    storage_key: Mapped[str | None] = mapped_column(String(512))
    provenance: Mapped[dict] = mapped_column(JSON, default=dict)


class AssetBlob(Base):
    __tablename__ = "asset_blobs"
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"), primary_key=True)
    data: Mapped[bytes] = mapped_column(LargeBinary)


class Dataset(TimestampMixin, Base):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200))
    group: Mapped[str] = mapped_column(String(32))  # identity|body|expression|pose|outfit|appearance|style|scene
    character_id: Mapped[str | None] = mapped_column(ForeignKey("characters.id"))
    source: Mapped[str | None] = mapped_column(Text)
    license: Mapped[str | None] = mapped_column(String(200))


class DatasetItem(TimestampMixin, Base):
    __tablename__ = "dataset_items"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id"))
    caption: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|approved|rejected
    split: Mapped[str] = mapped_column(String(8), default="train")
    synthetic: Mapped[bool] = mapped_column(default=False)
    source_asset_id: Mapped[str | None] = mapped_column(String(36))  # ảnh AI nguồn nếu synthetic
    filename: Mapped[str | None] = mapped_column(String(300))
    phash: Mapped[str | None] = mapped_column(String(16), index=True)  # dHash 64-bit hex
    flags: Mapped[list] = mapped_column(JSON, default=list)  # low_resolution, near_duplicate:<item_id>, ...


class DatasetVersion(TimestampMixin, Base):
    """Snapshot bất biến: không có thao tác UPDATE nào trên bảng này hay dataset_version_items."""
    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("dataset_id", "version"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    version: Mapped[int] = mapped_column(Integer)
    manifest_hash: Mapped[str] = mapped_column(String(64))
    item_count: Mapped[int] = mapped_column(Integer)
    split_counts: Mapped[dict] = mapped_column(JSON, default=dict)
    warnings: Mapped[list] = mapped_column(JSON, default=list)


class DatasetVersionItem(Base):
    __tablename__ = "dataset_version_items"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    version_id: Mapped[str] = mapped_column(ForeignKey("dataset_versions.id"), index=True)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id"))
    sha256: Mapped[str] = mapped_column(String(64))
    caption: Mapped[str] = mapped_column(Text)
    split: Mapped[str] = mapped_column(String(8))
    synthetic: Mapped[bool] = mapped_column(default=False)


class GenerationJob(TimestampMixin, Base):
    __tablename__ = "generation_jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), unique=True)
    request_hash: Mapped[str | None] = mapped_column(String(64))
    character_version_id: Mapped[str | None] = mapped_column(ForeignKey("character_versions.id"))
    prompt_original: Mapped[str] = mapped_column(Text)
    prompt_compiled: Mapped[str | None] = mapped_column(Text)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    state: Mapped[str] = mapped_column(String(24), default="queued")
    error: Mapped[str | None] = mapped_column(Text)


class GenerationOutput(Base):
    __tablename__ = "generation_outputs"
    __table_args__ = (UniqueConstraint("job_id", "output_index"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("generation_jobs.id"))
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id"))
    output_index: Mapped[int] = mapped_column(Integer)
    seed: Mapped[int | None] = mapped_column(Integer)


class ModelVersion(TimestampMixin, Base):
    """Registry: base checkpoint / Character LoRA / Style LoRA. Chỉ seed base thật sự tích hợp."""
    __tablename__ = "model_versions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    kind: Mapped[str] = mapped_column(String(24))  # base | character_lora | style_lora
    name: Mapped[str] = mapped_column(String(200))
    family: Mapped[str] = mapped_column(String(32), default="sdxl")
    model_id: Mapped[str | None] = mapped_column(String(300))   # HF repo id cho base
    revision: Mapped[str | None] = mapped_column(String(64))
    license: Mapped[str | None] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(24))  # verified|experimental|unavailable | awaiting_review|approved|rejected
    trainable: Mapped[bool] = mapped_column(default=False)
    artifact_path: Mapped[str | None] = mapped_column(String(600))
    sha256: Mapped[str | None] = mapped_column(String(64))
    base_model_version_id: Mapped[str | None] = mapped_column(ForeignKey("model_versions.id"))
    training_run_id: Mapped[str | None] = mapped_column(String(36))
    character_id: Mapped[str | None] = mapped_column(ForeignKey("characters.id"))
    trigger_token: Mapped[str | None] = mapped_column(String(64))
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    review_note: Mapped[str | None] = mapped_column(Text)


class TrainingRun(TimestampMixin, Base):
    __tablename__ = "training_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    dataset_version_id: Mapped[str] = mapped_column(ForeignKey("dataset_versions.id"))
    base_model_version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"))
    objective: Mapped[str] = mapped_column(String(16), default="character")
    trigger_token: Mapped[str] = mapped_column(String(64))
    config: Mapped[dict] = mapped_column(JSON, default=dict)   # cấu hình đã resolve
    state: Mapped[str] = mapped_column(String(24), default="queued")
    step: Mapped[int] = mapped_column(Integer, default=0)
    max_steps: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    phase: Mapped[str | None] = mapped_column(String(64))
    cancel_requested: Mapped[bool] = mapped_column(default=False)
    trainer_revision: Mapped[str | None] = mapped_column(String(100))
    base_revision: Mapped[str | None] = mapped_column(String(64))
    manifest_hash: Mapped[str | None] = mapped_column(String(64))
    work_dir: Mapped[str | None] = mapped_column(String(600))
    output_model_version_id: Mapped[str | None] = mapped_column(String(36))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    validation_asset_ids: Mapped[list] = mapped_column(JSON, default=list)


class TrainingMetric(Base):
    __tablename__ = "training_metrics"
    __table_args__ = (UniqueConstraint("run_id", "step"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("training_runs.id"), index=True)
    step: Mapped[int] = mapped_column(Integer)
    loss: Mapped[float] = mapped_column()
    lr: Mapped[float | None] = mapped_column()
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class VideoJob(TimestampMixin, Base):
    """Job ảnh -> video (ComfyUI + LTX-2.3 + PromptRelay). Mỗi lần render là một VideoAttempt; job giữ lịch sử sửa."""
    __tablename__ = "video_jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), unique=True)
    request_hash: Mapped[str | None] = mapped_column(String(64))
    source_keyframe_asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id"))
    source_generation_job_id: Mapped[str | None] = mapped_column(String(36))  # job ảnh nguồn (nếu có)
    character_version_id: Mapped[str | None] = mapped_column(ForeignKey("character_versions.id"))
    scene_version: Mapped[str | None] = mapped_column(String(64))
    global_prompt: Mapped[str] = mapped_column(Text)
    motion_segments: Mapped[list] = mapped_column(JSON, default=list)   # [{"start":0,"end":25,"prompt":"..."}], đơn vị % tương đối
    continuity_constraints: Mapped[list] = mapped_column(JSON, default=list)
    config: Mapped[dict] = mapped_column(JSON, default=dict)            # width,height,fps,frame_count,seed,audio_mode,...
    workflow_version: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str | None] = mapped_column(String(200))
    audio_mode: Mapped[str] = mapped_column(String(16), default="generated")  # generated | none
    budget_limit: Mapped[float] = mapped_column(default=0.0)            # 0 = không giới hạn
    cost_spent: Mapped[float] = mapped_column(default=0.0)
    repair_count: Mapped[int] = mapped_column(Integer, default=0)       # số lượt sửa tự động đã dùng (tối đa 2)
    state: Mapped[str] = mapped_column(String(24), default="queued")
    error: Mapped[str | None] = mapped_column(Text)
    output_asset_id: Mapped[str | None] = mapped_column(String(36))     # video của attempt mới nhất đạt render
    review_reason: Mapped[str | None] = mapped_column(String(32))
    review_note: Mapped[str | None] = mapped_column(Text)


class VideoAttempt(TimestampMixin, Base):
    __tablename__ = "video_attempts"
    __table_args__ = (UniqueConstraint("job_id", "attempt_no"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("video_jobs.id"), index=True)
    attempt_no: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(16), default="initial")    # initial | repair
    params: Mapped[dict] = mapped_column(JSON, default=dict)            # tham số đã resolve (seed, prompt, ...)
    state: Mapped[str] = mapped_column(String(24), default="queued")    # queued|running|rendered|failed
    prompt_id: Mapped[str | None] = mapped_column(String(64))           # id ComfyUI, ghi TRƯỚC khi chờ (đối chiếu khi timeout)
    output_asset_id: Mapped[str | None] = mapped_column(String(36))
    error: Mapped[str | None] = mapped_column(Text)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)            # VRAM/RAM đỉnh (ComfyUI /system_stats), bước, batch
    gpu_seconds: Mapped[float | None] = mapped_column()
    queue_seconds: Mapped[float | None] = mapped_column()
    total_seconds: Mapped[float | None] = mapped_column()
    cost: Mapped[float] = mapped_column(default=0.0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    review: Mapped[dict] = mapped_column(JSON, default=dict)            # decision, reason, note, scores
