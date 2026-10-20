"""Bảng cho pipeline ảnh theo báo cáo EMVOOX1 (mục 3): Model Registry, Scene/Costume Library, VisualJob,
Quality Gate, Export (nhãn AI + C2PA), Benchmark chấm mù."""
from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base
from app.models.tables import TimestampMixin, _uuid


class SystemStatus(Base):
    """Khóa/giá trị nhỏ do worker ghi (VRAM, GPU, phiên bản...) để API biết năng lực worker thật."""
    __tablename__ = "system_status"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ProviderModel(TimestampMixin, Base):
    """Model Registry (mục 1.3): mỗi dòng là một model/tuyến tạo ảnh, kèm năng lực, giấy phép, giá, ngày kiểm tra."""
    __tablename__ = "provider_models"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    vendor: Mapped[str] = mapped_column(String(100))
    family: Mapped[str] = mapped_column(String(32))            # flux | gemini | seedream | gpt-image | qwen | z-image | sdxl | midjourney
    route: Mapped[str] = mapped_column(String(16))             # local | api | manual
    capabilities: Mapped[dict] = mapped_column(JSON, default=dict)   # txt2img, edit, multi_ref, lora, training, text_render, pose_control
    license: Mapped[str | None] = mapped_column(String(200))
    commercial_ok: Mapped[bool | None] = mapped_column(nullable=True)
    pricing: Mapped[dict] = mapped_column(JSON, default=dict)        # giá công khai tham khảo, đơn vị trong "unit"
    min_vram_gb: Mapped[float | None] = mapped_column()
    tasks: Mapped[list] = mapped_column(JSON, default=list)          # nhóm tác vụ được đề xuất (mục 1.3)
    verification: Mapped[str] = mapped_column(String(24), default="unverified")  # verified | connector_unverified | unverified | manual
    checked_at: Mapped[str | None] = mapped_column(String(16))
    notes: Mapped[str | None] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(default=True)
    benchmark: Mapped[dict] = mapped_column(JSON, default=dict)


class Scene(TimestampMixin, Base):
    __tablename__ = "scenes"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    code: Mapped[str] = mapped_column(String(32), unique=True)       # vd B02
    name: Mapped[str] = mapped_column(String(200))
    current_version_id: Mapped[str | None] = mapped_column(String(36))


class SceneVersion(TimestampMixin, Base):
    __tablename__ = "scene_versions"
    __table_args__ = (UniqueConstraint("scene_id", "version"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    scene_id: Mapped[str] = mapped_column(ForeignKey("scenes.id"))
    version: Mapped[int] = mapped_column(Integer)
    spec: Mapped[dict] = mapped_column(JSON, default=dict)           # layout, cửa/đồ vật, thời điểm, hướng sáng, bảng màu, đạo cụ
    reference_asset_ids: Mapped[list] = mapped_column(JSON, default=list)


class Costume(TimestampMixin, Base):
    __tablename__ = "costumes"
    __table_args__ = (UniqueConstraint("character_id", "code"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    character_id: Mapped[str] = mapped_column(ForeignKey("characters.id"))
    code: Mapped[str] = mapped_column(String(32))
    description_en: Mapped[str] = mapped_column(Text)
    reference_asset_id: Mapped[str | None] = mapped_column(String(36))


class VisualJob(TimestampMixin, Base):
    """Job ảnh theo hợp đồng POST /visual/jobs (mục 3.3). Một job có nhiều attempt (tạo, sửa) và nhiều candidate."""
    __tablename__ = "visual_jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), unique=True)
    request_hash: Mapped[str | None] = mapped_column(String(64))
    project_id: Mapped[str] = mapped_column(String(64), index=True)
    episode_id: Mapped[str | None] = mapped_column(String(64))
    shot_id: Mapped[str] = mapped_column(String(64), index=True)
    character_version_ids: Mapped[list] = mapped_column(JSON, default=list)
    scene_version_id: Mapped[str | None] = mapped_column(String(36))
    costume_ids: Mapped[list] = mapped_column(JSON, default=list)
    scene_spec: Mapped[dict] = mapped_column(JSON, default=dict)
    reference_asset_ids: Mapped[list] = mapped_column(JSON, default=list)   # đã resolve + ghim phiên bản
    aspect_ratio: Mapped[str] = mapped_column(String(8), default="9:16")
    num_variants: Mapped[int] = mapped_column(Integer, default=2)
    seed: Mapped[int] = mapped_column(Integer, default=42)
    pinned_model_key: Mapped[str | None] = mapped_column(String(64))
    route_plan: Mapped[list] = mapped_column(JSON, default=list)            # [{model_key, reason, est_cost}]
    prompt: Mapped[str | None] = mapped_column(Text)
    negative_prompt: Mapped[str | None] = mapped_column(Text)
    budget_limit: Mapped[float] = mapped_column(default=0.0)                 # USD; 0 = không giới hạn
    cost_spent: Mapped[float] = mapped_column(default=0.0)
    auto_repair: Mapped[bool] = mapped_column(default=True)
    repair_count: Mapped[int] = mapped_column(Integer, default=0)
    benchmark_run_id: Mapped[str | None] = mapped_column(String(36), index=True)
    benchmark_scene_id: Mapped[str | None] = mapped_column(String(32))
    state: Mapped[str] = mapped_column(String(24), default="queued")        # queued|running|review_required|approved|rejected|failed|needs_human
    error: Mapped[str | None] = mapped_column(Text)
    selected_candidate_id: Mapped[str | None] = mapped_column(String(36))
    reviewer: Mapped[str | None] = mapped_column(String(100))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    review_note: Mapped[str | None] = mapped_column(Text)
    review_history: Mapped[list] = mapped_column(JSON, default=list)


class VisualAttempt(TimestampMixin, Base):
    __tablename__ = "visual_attempts"
    __table_args__ = (UniqueConstraint("job_id", "attempt_no"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("visual_jobs.id"), index=True)
    attempt_no: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(16), default="initial")        # initial | repair | human_repair
    model_key: Mapped[str] = mapped_column(String(64), index=True)
    route: Mapped[str] = mapped_column(String(16))                           # local | api
    prompt: Mapped[str | None] = mapped_column(Text)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    remote_id: Mapped[str | None] = mapped_column(String(200))               # id job phía nhà cung cấp (đối chiếu khi timeout)
    state: Mapped[str] = mapped_column(String(16), default="running")        # running | done | failed
    error: Mapped[str | None] = mapped_column(Text)
    cost_api: Mapped[float] = mapped_column(default=0.0)
    cost_gpu: Mapped[float] = mapped_column(default=0.0)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)                # wall_s, queue_s, infer_s, gpu_peak_mb, ram_peak_mb...
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    instruction: Mapped[str | None] = mapped_column(Text)                    # chỉ dẫn sửa (repair)


class VisualCandidate(TimestampMixin, Base):
    __tablename__ = "visual_candidates"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("visual_jobs.id"), index=True)
    attempt_id: Mapped[str] = mapped_column(ForeignKey("visual_attempts.id"), index=True)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id"))
    seed: Mapped[int | None] = mapped_column(Integer)
    qc: Mapped[dict] = mapped_column(JSON, default=dict)
    qc_verdict: Mapped[str] = mapped_column(String(16), default="pending")   # pass | review | fail | pending
    qc_score: Mapped[float | None] = mapped_column()


class ExportRecord(TimestampMixin, Base):
    """Bản xuất cuối (mục 3.6): nhãn hiển thị, ký C2PA, xác minh, hồ sơ nguồn gốc nội bộ."""
    __tablename__ = "export_records"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_id: Mapped[str | None] = mapped_column(String(36), index=True)       # visual job hoặc video job
    kind: Mapped[str] = mapped_column(String(8), default="image")            # image | video
    source_asset_id: Mapped[str] = mapped_column(String(36))
    export_asset_id: Mapped[str | None] = mapped_column(String(36))
    target: Mapped[str] = mapped_column(String(16))                          # 1080x1920 | 1920x1080 | native
    label: Mapped[dict] = mapped_column(JSON, default=dict)
    c2pa: Mapped[dict] = mapped_column(JSON, default=dict)                   # {"signed":bool,"verified":bool,"error":...}
    invisible_watermark: Mapped[str] = mapped_column(String(64), default="none")
    status: Mapped[str] = mapped_column(String(16), default="released")      # released | blocked
    provenance: Mapped[dict] = mapped_column(JSON, default=dict)
    sha256: Mapped[str | None] = mapped_column(String(64))


class BenchmarkRun(TimestampMixin, Base):
    __tablename__ = "benchmark_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200))
    round_no: Mapped[int] = mapped_column(Integer, default=1)               # 1 = prompt nền, 2 = tham chiếu/LoRA
    model_keys: Mapped[list] = mapped_column(JSON, default=list)
    scene_ids: Mapped[list] = mapped_column(JSON, default=list)
    variants: Mapped[int] = mapped_column(Integer, default=4)
    status: Mapped[str] = mapped_column(String(16), default="created")


class BenchmarkRating(TimestampMixin, Base):
    """Chấm mù: hai người độc lập, ẩn tên công cụ. Trọng số 40/30/20/10 (mục 1.2)."""
    __tablename__ = "benchmark_ratings"
    __table_args__ = (UniqueConstraint("candidate_id", "rater"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    candidate_id: Mapped[str] = mapped_column(String(36), index=True)
    rater: Mapped[str] = mapped_column(String(64))
    consistency: Mapped[int] = mapped_column(Integer)
    scene_fit: Mapped[int] = mapped_column(Integer)
    anatomy: Mapped[int] = mapped_column(Integer)
    aesthetics: Mapped[int] = mapped_column(Integer)
    core_error: Mapped[bool] = mapped_column(default=False)                 # sai nhân vật/hành động cốt lõi = lỗi loại
    weighted: Mapped[float] = mapped_column()
