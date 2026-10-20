"""Tạo VisualJob: resolve tài sản có phiên bản (Character/Scene/Costume Library), chọn tuyến, ghi job + attempt đầu tiên."""
import hashlib
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import (
    Asset, Character, CharacterReference, CharacterVersion, Costume, ModelVersion, ProviderModel, Scene, SceneVersion,
    SystemStatus, VisualAttempt, VisualJob,
)
from app.visual import router as R
from app.visual.compiler import Context, compile_prompt, ref_plan

MAX_REFS = {"sdxl-1.0": 4, "flux2-klein-4b": 4, "flux2-klein-base-4b": 4, "qwen-image-edit": 3, "flux2-pro": 8,
            "flux2-klein-4b-api": 4, "flux3-image": 8, "gemini-nano-banana": 6, "seedream-4-5": 6, "gpt-image": 6}
TASKS = Literal["concept", "action_simple", "action_face", "closeup", "multi_person", "text_poster", "series_bulk"]


class ServiceError(Exception):
    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status, self.msg = status, msg


class CharacterIn(BaseModel):
    character_version_id: str
    costume_code: str | None = None
    position: str | None = Field(None, max_length=60)      # vd "left", "right foreground"


class CameraIn(BaseModel):
    size: str = Field("", max_length=80)                    # vd "medium shot", "close-up"
    angle: str = Field("", max_length=80)                   # vd "eye level"


class SceneSpecIn(BaseModel):
    action: str = Field(min_length=1, max_length=800)
    emotion: str = Field("", max_length=200)
    camera: CameraIn = CameraIn()
    lighting: str = Field("", max_length=300)
    style: str = Field("", max_length=300)
    keep: list[str] = Field(default_factory=list, max_length=10)
    text_in_image: str = Field("", max_length=200)
    task: TASKS | None = None


class VisualRequest(BaseModel):
    project_id: str = Field(min_length=1, max_length=64)
    episode_id: str | None = Field(None, max_length=64)
    shot_id: str = Field(min_length=1, max_length=64)
    characters: list[CharacterIn] = Field(default_factory=list, max_length=4)
    scene_code: str | None = None
    scene_version: int | None = None
    scene_spec: SceneSpecIn
    reference_ids: list[str] = Field(default_factory=list, max_length=4)
    aspect_ratio: Literal["9:16", "16:9", "1:1"] = "9:16"
    num_variants: int = Field(settings.visual_default_variants, ge=1, le=4)
    seed: int = Field(42, ge=0, lt=2**31 - 8)
    model_key: str | None = None
    budget_limit: float = Field(0.0, ge=0)
    auto_repair: bool = True


def worker_status(db: Session) -> dict | None:
    row = db.get(SystemStatus, "worker")
    w = dict(row.value) if row and row.value else None
    if settings.worker_vram_gb and (w is None or not w.get("total_gb")):
        w = {**(w or {}), "cuda": True if w is None else w.get("cuda", True), "total_gb": settings.worker_vram_gb}
    return w


def resolve_context(db: Session, req: VisualRequest, strip_refs: bool = False) -> tuple[Context, list[str], dict]:
    """Ghim phiên bản + gom tham chiếu. Trả (ctx, character_ids, pins)."""
    ctx = Context()
    char_ids: list[str] = []
    costume_ids: list[str] = []
    seen_chars = set()
    for ci in req.characters:
        cv = db.get(CharacterVersion, ci.character_version_id)
        if not cv:
            raise ServiceError(422, f"character_version_id không tồn tại: {ci.character_version_id}")
        ch = db.get(Character, cv.character_id)
        if ch.id in seen_chars:
            raise ServiceError(422, f"nhân vật {ch.code} xuất hiện hai lần")
        seen_chars.add(ch.id)
        char_ids.append(ch.id)
        ident = cv.identity_spec or {}
        if not ident.get("identity_en"):
            raise ServiceError(422, f"nhân vật {ch.code} v{cv.version} chưa có đoạn nhận dạng (identity_en)")
        refs = [r.asset_id for r in db.scalars(select(CharacterReference).where(CharacterReference.character_version_id == cv.id))]
        entry = {"code": ch.code, "name": ch.name, "version": cv.version, "version_id": cv.id, "identity_en": ident["identity_en"],
                 "negative": ident.get("negative_prompt", ""), "position": ci.position or "", "costume_en": "",
                 "refs": [{"asset_id": a, "role": "identity" if i == 0 else "body", "label":
                           f"{'face' if i == 0 else 'full-body'} reference of {ch.code}"} for i, a in enumerate(refs)]}
        if ci.costume_code:
            co = db.scalar(select(Costume).where(Costume.character_id == ch.id, Costume.code == ci.costume_code))
            if not co:
                raise ServiceError(422, f"trang phục {ci.costume_code} chưa có cho {ch.code}")
            costume_ids.append(co.id)
            entry["costume_en"] = co.description_en
            if co.reference_asset_id:
                entry["costume_ref"] = {"asset_id": co.reference_asset_id, "role": "costume",
                                        "label": f"costume reference ({co.code}) worn by {ch.code}"}
        ctx.characters.append(entry)

    scene_version_id = None
    if req.scene_code:
        sc = db.scalar(select(Scene).where(Scene.code == req.scene_code))
        if not sc:
            raise ServiceError(422, f"scene {req.scene_code} không tồn tại")
        sv = (db.scalar(select(SceneVersion).where(SceneVersion.scene_id == sc.id, SceneVersion.version == req.scene_version))
              if req.scene_version else db.get(SceneVersion, sc.current_version_id))
        if not sv:
            raise ServiceError(422, f"scene {req.scene_code} v{req.scene_version} không tồn tại")
        scene_version_id = sv.id
        sp = sv.spec or {}
        bits = [f"Setting: {sc.name}", sp.get("layout"), sp.get("time_of_day"), (f"light from {sp['light_direction']}" if sp.get("light_direction") else None),
                (f"palette {sp['palette']}" if sp.get("palette") else None), (f"props: {', '.join(sp['props'])}" if sp.get("props") else None),
                (f"door/furniture layout: {sp['openings']}" if sp.get("openings") else None)]
        ctx.scene = {"code": sc.code, "name": sc.name, "version": sv.version, "version_id": sv.id,
                     "text": ", ".join(b for b in bits if b),
                     "refs": [{"asset_id": a, "role": "scene", "label": f"background reference of location {sc.code}"}
                              for a in (sv.reference_asset_ids or [])]}
    sp = req.scene_spec
    ctx.action, ctx.emotion, ctx.camera = sp.action, sp.emotion, sp.camera.model_dump()
    ctx.lighting, ctx.style, ctx.keep, ctx.text_in_image = sp.lighting, sp.style, list(sp.keep), sp.text_in_image
    for aid in req.reference_ids:
        if not db.get(Asset, aid):
            raise ServiceError(422, f"reference_ids không tồn tại: {aid}")
        ctx.extra_refs.append({"asset_id": aid, "role": "extra", "label": "additional reference"})
    if strip_refs:   # benchmark vòng 1: chỉ dùng prompt (mục 1.2), không tham chiếu
        for c in ctx.characters:
            c["refs"] = []
            c.pop("costume_ref", None)
        if ctx.scene:
            ctx.scene["refs"] = []
        ctx.extra_refs = []
    all_ids = [r["asset_id"] for c in ctx.characters for r in c["refs"]]
    missing_refs = [c["code"] for c in ctx.characters if not c["refs"]]  # chưa có ảnh chuẩn: router chỉ dùng được mô tả/LoRA
    pins = {"character_version_ids": [c["version_id"] for c in ctx.characters], "costume_ids": costume_ids,
            "scene_version_id": scene_version_id, "reference_asset_ids": list(dict.fromkeys(
                all_ids + [r["asset_id"] for r in ctx.extra_refs])), "characters_without_refs": missing_refs}
    return ctx, char_ids, pins


def lora_for(db: Session, char_ids: list[str]) -> ModelVersion | None:
    if len(char_ids) != 1:
        return None
    return db.scalar(select(ModelVersion).where(ModelVersion.kind == "character_lora", ModelVersion.status == "approved",
                                                ModelVersion.character_id == char_ids[0]).order_by(ModelVersion.created_at.desc()))


def create_visual_job(db: Session, req: VisualRequest, idem: str | None, benchmark: dict | None = None,
                      strip_refs: bool = False) -> VisualJob:
    payload = req.model_dump_json()
    req_hash = hashlib.sha256(payload.encode()).hexdigest()
    if idem:
        old = db.scalar(select(VisualJob).where(VisualJob.idempotency_key == idem))
        if old:
            if old.request_hash != req_hash:
                raise ServiceError(409, "Idempotency-Key đã dùng với payload khác")
            return old
    ctx, char_ids, pins = resolve_context(db, req, strip_refs)
    sp = req.scene_spec
    close = "close" in (sp.camera.size or "").lower()
    task = R.classify_task(sp.task, len(ctx.characters), close, bool(sp.text_in_image))
    from app.visual.providers.local import size_for
    w, h = size_for("flux", req.aspect_ratio)
    has_refs = bool(pins["reference_asset_ids"])
    need = R.Need(task=task, identity=bool(ctx.characters) and has_refs, n_characters=len(ctx.characters), text=bool(sp.text_in_image),
                  width=w, height=h, n_variants=req.num_variants, n_refs=len(pins["reference_asset_ids"]))
    lora = lora_for(db, char_ids)
    try:
        plan, rejected = R.plan_route(db, need, worker_status(db), lora_ok=lora is not None, pinned=req.model_key)
    except ValueError as e:
        raise ServiceError(422, str(e))
    if not plan:
        raise ServiceError(503, "không có tuyến tạo ảnh khả dụng: " + "; ".join(f"{r['model_key']}: {r['reason']}" for r in rejected[:8]))
    if req.budget_limit and plan[0]["est_cost"] > req.budget_limit:
        cheaper = [p for p in plan if p["est_cost"] <= req.budget_limit]
        if not cheaper:
            raise ServiceError(422, f"ngân sách {req.budget_limit} thấp hơn ước tính tuyến rẻ nhất ({min(p['est_cost'] for p in plan)})")
        plan = cheaper + [p for p in plan if p not in cheaper]
    primary = db.get(ProviderModel, plan[0]["model_key"])
    refs = ref_plan(ctx, MAX_REFS.get(primary.key, 4)) if (primary.capabilities.get("ref_identity") or primary.capabilities.get("multi_ref")) else []
    prompt, neg = compile_prompt(ctx, primary.capabilities, refs)
    spec = req.scene_spec.model_dump()
    spec["_resolved"] = ctx.to_dict()
    spec["_task"] = task
    job = VisualJob(
        idempotency_key=idem, request_hash=req_hash, project_id=req.project_id, episode_id=req.episode_id, shot_id=req.shot_id,
        character_version_ids=pins["character_version_ids"], scene_version_id=pins["scene_version_id"], costume_ids=pins["costume_ids"],
        scene_spec=spec, reference_asset_ids=pins["reference_asset_ids"], aspect_ratio=req.aspect_ratio, num_variants=req.num_variants,
        seed=req.seed, pinned_model_key=req.model_key, route_plan=plan, prompt=prompt, negative_prompt=neg,
        budget_limit=req.budget_limit, auto_repair=req.auto_repair,
        benchmark_run_id=(benchmark or {}).get("run_id"), benchmark_scene_id=(benchmark or {}).get("scene_id"))
    db.add(job)
    db.flush()
    db.add(VisualAttempt(job_id=job.id, attempt_no=1, kind="initial", model_key=primary.key, route=primary.route, state="queued"))
    db.commit()
    job._rejected = rejected  # chỉ để trả về trong phản hồi tạo job
    return job


def continuity_warnings(db: Session, job: VisualJob) -> list[str]:
    """Consistency Engine — theo dõi liên tục (mục 3.4): so với shot đã duyệt gần nhất cùng bối cảnh trong cùng tập.
    Chỉ cảnh báo, không tự sửa; không bảo đảm nhất quán tuyệt đối."""
    if not job.scene_version_id:
        return []
    prev = db.scalar(select(VisualJob).where(
        VisualJob.project_id == job.project_id, VisualJob.episode_id == job.episode_id, VisualJob.state == "approved",
        VisualJob.scene_version_id == job.scene_version_id, VisualJob.id != job.id).order_by(VisualJob.reviewed_at.desc()))
    if not prev:
        return []
    warn = []
    a, b = job.scene_spec.get("_resolved", {}), prev.scene_spec.get("_resolved", {})
    if (a.get("lighting") or "").strip().lower() != (b.get("lighting") or "").strip().lower():
        warn.append(f"ánh sáng khác shot {prev.shot_id} cùng bối cảnh: '{b.get('lighting')}' → '{a.get('lighting')}'")
    prev_costume = {c["code"]: c.get("costume_en", "") for c in b.get("characters", [])}
    for c in a.get("characters", []):
        if c["code"] in prev_costume and prev_costume[c["code"]] != c.get("costume_en", ""):
            warn.append(f"trang phục {c['code']} khác shot {prev.shot_id} trong cùng bối cảnh/tập")
    return warn
