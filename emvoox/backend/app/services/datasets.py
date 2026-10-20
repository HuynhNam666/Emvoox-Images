import hashlib
import io
import json
import posixpath
import stat
import uuid
import zipfile
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Dataset, DatasetItem, DatasetVersion, DatasetVersionItem
from app.services.assets import store_asset
from app.services.imaging import ImageRejected, hamming_hex, inspect_image

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class ImportReport:
    added: int = 0
    duplicates: int = 0
    rejected: list[dict] = field(default_factory=list)


def _add_item(db: Session, ds: Dataset, name: str, data: bytes, caption: str, report: ImportReport,
              source: str | None, license_: str | None, phashes: list[str]) -> None:
    s = get_settings()
    try:
        info = inspect_image(data)
    except ImageRejected as e:
        report.rejected.append({"file": name, "code": e.code, "message": str(e)})
        return
    if db.query(DatasetItem.id).filter_by(dataset_id=ds.id, sha256=info.sha256).first():
        report.duplicates += 1
        return
    flags = []
    if min(info.width, info.height) < s.min_image_side:
        flags.append("low_resolution")
    if any(hamming_hex(p, info.phash) <= s.near_duplicate_hamming for p in phashes):
        flags.append("near_duplicate")
    phashes.append(info.phash)
    asset = store_asset(db, project_id=ds.project_id, data=data, kind="dataset_image",
                        provenance={"source": source, "license": license_, "filename": name})
    db.add(DatasetItem(
        dataset_id=ds.id, asset_id=asset.id, caption=caption, sha256=info.sha256, phash=info.phash,
        source=source, license=license_, quality_flags=flags,
    ))
    report.added += 1


def import_files(db: Session, ds: Dataset, files: list[tuple[str, bytes]], source: str | None,
                 license_: str | None) -> ImportReport:
    report = ImportReport()
    captions: dict[str, str] = {}
    images: list[tuple[str, bytes]] = []
    for name, data in files:
        stem, ext = posixpath.splitext(posixpath.basename(name).lower())
        if ext in IMAGE_EXT:
            images.append((name, data))
        elif ext == ".txt":
            captions[stem] = data.decode("utf-8", errors="replace").strip()
        elif ext == ".zip":
            sub = _read_zip(data, report)
            for n, d in sub:
                s2, e2 = posixpath.splitext(posixpath.basename(n).lower())
                if e2 in IMAGE_EXT:
                    images.append((n, d))
                elif e2 == ".txt":
                    captions[s2] = d.decode("utf-8", errors="replace").strip()
        else:
            report.rejected.append({"file": name, "code": "unsupported_file", "message": "Chỉ nhận JPG/PNG/WebP, .txt, .zip."})
    phashes = [p for (p,) in db.query(DatasetItem.phash).filter_by(dataset_id=ds.id)]  # loaded once per import
    for name, data in images:
        stem = posixpath.splitext(posixpath.basename(name).lower())[0]
        _add_item(db, ds, name, data, captions.get(stem, ""), report, source, license_, phashes)
    return report


def _read_zip(data: bytes, report: ImportReport) -> list[tuple[str, bytes]]:
    """ZIP-slip, symlink, file-count and decompression-bomb safe extraction into memory."""
    s = get_settings()
    out: list[tuple[str, bytes]] = []
    total = 0
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        report.rejected.append({"file": "zip", "code": "bad_zip", "message": "ZIP hỏng."})
        return out
    infos = [i for i in zf.infolist() if not i.is_dir()]
    if len(infos) > s.max_zip_files:
        report.rejected.append({"file": "zip", "code": "too_many_files", "message": f"ZIP có {len(infos)} file, vượt giới hạn {s.max_zip_files} (đặt MAX_ZIP_FILES để đổi). Hãy chia nhỏ ZIP hoặc chọn bớt ảnh."})
        return out
    for info in infos:
        name = info.filename
        norm = posixpath.normpath(name.replace("\\", "/"))
        if norm.startswith("/") or norm.startswith("..") or "/../" in norm or (len(norm) > 1 and norm[1] == ":"):
            report.rejected.append({"file": name, "code": "zip_slip", "message": "Đường dẫn không an toàn."})
            continue
        if stat.S_ISLNK(info.external_attr >> 16):
            report.rejected.append({"file": name, "code": "symlink", "message": "Symlink không được phép."})
            continue
        total += info.file_size
        if total > s.max_zip_uncompressed_bytes or info.file_size > s.max_upload_bytes:
            report.rejected.append({"file": name, "code": "too_large", "message": "Vượt giới hạn giải nén."})
            continue
        with zf.open(info) as fh:  # read with a hard cap: header sizes can lie
            buf = fh.read(s.max_upload_bytes + 1)
        if len(buf) > s.max_upload_bytes:
            report.rejected.append({"file": name, "code": "too_large", "message": "File giải nén quá lớn."})
            continue
        out.append((norm, buf))
    return out


def _groups(items: list[DatasetItem], threshold: int) -> list[list[DatasetItem]]:
    parent = list(range(len(items)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if hamming_hex(items[i].phash, items[j].phash) <= threshold:
                parent[find(i)] = find(j)
    buckets: dict[int, list[DatasetItem]] = {}
    for i, it in enumerate(items):
        buckets.setdefault(find(i), []).append(it)
    return list(buckets.values())


class VersionError(Exception):
    pass


def create_version(db: Session, ds: Dataset, seed: int = 0) -> DatasetVersion:
    """Snapshot approved items; near-duplicate groups never straddle splits."""
    s = get_settings()
    items = db.query(DatasetItem).filter_by(dataset_id=ds.id, status="approved").order_by(DatasetItem.sha256).all()
    if not items:
        raise VersionError("Chưa có ảnh nào được duyệt.")
    warnings: list[str] = []
    assignment: dict[uuid.UUID, str] = {}
    groups = _groups(items, s.near_duplicate_hamming)
    for g in groups:
        explicit = {i.split for i in g if i.split}
        if len(explicit) > 1:
            raise VersionError("Nhóm ảnh gần trùng bị gán vào nhiều split khác nhau; hãy thống nhất hoặc loại bớt.")
        if explicit:
            for i in g:
                assignment[i.id] = next(iter(explicit))
    free = [g for g in groups if not any(i.id in assignment for i in g)]
    free.sort(key=lambda g: hashlib.sha256(f"{seed}:{g[0].sha256}".encode()).hexdigest())
    n = len(items)
    want_val = max(1, round(n * 0.1)) if n >= 10 else 0
    want_test = max(1, round(n * 0.1)) if n >= 20 else 0
    val = test = 0
    for g in free:
        if test < want_test:
            label, test = "test", test + len(g)
        elif val < want_val:
            label, val = "val", val + len(g)
        else:
            label = "train"
        for i in g:
            assignment[i.id] = label
    if want_val == 0:
        warnings.append("Dataset quá nhỏ (<10 ảnh): không có validation; không thể chọn checkpoint khách quan.")
    if want_test == 0:
        warnings.append("Dataset quá nhỏ (<20 ảnh): không có test split riêng; đánh giá cuối không độc lập.")
    if sum(1 for v in assignment.values() if v == "train") < 15:
        warnings.append("Dưới 15 ảnh train: dễ overfit, kết quả chỉ mang tính thăm dò.")
    last = db.query(DatasetVersion).filter_by(dataset_id=ds.id).order_by(DatasetVersion.version.desc()).first()
    manifest = [
        {"sha256": i.sha256, "caption": i.caption, "split": assignment[i.id]} for i in items
    ]
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    ver = DatasetVersion(
        dataset_id=ds.id, version=(last.version + 1 if last else 1), manifest_sha256=manifest_hash,
        item_count=n, warnings=warnings,
    )
    db.add(ver)
    db.flush()
    for i in items:
        db.add(DatasetVersionItem(
            dataset_version_id=ver.id, asset_id=i.asset_id, caption=i.caption, split=assignment[i.id], sha256=i.sha256,
        ))
    db.flush()
    return ver
