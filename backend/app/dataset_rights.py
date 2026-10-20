"""Kiểm tra quyền sử dụng dataset trước khi huấn luyện (báo cáo mục 3.1: "loại ảnh không rõ quyền sử dụng").

Phân loại theo văn bản license/nguồn người dùng ghi ở cấp dataset:
  ok           — ảnh AI tự sinh (mọi ảnh synthetic) hoặc license cho phép thương mại rõ ràng (CC0, MIT, Apache, CC BY không NC, Unsplash Lite)
  noncommercial— phi thương mại / chỉ nghiên cứu / cần ký thỏa thuận
  unknown      — không ghi hoặc không nhận ra được
COMMERCIAL_USE=true: noncommercial và unknown đều CHẶN huấn luyện; COMMERCIAL_USE=false: chỉ cảnh báo.
Đây là kiểm tra chuỗi văn bản, không thay cho việc đọc điều khoản gốc và tư vấn pháp lý.
"""
import re

NC = re.compile(r"non[- ]?commercial|\bcc[- ]by[- ]nc|\bby-nc|phi thương mại|chỉ nghiên cứu|research only|research use|agreement|thỏa thuận|"
                r"academic", re.I)
OK = re.compile(r"\bcc0\b|\bmit\b|apache|\bbsd\b|cc[- ]by(?![- ]nc)[- ]?\d|unsplash dataset lite|public domain|tự sinh|ai tự sinh|"
                r"synthetic", re.I)


def rights_status(license_text: str | None, source_text: str | None, n_items: int, n_synthetic: int) -> tuple[str, str]:
    """Trả (trạng thái, giải thích)."""
    lic = (license_text or "").strip()
    if n_items and n_synthetic == n_items:
        return "ok", "toàn bộ ảnh là ảnh AI tự sinh trong hệ thống (quyền theo giấy phép của model sinh ảnh)"
    if lic and NC.search(lic):
        return "noncommercial", f"license ghi phi thương mại/nghiên cứu: '{lic[:80]}'"
    if lic and OK.search(lic):
        return "ok", f"license cho phép thương mại theo ghi chú: '{lic[:80]}' (vẫn nên đọc điều khoản gốc)"
    if not lic:
        return "unknown", "dataset chưa ghi license/quyền sử dụng"
    return "unknown", f"không nhận ra license '{lic[:80]}'"
