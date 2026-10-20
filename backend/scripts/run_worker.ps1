# Chạy worker GPU và tự khởi động lại khi tiến trình thoát (vd sau lỗi CUDA nặng). Dùng cho chạy trực tiếp trên máy.
# Chạy từ thư mục backend:  powershell -File scripts/run_worker.ps1
$env:PYTHONPATH = "."; $env:PYTHONUTF8 = "1"; $env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
if (-not $env:REDIS_URL) { $env:REDIS_URL = "redis://127.0.0.1:56379/0" }
while ($true) {
    ..\.venv\Scripts\python.exe -m celery -A app.workers.celery_app worker -Q generation,training,video,visual -P solo -c 1 --loglevel=info
    Write-Host "Worker thoát (mã $LASTEXITCODE). Khởi động lại sau 3 giây… (Ctrl+C để dừng)"
    Start-Sleep 3
}
