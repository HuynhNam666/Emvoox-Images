from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    database_url: str = "postgresql+psycopg://emvoox:change-me@localhost:5432/emvoox_images"
    redis_url: str = "redis://localhost:6379/0"
    asset_storage_backend: str = "postgres"
    dataset_dir: str = "./data/datasets"
    model_cache_dir: str = "./data/models"
    sdxl_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    sdxl_cpu_offload: bool = False
    sdxl_scheduler: str = "dpmpp_2m_karras"   # "default" = scheduler mặc định của pipeline (cấu hình đã dùng ở các thử nghiệm trước 20/10/2026)
    max_upload_bytes: int = 20 * 1024 * 1024
    max_zip_entries: int = 500
    max_zip_uncompressed_bytes: int = 500 * 1024 * 1024
    max_image_pixels: int = 50_000_000
    min_image_side: int = 512
    near_dup_hamming: int = 4
    # --- Video (ComfyUI chạy như dịch vụ riêng; EMVOOX chỉ gọi API) ---
    comfyui_url: str = "http://127.0.0.1:8188"
    comfyui_timeout_s: int = 1800          # tổng thời gian chờ một lần render
    comfyui_poll_s: float = 2.0
    video_workflow_dir: str = ""           # rỗng = thư mục app/ml/workflows
    video_workflow_name: str = "ltx23_promptrelay"
    video_cost_per_gpu_second: float = 0.0  # 0 = chi phí tính theo giây GPU (đơn vị tự chọn khi đặt budget_limit)
    video_max_repairs: int = 2
    video_max_upload_bytes: int = 200 * 1024 * 1024
    # --- Server ---
    api_token: str = ""                   # rỗng = không bắt đăng nhập (chỉ dùng localhost); server PHẢI đặt
    cookie_secure: bool = False           # true khi chạy sau HTTPS
    commercial_use: bool = False          # true: chặn dataset phi thương mại / chưa rõ license ở training
    worker_vram_gb: float = 0.0           # 0 = tự đọc từ worker; đặt tay nếu API chưa thấy worker
    # --- Visual pipeline ---
    visual_max_repairs: int = 2
    visual_default_variants: int = 2
    gpu_cost_per_hour: float = 0.0        # USD/giờ GPU (thuê hoặc khấu hao); 0 = chỉ báo giờ GPU, không quy ra tiền
    storage_cost_per_gb_month: float = 0.0
    qa_cost_per_review: float = 0.0       # công kiểm duyệt mỗi lượt (USD)
    qc_identity_pass: float = 0.60        # FaceNet cosine; ngưỡng tham khảo, cần hiệu chuẩn trên A01/A02
    qc_identity_fail: float = 0.45
    # --- API ảnh bên ngoài (để trống = chưa cấu hình) ---
    bfl_api_key: str = ""
    bfl_base_url: str = "https://api.bfl.ai/v1"
    bfl_flux3_endpoint: str = ""          # chưa xác minh endpoint "FLUX 3 Image": điền khi đã có
    gemini_api_key: str = ""
    gemini_image_model: str = "gemini-2.5-flash-image"   # đổi sang id Nano Banana 2.x khi đã xác minh
    seedream_api_key: str = ""
    seedream_base_url: str = "https://ark.ap-southeast.bytepluses.com/api/v3"
    seedream_model: str = "seedream-4-5-251128"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_image_model: str = "gpt-image-1"
    api_timeout_s: int = 180
    api_poll_s: float = 2.0
    api_max_retries: int = 2
    # --- Xuất bản / nhãn AI / C2PA ---
    export_dir: str = "./data/exports"
    ffmpeg_bin: str = ""                 # rỗng = tìm ffmpeg trong PATH
    require_label: bool = True            # true: không xuất bản nếu thiếu nhãn AI hiển thị
    benchmark_pass_score: float = 70.0    # điểm có trọng số (0–100) tối thiểu để một ảnh benchmark tính là đạt
    require_c2pa: bool = False            # true: không phát hành bản xuất nếu chưa ký + xác minh C2PA
    c2pa_cert_path: str = ""              # chuỗi chứng thư PEM
    c2pa_key_path: str = ""               # khóa riêng PEM
    c2pa_tsa_url: str = ""
    ai_label_text_generated: str = "Hình ảnh được tạo bằng AI"
    ai_label_text_edited: str = "Hình ảnh được chỉnh sửa bằng AI"
    ai_label_font: str = ""               # đường dẫn .ttf hỗ trợ tiếng Việt (mặc định tự dò DejaVu/Arial)


settings = Settings()
