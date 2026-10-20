"""Danh mục trọng số cần tải cho server (kích thước lấy từ Hugging Face API ngày 20/10/2026, làm tròn).

Nhóm:
  video  — LTX-2.3 distilled fp8 + VAE video/audio + Gemma 3 12B fp8 + text projection (cho ComfyUI)        ≈ 42 GB
  sdxl   — SDXL 1.0 fp16 + VAE fp16-fix + IP-Adapter plus-face + CLIP ViT-H                                 ≈ 12 GB
  klein  — FLUX.2 Klein 4B (distilled) + Klein Base 4B (để tinh chỉnh)                                       ≈ 32 GB
  zimage — Z-Image-Turbo                                                                                    ≈ 33 GB
  qwen   — Qwen-Image + Qwen-Image-Edit-2511 (rất nặng; cần GPU ≥ 24GB hoặc offload CPU rất chậm)         ≈ 115 GB
  face   — FaceNet VGGFace2 (Quality Gate đo nhận dạng)                                                     ≈ 0,1 GB
Model API (FLUX 3/FLUX.2 Pro, Gemini, Seedream, GPT Image) không có trọng số để tải: chỉ cần khóa API trong .env.
Midjourney không có API: tạo ngoài hệ thống rồi nhập ảnh.
"""

LTX = "Kijai/LTX2.3_comfy"
GEMMA = "Comfy-Org/ltx-2"

# type=comfy_file: tải 1 file về /data/models/comfyui/<dest>; type=hf_snapshot: vào cache Hugging Face (cùng bố cục mà worker đọc)
GROUPS: dict[str, dict] = {
    "video": {"size_gb": 42.4, "items": [
        {"type": "comfy_file", "repo": LTX, "file": "diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors",
         "dest": "diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors", "size_gb": 25.23},
        {"type": "comfy_file", "repo": LTX, "file": "vae/LTX23_video_vae_bf16.safetensors", "dest": "vae/LTX23_video_vae_bf16.safetensors", "size_gb": 1.45},
        {"type": "comfy_file", "repo": LTX, "file": "vae/LTX23_audio_vae_bf16.safetensors", "dest": "vae/LTX23_audio_vae_bf16.safetensors", "size_gb": 0.36},
        {"type": "comfy_file", "repo": LTX, "file": "text_encoders/ltx-2.3_text_projection_bf16.safetensors",
         "dest": "text_encoders/ltx-2.3_text_projection_bf16.safetensors", "size_gb": 2.31},
        {"type": "comfy_file", "repo": GEMMA, "file": "split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors",
         "dest": "text_encoders/gemma_3_12B_it_fp8_scaled.safetensors", "size_gb": 13.21},
    ]},
    "sdxl": {"size_gb": 12.0, "items": [
        {"type": "hf_snapshot", "repo": "stabilityai/stable-diffusion-xl-base-1.0", "size_gb": 7.0, "allow": [
            "model_index.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*", "unet/config.json", "vae/config.json",
            "text_encoder/config.json", "text_encoder_2/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/diffusion_pytorch_model.fp16.safetensors", "text_encoder/model.fp16.safetensors", "text_encoder_2/model.fp16.safetensors"]},
        {"type": "hf_snapshot", "repo": "madebyollin/sdxl-vae-fp16-fix", "size_gb": 0.34, "allow": ["config.json", "diffusion_pytorch_model.safetensors"]},
        {"type": "hf_snapshot", "repo": "h94/IP-Adapter", "size_gb": 3.3, "allow": [
            "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors", "models/image_encoder/*"]},
    ]},
    "klein": {"size_gb": 32.0, "items": [
        {"type": "hf_snapshot", "repo": "black-forest-labs/FLUX.2-klein-4B", "size_gb": 16.0,
         "allow": ["model_index.json", "transformer/*", "text_encoder/*", "vae/*", "tokenizer/*", "scheduler/*"]},
        {"type": "hf_snapshot", "repo": "black-forest-labs/FLUX.2-klein-base-4B", "size_gb": 16.0,
         "allow": ["model_index.json", "transformer/*", "text_encoder/*", "vae/*", "tokenizer/*", "scheduler/*"]},
    ]},
    "zimage": {"size_gb": 33.0, "items": [
        {"type": "hf_snapshot", "repo": "Tongyi-MAI/Z-Image-Turbo", "size_gb": 33.0,
         "allow": ["model_index.json", "transformer/*", "text_encoder/*", "vae/*", "tokenizer/*", "scheduler/*"]},
    ]},
    "qwen": {"size_gb": 115.4, "items": [
        {"type": "hf_snapshot", "repo": "Qwen/Qwen-Image", "size_gb": 57.7,
         "allow": ["model_index.json", "transformer/*", "text_encoder/*", "vae/*", "tokenizer/*", "scheduler/*"]},
        {"type": "hf_snapshot", "repo": "Qwen/Qwen-Image-Edit-2511", "size_gb": 57.7,
         "allow": ["model_index.json", "transformer/*", "text_encoder/*", "vae/*", "tokenizer/*", "scheduler/*", "processor/*"]},
    ]},
    "face": {"size_gb": 0.11, "items": [{"type": "facenet", "size_gb": 0.11}]},
}
CORE = ["video", "sdxl", "klein", "face"]          # đủ cho pilot A01/A02: ảnh (SDXL + Klein) + video LTX
ALL = list(GROUPS)
