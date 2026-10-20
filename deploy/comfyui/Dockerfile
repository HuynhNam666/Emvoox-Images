# ComfyUI cho video (LTX-2.3 + PromptRelay). Cần NVIDIA driver trên máy chủ + NVIDIA Container Toolkit.
# Các commit dưới đây là đúng bộ đã được đối chiếu schema bằng backend/scripts/validate_comfy_workflow.py (20/10/2026).
# Đổi commit => chạy lại script validate trước khi dùng cho sản xuất.
FROM python:3.12-slim

ARG COMFY_COMMIT=b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c
ARG PROMPTRELAY_COMMIT=ca5d4e3edb6abd9c2a4c68a3a6798eec1980f450
ARG KJNODES_COMMIT=f20bdd5337ace70abbec147bdb68ea345d340836
ARG VHS_COMMIT=4d907bee61e92c2e65af3bd6383a4e4d356126d1
# cu128 hỗ trợ GPU Ampere → Blackwell. Dùng cu126 nếu driver cũ (< 570).
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone https://github.com/comfyanonymous/ComfyUI.git . && git checkout ${COMFY_COMMIT}
RUN pip install torch torchvision torchaudio --index-url ${TORCH_INDEX} \
    && pip install -r requirements.txt
RUN cd custom_nodes \
    && git clone https://github.com/kijai/ComfyUI-PromptRelay.git && (cd ComfyUI-PromptRelay && git checkout ${PROMPTRELAY_COMMIT}) \
    && git clone https://github.com/kijai/ComfyUI-KJNodes.git && (cd ComfyUI-KJNodes && git checkout ${KJNODES_COMMIT}) \
    && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && (cd ComfyUI-VideoHelperSuite && git checkout ${VHS_COMMIT}) \
    && for d in ComfyUI-PromptRelay ComfyUI-KJNodes ComfyUI-VideoHelperSuite; do \
         [ -f $d/requirements.txt ] && pip install -r $d/requirements.txt || true; done

COPY extra_model_paths.yaml /app/extra_model_paths.yaml
EXPOSE 8188
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 CMD curl -fs http://127.0.0.1:8188/system_stats || exit 1
# Không có xác thực: CHỈ chạy trong mạng nội bộ của compose, không publish cổng ra ngoài.
CMD ["sh", "-c", "python main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch --extra-model-paths-config /app/extra_model_paths.yaml --output-directory /data/comfy/output --temp-directory /data/comfy/temp --input-directory /data/comfy/input ${COMFY_ARGS}"]
