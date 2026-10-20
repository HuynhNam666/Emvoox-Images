FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-gpu.txt ./
RUN pip install --no-cache-dir -r requirements-gpu.txt
# Trainer script pinned to the same diffusers tag as the library.
RUN mkdir -p /opt/trainer && curl -fsSL \
  https://raw.githubusercontent.com/huggingface/diffusers/v0.30.3/examples/dreambooth/train_dreambooth_lora_sdxl.py \
  -o /opt/trainer/train_dreambooth_lora_sdxl.py
COPY . .
ENV PYTHONUNBUFFERED=1 TRAINER_SCRIPT=/opt/trainer/train_dreambooth_lora_sdxl.py
