# HuggingFace Spaces — Docker SDK
# Hardware target: T4 Small (16 GB VRAM)
#
# First cold start downloads models (~10-15 min):
#   google/medgemma-1.5-4b-it  ~8 GB
#   google/medsiglip-448        ~1.7 GB
#   LoRA adapter (optional)     ~100 MB
#
# Required Space secrets:
#   HF_TOKEN      — HuggingFace token with read access to gated models
#   LORA_ADAPTER  — e.g. AryanMarwah/medgemma-trauma-lora  (optional)

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Writable directories
RUN mkdir -p uploads

# HF Spaces routes external traffic to port 7860
EXPOSE 7860

# Point all caches to /tmp — the only writable path in HF Spaces containers
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface/transformers
ENV HF_HUB_CACHE=/tmp/huggingface/hub
ENV TORCH_HOME=/tmp/torch
ENV HF_HUB_DISABLE_PROGRESS_BARS=1

CMD ["python", "app.py"]
