FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    runpod \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    xformers \
    omegaconf \
    invisible-watermark \
    compel \
    cloudinary \
    pillow

# Download model at build time
RUN mkdir -p /app/models
RUN wget -O /app/models/civitai_new.safetensors \
    "https://civitai.com/api/download/models/2155386?type=Model&format=SafeTensor&size=pruned&fp=fp16&token=8da2037b00e9b0f247f4d408944d473e"

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
