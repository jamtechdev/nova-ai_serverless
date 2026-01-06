FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install all dependencies
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

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
