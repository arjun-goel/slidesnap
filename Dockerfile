# Hugging Face Spaces (Free Tier: 2 vCPU, 16 GB RAM) & Render/Fly.io Ready Dockerfile
FROM python:3.11-slim

# Install FFmpeg and OpenCV headless system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    SLIDESNAP_STORAGE_DIR=/home/user/app/storage

WORKDIR /home/user/app

COPY --chown=user ./backend/requirements.txt /home/user/app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /home/user/app/requirements.txt

COPY --chown=user ./backend/app /home/user/app/app

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
