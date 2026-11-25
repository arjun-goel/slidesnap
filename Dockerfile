# Hugging Face Spaces & Render Free Tier (512 MB RAM Safe) Dockerfile
FROM python:3.11-slim

# Install FFmpeg and OpenCV headless system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Prevent glibc arena bloat and 64-core OpenCV/OMP thread-pool RAM spikes on cloud hosts
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    SLIDESNAP_STORAGE_DIR=/home/user/app/storage \
    PYTHONUNBUFFERED=1 \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    OPENCV_NUM_THREADS=1

WORKDIR /home/user/app

COPY --chown=user ./backend/requirements.txt /home/user/app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /home/user/app/requirements.txt

COPY --chown=user ./backend/app /home/user/app/app

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--limit-concurrency", "20"]
