from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = Path(os.getenv("SLIDESNAP_STORAGE_DIR", str(BASE_DIR / "storage")))
DB_PATH = Path(os.getenv("SLIDESNAP_DB_PATH", str(STORAGE_DIR / "slidesnap.db")))


@dataclass(frozen=True)
class Settings:
    app_name: str = "SlideSnap CV Worker & API"
    version: str = "1.0.0"
    storage_dir: Path = STORAGE_DIR
    uploads_dir: Path = field(default_factory=lambda: STORAGE_DIR / "uploads")
    outputs_dir: Path = field(default_factory=lambda: STORAGE_DIR / "outputs")
    scratch_dir: Path = field(default_factory=lambda: STORAGE_DIR / "scratch")
    db_path: Path = DB_PATH

    # Rate Limiting & Resource Controls ($0 Free-Tier Guardrails)
    max_jobs_per_hour_per_ip: int = int(os.getenv("MAX_JOBS_PER_HOUR", "10"))
    max_concurrent_workers: int = int(os.getenv("MAX_CONCURRENT_WORKERS", "2"))
    max_video_duration_sec: int = int(os.getenv("MAX_VIDEO_DURATION_SEC", "7200"))  # 2 hours
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))
    retention_hours: int = int(os.getenv("RETENTION_HOURS", "24"))
    max_retries: int = 2

    # Computer Vision Hyperparameters
    sample_fps: float = float(os.getenv("SAMPLE_FPS", "2.0"))
    min_slide_duration_sec: float = float(os.getenv("MIN_SLIDE_DURATION_SEC", "1.5"))
    dhash_scene_threshold: int = int(os.getenv("DHASH_SCENE_THRESHOLD", "11"))
    global_dedup_dhash_threshold: int = int(os.getenv("GLOBAL_DEDUP_DHASH_THRESHOLD", "8"))
    clean_occlusion_threshold: float = float(os.getenv("CLEAN_OCCLUSION_THRESHOLD", "0.015"))  # 1.5%
    min_sharpness_laplacian: float = float(os.getenv("MIN_SHARPNESS_LAPLACIAN", "35.0"))

    # Security
    worker_shared_secret: str = os.getenv("WORKER_SHARED_SECRET", "dev-student-secret-key")


settings = Settings()

for directory in (
    settings.storage_dir,
    settings.uploads_dir,
    settings.outputs_dir,
    settings.scratch_dir,
):
    directory.mkdir(parents=True, exist_ok=True)
