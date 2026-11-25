from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = Path(os.getenv("SLIDESNAP_STORAGE_DIR", str(BASE_DIR / "storage")))
DB_PATH = Path(os.getenv("SLIDESNAP_DB_PATH", str(STORAGE_DIR / "slidesnap.db")))


@dataclass(frozen=True)
class QualityPreset:
    name: str
    label: str
    target_width: int
    target_height: int
    base_fps: float
    mid_fps: float
    long_fps: float
    max_segment_buffer: int
    jpeg_quality: int
    youtube_max_height: int


QUALITY_PRESETS: Dict[str, QualityPreset] = {
    "fast": QualityPreset(
        name="fast",
        label="Turbo Draft (480p)",
        target_width=854,
        target_height=480,
        base_fps=1.2,
        mid_fps=0.75,
        long_fps=0.4,
        max_segment_buffer=10,
        jpeg_quality=86,
        youtube_max_height=480,
    ),
    "balanced": QualityPreset(
        name="balanced",
        label="Balanced HD (720p)",
        target_width=1280,
        target_height=720,
        base_fps=2.0,
        mid_fps=1.0,
        long_fps=0.5,
        max_segment_buffer=16,
        jpeg_quality=93,
        youtube_max_height=720,
    ),
    "high": QualityPreset(
        name="high",
        label="Ultra Crisp (1080p)",
        target_width=1920,
        target_height=1080,
        base_fps=2.5,
        mid_fps=1.25,
        long_fps=0.65,
        max_segment_buffer=20,
        jpeg_quality=96,
        youtube_max_height=1080,
    ),
}


def _default_workers() -> int:
    # Render Free Tier provides 512 MB RAM; 1 worker prevents concurrent video OOM spikes
    if os.getenv("RENDER"):
        return int(os.getenv("MAX_CONCURRENT_WORKERS", "1"))
    return int(os.getenv("MAX_CONCURRENT_WORKERS", "2"))


@dataclass(frozen=True)
class Settings:
    app_name: str = "SlideSnap CV Worker & API"
    version: str = "2.0.0"
    storage_dir: Path = STORAGE_DIR
    uploads_dir: Path = field(default_factory=lambda: STORAGE_DIR / "uploads")
    outputs_dir: Path = field(default_factory=lambda: STORAGE_DIR / "outputs")
    scratch_dir: Path = field(default_factory=lambda: STORAGE_DIR / "scratch")
    db_path: Path = DB_PATH

    # Rate Limiting & Resource Controls ($0 Free-Tier Guardrails)
    max_jobs_per_hour_per_ip: int = int(os.getenv("MAX_JOBS_PER_HOUR", "20"))
    max_concurrent_workers: int = _default_workers()
    max_video_duration_sec: int = int(os.getenv("MAX_VIDEO_DURATION_SEC", "14400"))  # 4 hours
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
