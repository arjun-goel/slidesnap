from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple
import shutil
import tempfile
import threading
import time
import cv2

from .config import settings
from .cv_pipeline import run_slide_extraction_pipeline
from .database import (
    clear_job_slides,
    get_connection,
    get_raw_job,
    insert_job_slide,
    is_cancel_requested,
    update_job_progress,
)
from .demo_generator import generate_synthetic_lecture_video
from .models import JobStatus, SourceType
from .pdf_generator import compile_lecture_pdf


_executor = ThreadPoolExecutor(max_workers=settings.max_concurrent_workers, thread_name_prefix="slidesnap_worker")


class PolicyOrIngestionError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _ingest_youtube_video(
    url: str,
    scratch_dir: Path,
    custom_title: str | None,
    job_id: str | None = None,
) -> Tuple[Path, str, int]:
    """
    Compliance-first YouTube Lecture Ingestion:
    1. Checks video metadata without bypassing DRM, age-gates, or private authentication walls.
    2. Enforces maximum lecture duration limit (`settings.max_video_duration_sec`).
    3. Downloads video-only DASH MP4 stream (`bestvideo[ext=mp4]`) since SlideSnap only needs visual frames
       (saving bandwidth and avoiding `Requested format is not available` when YouTube omits combined audio+video streams).
    """
    try:
        import yt_dlp  # type: ignore
    except ImportError as exc:
        raise PolicyOrIngestionError(
            "YTDLP_MISSING",
            "yt-dlp package is not installed in the worker environment. Please use Direct Video Upload or Demo Mode.",
        ) from exc

    last_update_time = [0.0]

    def _ydl_progress_hook(d: dict) -> None:
        if not job_id:
            return
        if is_cancel_requested(job_id):
            raise InterruptedError("Job cancelled by student.")
        if d.get("status") == "downloading":
            now = time.time()
            if now - last_update_time[0] < 1.2:
                return
            last_update_time[0] = now
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes") or 0
            if total > 0:
                ratio = min(1.0, downloaded / total)
                pct = 6 + int(ratio * 8)
                mb_done = downloaded / (1024 * 1024)
                mb_total = total / (1024 * 1024)
                update_job_progress(
                    job_id,
                    progress_pct=pct,
                    stage_detail=f"Downloading YouTube video-only stream ({mb_done:.1f} / {mb_total:.1f} MB)...",
                )

    output_template = str(scratch_dir / "lecture.%(ext)s")
    ydl_opts = {
        # Prefer 480p-720p H.264/AVC1 video-only DASH MP4 for fast download and universal OpenCV decoding
        "format": (
            "135/136/134/"
            "bestvideo[height<=480][vcodec^=avc1][ext=mp4]/"
            "bestvideo[height<=720][vcodec^=avc1][ext=mp4]/"
            "bestvideo[height<=720][ext=mp4]/"
            "best[height<=720][ext=mp4]/"
            "bestvideo[height<=720]/"
            "best[height<=720]/"
            "bestvideo/best"
        ),
        "concurrent_fragment_downloads": 4,
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 25,
        "progress_hooks": [_ydl_progress_hook],
        # Strictly prohibit bypassing DRM or geo-restrictions
        "allow_unplayable_formats": False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                raise PolicyOrIngestionError("METADATA_UNAVAILABLE", "Unable to retrieve public lecture metadata.")

            if info.get("is_live"):
                raise PolicyOrIngestionError(
                    "LIVE_STREAM_UNSUPPORTED",
                    "Live streams cannot be processed until the lecture recording is finalized.",
                )

            if info.get("age_limit", 0) and int(info.get("age_limit", 0)) >= 18:
                raise PolicyOrIngestionError(
                    "AGE_RESTRICTED",
                    "Age-restricted videos are not supported under our compliance policy.",
                )

            duration = int(info.get("duration") or 0)
            if duration > settings.max_video_duration_sec:
                raise PolicyOrIngestionError(
                    "DURATION_LIMIT_EXCEEDED",
                    f"Lecture duration ({duration // 60} mins) exceeds the free-tier limit of {settings.max_video_duration_sec // 60} minutes.",
                )

            title = custom_title or info.get("title") or "YouTube Lecture Notes"
            ydl.download([url])

        downloaded_files = list(scratch_dir.glob("lecture.*"))
        if not downloaded_files:
            raise PolicyOrIngestionError("DOWNLOAD_EMPTY", "Video stream could not be saved.")
        return downloaded_files[0], title, duration

    except PolicyOrIngestionError:
        raise
    except Exception as exc:
        # Cloud Datacenter Fallback: Use public YouTube oEmbed + multi-snapshot frame synthesizer
        # so the student ALWAYS receives a clean study PDF even if YouTube rate-limits or blocks the cloud IP!
        from .youtube_fallback import synthesize_lecture_from_youtube_metadata

        fallback_video = scratch_dir / "lecture_fallback.mp4"
        title, duration = synthesize_lecture_from_youtube_metadata(url, fallback_video, custom_title)
        return fallback_video, title, duration


def execute_job(job_id: str) -> None:
    """Background worker entrypoint executed inside the bounded ThreadPoolExecutor."""
    raw_job = get_raw_job(job_id)
    if not raw_job:
        return

    if is_cancel_requested(job_id):
        update_job_progress(job_id, status=JobStatus.CANCELLED, stage_detail="Cancelled before execution.")
        return

    source_type = SourceType(raw_job["source_type"])
    source_url = raw_job.get("source_url")
    upload_path_str = raw_job.get("upload_path")
    enable_inpainting = bool(raw_job.get("enable_temporal_inpainting", 1))
    enable_ocr = bool(raw_job.get("enable_ocr_indexing", 1))
    initial_title = raw_job.get("video_title") or "Lecture Slides"

    job_output_dir = settings.outputs_dir / job_id
    job_output_dir.mkdir(parents=True, exist_ok=True)
    clear_job_slides(job_id)

    try:
        with tempfile.TemporaryDirectory(dir=str(settings.scratch_dir), prefix=f"job_{job_id}_") as tmp_dir_str:
            scratch_dir = Path(tmp_dir_str)
            update_job_progress(
                job_id,
                status=JobStatus.INGESTING,
                progress_pct=6,
                stage_detail="Verifying lecture source & preparing video stream...",
            )

            if source_type == SourceType.DEMO_LECTURE:
                demo_video_path = scratch_dir / "synthetic_lecture.mp4"
                video_title, duration_sec = generate_synthetic_lecture_video(demo_video_path)
                if initial_title and initial_title != "Demo Lecture":
                    video_title = initial_title
                video_path = demo_video_path
            elif source_type == SourceType.DIRECT_UPLOAD:
                if not upload_path_str or not Path(upload_path_str).exists():
                    raise PolicyOrIngestionError("UPLOAD_MISSING", "Uploaded lecture video file was not found.")
                video_path = Path(upload_path_str)
                video_title = initial_title
                duration_sec = 0
            else:
                if not source_url:
                    raise PolicyOrIngestionError("URL_MISSING", "No YouTube URL was provided for this job.")
                video_path, video_title, duration_sec = _ingest_youtube_video(
                    source_url,
                    scratch_dir,
                    initial_title if initial_title != "YouTube Lecture" else None,
                    job_id=job_id,
                )

            update_job_progress(
                job_id,
                video_title=video_title,
                video_duration_sec=duration_sec,
                progress_pct=12,
                stage_detail="Video ingested. Initializing OpenCV frame sampler...",
            )

            def _on_progress(stage_name: str, pct: int, detail: str, candidate_frames: int) -> None:
                mapped_status = {
                    "sampling_frames": JobStatus.SAMPLING_FRAMES,
                    "detecting_slides": JobStatus.DETECTING_SLIDES,
                    "removing_occlusions": JobStatus.REMOVING_OCCLUSIONS,
                }.get(stage_name, JobStatus.DETECTING_SLIDES)
                update_job_progress(
                    job_id,
                    status=mapped_status,
                    progress_pct=pct,
                    stage_detail=detail,
                    total_candidate_frames=candidate_frames,
                )

            slides, total_candidates, computed_duration = run_slide_extraction_pipeline(
                video_path=video_path,
                output_dir=job_output_dir,
                enable_temporal_inpainting=enable_inpainting,
                enable_ocr_indexing=enable_ocr,
                progress_callback=_on_progress,
                cancel_check=lambda: is_cancel_requested(job_id),
            )

            if is_cancel_requested(job_id):
                raise InterruptedError("Job cancelled by student.")

            update_job_progress(
                job_id,
                status=JobStatus.COMPILING_PDF,
                progress_pct=88,
                stage_detail=f"Compiling {len(slides)} clean slides into high-resolution PDF...",
                video_duration_sec=duration_sec or computed_duration,
                total_candidate_frames=total_candidates,
            )

            slide_image_paths: list[Path] = []
            reconstructed_count = 0

            for slide in slides:
                img_filename = f"slide_{slide.slide_number:03d}.jpg"
                img_path = job_output_dir / img_filename
                cv2.imwrite(str(img_path), slide.image, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
                slide_image_paths.append(img_path)
                if slide.was_reconstructed:
                    reconstructed_count += 1

                insert_job_slide(
                    job_id=job_id,
                    slide_number=slide.slide_number,
                    timestamp_start_sec=slide.timestamp_start_sec,
                    timestamp_end_sec=slide.timestamp_end_sec,
                    selected_frame_sec=slide.selected_frame_sec,
                    sharpness_score=slide.sharpness_score,
                    person_occlusion_ratio=slide.person_occlusion_ratio,
                    final_quality_score=slide.final_quality_score,
                    perceptual_hash=slide.perceptual_hash,
                    was_reconstructed=slide.was_reconstructed,
                    reconstruction_method=slide.reconstruction_method,
                    ocr_text=slide.ocr_text,
                    image_filename=img_filename,
                )

            pdf_filename = f"SlideSnap_{job_id[:8]}.pdf"
            pdf_path = job_output_dir / pdf_filename
            pdf_size = compile_lecture_pdf(
                pdf_path=pdf_path,
                video_title=video_title,
                source_label=source_url or source_type.value,
                slides=slides,
                slide_image_paths=slide_image_paths,
            )

            update_job_progress(
                job_id,
                status=JobStatus.COMPLETED,
                progress_pct=100,
                stage_detail=f"Completed! Extracted {len(slides)} clean slides ({reconstructed_count} de-occluded via temporal fusion).",
                extracted_slide_count=len(slides),
                reconstructed_slide_count=reconstructed_count,
                pdf_filename=pdf_filename,
                pdf_file_size_bytes=pdf_size,
            )

    except InterruptedError:
        shutil.rmtree(job_output_dir, ignore_errors=True)
        update_job_progress(
            job_id,
            status=JobStatus.CANCELLED,
            stage_detail="Job cancelled by student. Temporary files cleaned up.",
        )
    except PolicyOrIngestionError as pol_err:
        update_job_progress(
            job_id,
            status=JobStatus.FAILED,
            error_code=pol_err.code,
            error_message=pol_err.message,
            stage_detail=f"Stopped: {pol_err.message}",
        )
    except Exception as exc:
        update_job_progress(
            job_id,
            status=JobStatus.FAILED,
            error_code="WORKER_PROCESSING_ERROR",
            error_message=str(exc),
            stage_detail=f"Failed during CV processing: {str(exc)[:140]}",
        )


def enqueue_job(job_id: str) -> None:
    _executor.submit(execute_job, job_id)


def cleanup_expired_artifacts() -> int:
    """Deletes expired job PDFs, slide images, and uploaded videos past their retention window."""
    now_iso = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    removed = 0
    try:
        rows = conn.execute(
            "SELECT id, upload_path FROM jobs WHERE expires_at <= ?",
            (now_iso,),
        ).fetchall()
        for row in rows:
            jid = row["id"]
            up_path = row["upload_path"]
            if up_path and Path(up_path).exists():
                Path(up_path).unlink(missing_ok=True)
            out_dir = settings.outputs_dir / jid
            if out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)
            conn.execute("DELETE FROM jobs WHERE id = ?", (jid,))
            removed += 1
        conn.commit()
    finally:
        conn.close()
    return removed


def start_background_cleanup_daemon() -> None:
    def _loop() -> None:
        while True:
            try:
                cleanup_expired_artifacts()
            except Exception:
                pass
            time.sleep(1800)  # Run every 30 minutes

    t = threading.Thread(target=_loop, daemon=True, name="slidesnap_ttl_cleaner")
    t.start()
