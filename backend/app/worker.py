from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Tuple

from .config import QUALITY_PRESETS, settings
from .cv_pipeline import ExtractedSlide, run_slide_extraction_pipeline
from .database import (
    clear_job_slides,
    get_connection,
    get_raw_job,
    insert_job_slide,
    is_cancel_requested,
    update_job_progress,
)
from .demo_generator import (
    generate_progressive_board_lecture_video,
    generate_synthetic_lecture_video,
)
from .models import JobStatus, SourceType
from .pdf_generator import compile_lecture_pdf


_executor = ThreadPoolExecutor(max_workers=settings.max_concurrent_workers, thread_name_prefix="slidesnap_worker")


class PolicyOrIngestionError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _build_youtube_format_selector(quality_mode: str) -> str:
    preset = QUALITY_PRESETS.get(quality_mode, QUALITY_PRESETS["balanced"])
    max_h = preset.youtube_max_height
    # On Render 512 MB Free Tier, cap stream at 720p max to prevent huge MP4 mux buffers
    if os.getenv("RENDER") and max_h > 720:
        max_h = 720

    if max_h <= 480:
        return (
            "135/134/18/"
            "bestvideo[height<=480][vcodec^=avc1][ext=mp4]/"
            "bestvideo[height<=480][ext=mp4]/"
            "best[height<=480][ext=mp4]/"
            "bestvideo[height<=480]/best[height<=480]/best"
        )
    if max_h >= 1080:
        return (
            "137/136/135/18/"
            "bestvideo[height<=1080][vcodec^=avc1][ext=mp4]/"
            "bestvideo[height<=1080][ext=mp4]/"
            "bestvideo[height<=720][vcodec^=avc1][ext=mp4]/"
            "best[height<=1080][ext=mp4]/best"
        )
    return (
        "136/135/134/18/"
        "bestvideo[height<=720][vcodec^=avc1][ext=mp4]/"
        "bestvideo[height<=720][ext=mp4]/"
        "best[height<=720][ext=mp4]/"
        "bestvideo[height<=720]/best[height<=720]/best"
    )


def _ingest_youtube_video_isolated(
    url: str,
    scratch_dir: Path,
    custom_title: str | None,
    quality_mode: str = "balanced",
    job_id: str | None = None,
) -> Tuple[Path, str, int, bool]:
    """
    Runs `yt-dlp` inside an ISOLATED child process (`sys.executable -m yt_dlp`) so that
    100% of yt-dlp's ~110 MB Python extractor memory is freed back to the OS the instant
    the download finishes—preventing 512 MB Render OOM kills!
    """
    preset = QUALITY_PRESETS.get(quality_mode, QUALITY_PRESETS["balanced"])
    info_json_path = scratch_dir / "meta.info.json"
    output_template = str(scratch_dir / "lecture.%(ext)s")

    if job_id:
        update_job_progress(
            job_id,
            progress_pct=8,
            stage_detail=f"Fetching YouTube {preset.label} stream in isolated low-RAM worker...",
        )

    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--no-warnings",
        "--socket-timeout",
        "22",
        "--concurrent-fragments",
        "2",
        "--extractor-args",
        "youtube:player_client=android_vr,mweb,tv,web",
        "-f",
        _build_youtube_format_selector(quality_mode),
        "--print-to-file",
        "%(title)s\n%(duration)s\n%(is_live)s\n%(age_limit)s",
        str(info_json_path),
        "-o",
        output_template,
        url,
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
        )
        if is_cancel_requested(job_id or ""):
            raise InterruptedError("Job cancelled by student.")

        title = custom_title or "YouTube Lecture Notes"
        duration = 0
        if info_json_path.exists():
            lines = info_json_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if len(lines) >= 1 and not custom_title and lines[0].strip() and lines[0].strip() != "NA":
                title = lines[0].strip()
            if len(lines) >= 2 and lines[1].strip().isdigit():
                duration = int(lines[1].strip())
            if len(lines) >= 3 and lines[2].strip().lower() == "true":
                raise PolicyOrIngestionError(
                    "LIVE_STREAM_UNSUPPORTED",
                    "Live streams cannot be processed until the lecture recording is finalized.",
                )
            if len(lines) >= 4 and lines[3].strip().isdigit() and int(lines[3].strip()) >= 18:
                raise PolicyOrIngestionError(
                    "AGE_RESTRICTED",
                    "Age-restricted videos are not supported under our compliance policy.",
                )
            info_json_path.unlink(missing_ok=True)

        if duration > settings.max_video_duration_sec:
            raise PolicyOrIngestionError(
                "DURATION_LIMIT_EXCEEDED",
                f"Lecture duration ({duration // 60} mins) exceeds the free-tier limit of {settings.max_video_duration_sec // 60} minutes.",
            )

        downloaded_files = [
            p for p in scratch_dir.glob("lecture.*")
            if p.suffix.lower() in (".mp4", ".webm", ".mkv", ".m4v", ".mov") and p.stat().st_size > 1024
        ]
        if proc.returncode == 0 and downloaded_files:
            return downloaded_files[0], title, duration, False

    except PolicyOrIngestionError:
        raise
    except InterruptedError:
        raise
    except Exception:
        pass

    # Datacenter IP Fallback: synthesize clean slides from public oEmbed + snapshots
    from .youtube_fallback import synthesize_lecture_from_youtube_metadata

    fallback_video = scratch_dir / "lecture_fallback.mp4"
    title, duration = synthesize_lecture_from_youtube_metadata(url, fallback_video, custom_title)
    gc.collect()
    return fallback_video, title, duration, True


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
    quality_mode = str(raw_job.get("quality_mode") or "balanced")
    enhance_contrast = bool(raw_job.get("enhance_contrast", 0))
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
                stage_detail="Verifying lecture source & preparing low-memory stream...",
            )

            used_fallback = False
            if source_type == SourceType.DEMO_LECTURE:
                demo_video_path = scratch_dir / "synthetic_lecture.mp4"
                video_title, duration_sec = generate_synthetic_lecture_video(demo_video_path)
                if initial_title and initial_title not in ("Demo Lecture", "YouTube Lecture"):
                    video_title = initial_title
                video_path = demo_video_path
            elif source_type == SourceType.PROGRESSIVE_BOARD_DEMO:
                demo_video_path = scratch_dir / "progressive_chalkboard.mp4"
                video_title, duration_sec = generate_progressive_board_lecture_video(demo_video_path)
                if initial_title and initial_title not in ("Progressive Board Demo", "Demo Lecture", "YouTube Lecture"):
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
                video_path, video_title, duration_sec, used_fallback = _ingest_youtube_video_isolated(
                    source_url,
                    scratch_dir,
                    initial_title if initial_title != "YouTube Lecture" else None,
                    quality_mode=quality_mode,
                    job_id=job_id,
                )

            gc.collect()
            update_job_progress(
                job_id,
                video_title=video_title,
                video_duration_sec=duration_sec,
                used_youtube_fallback=used_fallback,
                progress_pct=12,
                stage_detail="Stream ready. Running Zero-Stack uint8 Pre-Transition CV Engine...",
            )

            live_reconstructed = [0]
            live_total = [0]

            def _on_slide_ready(slide: ExtractedSlide) -> None:
                live_total[0] += 1
                if slide.was_reconstructed:
                    live_reconstructed[0] += 1
                insert_job_slide(
                    job_id=job_id,
                    slide_number=slide.slide_number,
                    timestamp_start_sec=slide.timestamp_start_sec,
                    timestamp_end_sec=slide.timestamp_end_sec,
                    selected_frame_sec=slide.selected_frame_sec,
                    sharpness_score=slide.sharpness_score,
                    person_occlusion_ratio=slide.person_occlusion_ratio,
                    final_quality_score=slide.final_quality_score,
                    content_completeness_score=slide.content_completeness_score,
                    perceptual_hash=slide.perceptual_hash,
                    was_reconstructed=slide.was_reconstructed,
                    reconstruction_method=slide.reconstruction_method,
                    ocr_text=slide.ocr_text,
                    image_filename=slide.image_path.name,
                    occluded_image_filename=slide.occluded_image_path.name if slide.occluded_image_path else None,
                )
                update_job_progress(
                    job_id,
                    extracted_slide_count=live_total[0],
                    reconstructed_slide_count=live_reconstructed[0],
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
                quality_mode=quality_mode,
                enhance_contrast=enhance_contrast,
                enable_temporal_inpainting=enable_inpainting,
                enable_ocr_indexing=enable_ocr,
                progress_callback=_on_progress,
                on_slide_extracted=_on_slide_ready,
                cancel_check=lambda: is_cancel_requested(job_id),
            )

            # Delete scratch video BEFORE compiling PDF to free disk/OS page cache immediately
            try:
                if video_path.parent == scratch_dir and video_path.exists():
                    video_path.unlink(missing_ok=True)
            except Exception:
                pass

            if is_cancel_requested(job_id):
                raise InterruptedError("Job cancelled by student.")

            update_job_progress(
                job_id,
                status=JobStatus.COMPILING_PDF,
                progress_pct=90,
                stage_detail=f"Compiling {len(slides)} clean slides into Landscape A4 PDF...",
                video_duration_sec=duration_sec or computed_duration,
                total_candidate_frames=total_candidates,
            )

            slide_image_paths = [s.image_path for s in slides]
            reconstructed_count = sum(1 for s in slides if s.was_reconstructed)

            pdf_filename = f"SlideSnap_{job_id[:8]}.pdf"
            pdf_path = job_output_dir / pdf_filename
            pdf_size = compile_lecture_pdf(
                pdf_path=pdf_path,
                video_title=video_title,
                source_label=source_url or source_type.value,
                slides=slides,
                slide_image_paths=slide_image_paths,
                quality_mode=quality_mode,
            )

            update_job_progress(
                job_id,
                status=JobStatus.COMPLETED,
                progress_pct=100,
                stage_detail=f"Completed! Extracted {len(slides)} clean slides ({reconstructed_count} de-occluded).",
                extracted_slide_count=len(slides),
                reconstructed_slide_count=reconstructed_count,
                pdf_filename=pdf_filename,
                pdf_file_size_bytes=pdf_size,
            )
            del slides
            gc.collect()

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
            time.sleep(1800)

    t = threading.Thread(target=_loop, daemon=True, name="slidesnap_ttl_cleaner")
    t.start()
