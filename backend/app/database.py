from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .config import settings
from .models import JobResponse, JobStatus, SlideMetadataResponse, SourceType

_db_lock = threading.Lock()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(settings.db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _db_lock:
        conn = get_connection()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    client_fingerprint TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_url TEXT,
                    upload_path TEXT,
                    video_title TEXT NOT NULL,
                    video_duration_sec INTEGER NOT NULL DEFAULT 0,
                    rights_attested INTEGER NOT NULL DEFAULT 1,
                    enable_temporal_inpainting INTEGER NOT NULL DEFAULT 1,
                    enable_ocr_indexing INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'queued',
                    progress_pct INTEGER NOT NULL DEFAULT 0,
                    current_stage_detail TEXT NOT NULL DEFAULT 'Queued for CV worker...',
                    total_candidate_frames INTEGER NOT NULL DEFAULT 0,
                    extracted_slide_count INTEGER NOT NULL DEFAULT 0,
                    reconstructed_slide_count INTEGER NOT NULL DEFAULT 0,
                    pdf_filename TEXT,
                    pdf_file_size_bytes INTEGER,
                    error_code TEXT,
                    error_message TEXT,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    expires_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS job_slides (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    slide_number INTEGER NOT NULL,
                    timestamp_start_sec REAL NOT NULL,
                    timestamp_end_sec REAL NOT NULL,
                    selected_frame_sec REAL NOT NULL,
                    sharpness_score REAL NOT NULL,
                    person_occlusion_ratio REAL NOT NULL,
                    final_quality_score REAL NOT NULL,
                    perceptual_hash TEXT NOT NULL,
                    was_reconstructed INTEGER NOT NULL DEFAULT 0,
                    reconstruction_method TEXT NOT NULL,
                    ocr_text TEXT NOT NULL DEFAULT '',
                    image_filename TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_client ON jobs(client_fingerprint, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_slides_job ON job_slides(job_id, slide_number ASC);
                """
            )
            conn.commit()
        finally:
            conn.close()


# Ensure tables exist even when imported directly in unit tests
init_db()


def count_recent_jobs_for_client(client_fingerprint: str, hours: int = 1) -> int:
    cutoff = _iso(_utc_now() - timedelta(hours=hours))
    with _db_lock:
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM jobs WHERE client_fingerprint = ? AND created_at >= ?",
                (client_fingerprint, cutoff),
            ).fetchone()
            return int(row["cnt"]) if row else 0
        finally:
            conn.close()


def create_job_record(
    *,
    client_fingerprint: str,
    source_type: SourceType,
    source_url: Optional[str],
    upload_path: Optional[str],
    video_title: str,
    rights_attested: bool,
    enable_temporal_inpainting: bool,
    enable_ocr_indexing: bool,
) -> str:
    job_id = str(uuid.uuid4())
    now = _utc_now()
    expires = now + timedelta(hours=settings.retention_hours)
    with _db_lock:
        conn = get_connection()
        try:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, client_fingerprint, source_type, source_url, upload_path,
                    video_title, rights_attested, enable_temporal_inpainting,
                    enable_ocr_indexing, status, progress_pct, current_stage_detail,
                    created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    client_fingerprint,
                    source_type.value,
                    source_url,
                    upload_path,
                    video_title,
                    1 if rights_attested else 0,
                    1 if enable_temporal_inpainting else 0,
                    1 if enable_ocr_indexing else 0,
                    JobStatus.QUEUED.value,
                    0,
                    "Queued for CV worker...",
                    _iso(now),
                    _iso(expires),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    return job_id


def update_job_progress(
    job_id: str,
    *,
    status: Optional[JobStatus] = None,
    progress_pct: Optional[int] = None,
    stage_detail: Optional[str] = None,
    video_title: Optional[str] = None,
    video_duration_sec: Optional[int] = None,
    total_candidate_frames: Optional[int] = None,
    extracted_slide_count: Optional[int] = None,
    reconstructed_slide_count: Optional[int] = None,
    pdf_filename: Optional[str] = None,
    pdf_file_size_bytes: Optional[int] = None,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    fields: List[str] = []
    values: List[Any] = []
    now_str = _iso(_utc_now())

    if status is not None:
        fields.append("status = ?")
        values.append(status.value)
        if status == JobStatus.INGESTING:
            fields.append("started_at = COALESCE(started_at, ?)")
            values.append(now_str)
        elif status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            fields.append("completed_at = ?")
            values.append(now_str)

    if progress_pct is not None:
        fields.append("progress_pct = ?")
        values.append(max(0, min(100, int(progress_pct))))
    if stage_detail is not None:
        fields.append("current_stage_detail = ?")
        values.append(stage_detail)
    if video_title is not None:
        fields.append("video_title = ?")
        values.append(video_title)
    if video_duration_sec is not None:
        fields.append("video_duration_sec = ?")
        values.append(int(video_duration_sec))
    if total_candidate_frames is not None:
        fields.append("total_candidate_frames = ?")
        values.append(int(total_candidate_frames))
    if extracted_slide_count is not None:
        fields.append("extracted_slide_count = ?")
        values.append(int(extracted_slide_count))
    if reconstructed_slide_count is not None:
        fields.append("reconstructed_slide_count = ?")
        values.append(int(reconstructed_slide_count))
    if pdf_filename is not None:
        fields.append("pdf_filename = ?")
        values.append(pdf_filename)
    if pdf_file_size_bytes is not None:
        fields.append("pdf_file_size_bytes = ?")
        values.append(int(pdf_file_size_bytes))
    if error_code is not None:
        fields.append("error_code = ?")
        values.append(error_code)
    if error_message is not None:
        fields.append("error_message = ?")
        values.append(error_message)

    if not fields:
        return

    values.append(job_id)
    with _db_lock:
        conn = get_connection()
        try:
            conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", values)
            conn.commit()
        finally:
            conn.close()


def insert_job_slide(
    *,
    job_id: str,
    slide_number: int,
    timestamp_start_sec: float,
    timestamp_end_sec: float,
    selected_frame_sec: float,
    sharpness_score: float,
    person_occlusion_ratio: float,
    final_quality_score: float,
    perceptual_hash: str,
    was_reconstructed: bool,
    reconstruction_method: str,
    ocr_text: str,
    image_filename: str,
) -> str:
    slide_id = str(uuid.uuid4())
    with _db_lock:
        conn = get_connection()
        try:
            conn.execute(
                """
                INSERT INTO job_slides (
                    id, job_id, slide_number, timestamp_start_sec, timestamp_end_sec,
                    selected_frame_sec, sharpness_score, person_occlusion_ratio,
                    final_quality_score, perceptual_hash, was_reconstructed,
                    reconstruction_method, ocr_text, image_filename, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    slide_id,
                    job_id,
                    slide_number,
                    round(timestamp_start_sec, 2),
                    round(timestamp_end_sec, 2),
                    round(selected_frame_sec, 2),
                    round(sharpness_score, 2),
                    round(person_occlusion_ratio, 4),
                    round(final_quality_score, 4),
                    perceptual_hash,
                    1 if was_reconstructed else 0,
                    reconstruction_method,
                    ocr_text,
                    image_filename,
                    _iso(_utc_now()),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    return slide_id


def clear_job_slides(job_id: str) -> None:
    with _db_lock:
        conn = get_connection()
        try:
            conn.execute("DELETE FROM job_slides WHERE job_id = ?", (job_id,))
            conn.commit()
        finally:
            conn.close()


def is_cancel_requested(job_id: str) -> bool:
    with _db_lock:
        conn = get_connection()
        try:
            row = conn.execute("SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return bool(row and row["cancel_requested"])
        finally:
            conn.close()


def request_job_cancellation(job_id: str) -> bool:
    with _db_lock:
        conn = get_connection()
        try:
            row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if not row:
                return False
            current_status = row["status"]
            if current_status in (JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value):
                return False
            conn.execute(
                "UPDATE jobs SET cancel_requested = 1, current_stage_detail = 'Cancellation requested...' WHERE id = ?",
                (job_id,),
            )
            conn.commit()
            return True
        finally:
            conn.close()


def reset_job_for_retry(job_id: str) -> bool:
    with _db_lock:
        conn = get_connection()
        try:
            row = conn.execute("SELECT status, retry_count FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if not row:
                return False
            if int(row["retry_count"]) >= settings.max_retries:
                return False
            conn.execute(
                """
                UPDATE jobs
                SET status = 'queued',
                    progress_pct = 0,
                    current_stage_detail = 'Re-queued for retry...',
                    error_code = NULL,
                    error_message = NULL,
                    cancel_requested = 0,
                    retry_count = retry_count + 1
                WHERE id = ?
                """,
                (job_id,),
            )
            conn.execute("DELETE FROM job_slides WHERE job_id = ?", (job_id,))
            conn.commit()
            return True
        finally:
            conn.close()


def get_raw_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _db_lock:
        conn = get_connection()
        try:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def _row_to_job_response(row: sqlite3.Row, slides: List[sqlite3.Row]) -> JobResponse:
    job_id = row["id"]
    pdf_url = f"/api/jobs/{job_id}/download" if row["pdf_filename"] and row["status"] == JobStatus.COMPLETED.value else None
    slide_Items = [
        SlideMetadataResponse(
            id=s["id"],
            job_id=job_id,
            slide_number=s["slide_number"],
            timestamp_start_sec=s["timestamp_start_sec"],
            timestamp_end_sec=s["timestamp_end_sec"],
            selected_frame_sec=s["selected_frame_sec"],
            sharpness_score=s["sharpness_score"],
            person_occlusion_ratio=s["person_occlusion_ratio"],
            final_quality_score=s["final_quality_score"],
            perceptual_hash=s["perceptual_hash"],
            was_reconstructed=bool(s["was_reconstructed"]),
            reconstruction_method=s["reconstruction_method"],
            ocr_text=s["ocr_text"],
            image_url=f"/api/jobs/{job_id}/slides/{s['image_filename']}",
        )
        for s in slides
    ]
    return JobResponse(
        id=job_id,
        source_type=SourceType(row["source_type"]),
        source_url=row["source_url"],
        video_title=row["video_title"],
        video_duration_sec=row["video_duration_sec"],
        rights_attested=bool(row["rights_attested"]),
        enable_temporal_inpainting=bool(row["enable_temporal_inpainting"]),
        enable_ocr_indexing=bool(row["enable_ocr_indexing"]),
        status=JobStatus(row["status"]),
        progress_pct=row["progress_pct"],
        current_stage_detail=row["current_stage_detail"],
        total_candidate_frames=row["total_candidate_frames"],
        extracted_slide_count=row["extracted_slide_count"],
        reconstructed_slide_count=row["reconstructed_slide_count"],
        pdf_download_url=pdf_url,
        pdf_file_size_bytes=row["pdf_file_size_bytes"],
        error_code=row["error_code"],
        error_message=row["error_message"],
        retry_count=row["retry_count"],
        cancel_requested=bool(row["cancel_requested"]),
        created_at=row["created_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        expires_at=row["expires_at"],
        slides=slide_Items,
    )


def get_job_response(job_id: str) -> Optional[JobResponse]:
    with _db_lock:
        conn = get_connection()
        try:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if not row:
                return None
            slides = conn.execute(
                "SELECT * FROM job_slides WHERE job_id = ? ORDER BY slide_number ASC",
                (job_id,),
            ).fetchall()
            return _row_to_job_response(row, slides)
        finally:
            conn.close()


def list_jobs(limit: int = 25) -> List[JobResponse]:
    with _db_lock:
        conn = get_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            result: List[JobResponse] = []
            for row in rows:
                slides = conn.execute(
                    "SELECT * FROM job_slides WHERE job_id = ? ORDER BY slide_number ASC",
                    (row["id"],),
                ).fetchall()
                result.append(_row_to_job_response(row, slides))
            return result
        finally:
            conn.close()
