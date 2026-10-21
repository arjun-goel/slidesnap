from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional
import shutil

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from .config import settings
from .database import (
    count_recent_jobs_for_client,
    create_job_record,
    get_job_response,
    get_raw_job,
    init_db,
    list_jobs,
    request_job_cancellation,
    reset_job_for_retry,
)
from .models import CreateJobRequest, JobResponse, SourceType
from .worker import enqueue_job, start_background_cleanup_daemon

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Asynchronous Computer Vision Lecture Slide Extractor & Presenter De-Occlusion API ($0 Student Cloud Tier)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_client_fingerprint(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "127.0.0.1")
    return hashlib.sha256(f"slidesnap:{ip}".encode("utf-8")).hexdigest()[:32]


def _enforce_rate_limit(request: Request) -> str:
    fp = _get_client_fingerprint(request)
    recent = count_recent_jobs_for_client(fp, hours=1)
    if recent >= settings.max_jobs_per_hour_per_ip:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit reached ({settings.max_jobs_per_hour_per_ip} lecture jobs per hour). Please wait before submitting another lecture.",
        )
    return fp


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    start_background_cleanup_daemon()


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_student_dashboard() -> HTMLResponse:
    index_file = Path(__file__).resolve().parent / "static" / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.get("/api/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.version,
        "max_concurrent_workers": settings.max_concurrent_workers,
        "retention_hours": settings.retention_hours,
    }


@app.post("/api/jobs", response_model=JobResponse, status_code=201)
def create_job(payload: CreateJobRequest, request: Request) -> JobResponse:
    fp = _enforce_rate_limit(request)

    if payload.source_type == SourceType.YOUTUBE and not payload.source_url:
        raise HTTPException(status_code=400, detail="A valid YouTube lecture URL is required when source_type='youtube'.")

    default_title = (
        payload.custom_title
        or ("Demo Lecture" if payload.source_type == SourceType.DEMO_LECTURE else "YouTube Lecture")
    )

    job_id = create_job_record(
        client_fingerprint=fp,
        source_type=payload.source_type,
        source_url=payload.source_url,
        upload_path=None,
        video_title=default_title,
        rights_attested=payload.rights_attested,
        enable_temporal_inpainting=payload.enable_temporal_inpainting,
        enable_ocr_indexing=payload.enable_ocr_indexing,
    )
    enqueue_job(job_id)

    job_resp = get_job_response(job_id)
    if not job_resp:
        raise HTTPException(status_code=500, detail="Job record could not be loaded after creation.")
    return job_resp


@app.post("/api/jobs/upload", response_model=JobResponse, status_code=201)
def create_upload_job(
    request: Request,
    video_file: UploadFile = File(...),
    rights_attested: bool = Form(...),
    enable_temporal_inpainting: bool = Form(True),
    enable_ocr_indexing: bool = Form(True),
    custom_title: Optional[str] = Form(None),
) -> JobResponse:
    if not rights_attested:
        raise HTTPException(
            status_code=400,
            detail="You must confirm copyright ownership, Creative Commons license, or instructor authorization.",
        )

    fp = _enforce_rate_limit(request)
    filename = video_file.filename or "lecture.mp4"
    ext = Path(filename).suffix.lower()
    if ext not in (".mp4", ".webm", ".mkv", ".mov", ".avi"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format. Please upload an .mp4, .webm, .mkv, or .mov lecture video.",
        )

    safe_upload_name = f"upload_{hashlib.md5(f'{filename}_{fp}'.encode()).hexdigest()[:12]}{ext}"
    dest_path = settings.uploads_dir / safe_upload_name

    # Stream upload to disk while enforcing size cap
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    written = 0
    with dest_path.open("wb") as out_f:
        while True:
            chunk = video_file.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                out_f.close()
                dest_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Uploaded file exceeds maximum free-tier size of {settings.max_upload_size_mb} MB.",
                )
            out_f.write(chunk)

    title = custom_title or Path(filename).stem.replace("_", " ")
    job_id = create_job_record(
        client_fingerprint=fp,
        source_type=SourceType.DIRECT_UPLOAD,
        source_url=None,
        upload_path=str(dest_path),
        video_title=title,
        rights_attested=True,
        enable_temporal_inpainting=enable_temporal_inpainting,
        enable_ocr_indexing=enable_ocr_indexing,
    )
    enqueue_job(job_id)

    job_resp = get_job_response(job_id)
    if not job_resp:
        raise HTTPException(status_code=500, detail="Job could not be loaded.")
    return job_resp


@app.get("/api/jobs", response_model=List[JobResponse])
def get_recent_jobs() -> List[JobResponse]:
    return list_jobs(limit=25)


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
def get_job_status(job_id: str) -> JobResponse:
    job = get_job_response(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job_endpoint(job_id: str) -> dict:
    ok = request_job_cancellation(job_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled (already finished or does not exist).")
    return {"status": "cancel_requested", "job_id": job_id}


@app.post("/api/jobs/{job_id}/retry", response_model=JobResponse)
def retry_job_endpoint(job_id: str) -> JobResponse:
    ok = reset_job_for_retry(job_id)
    if not ok:
        raise HTTPException(
            status_code=400,
            detail=f"Job cannot be retried (maximum {settings.max_retries} retries reached or job not found).",
        )
    enqueue_job(job_id)
    job = get_job_response(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.get("/api/jobs/{job_id}/download")
def download_job_pdf(job_id: str) -> FileResponse:
    raw = get_raw_job(job_id)
    if not raw or not raw.get("pdf_filename"):
        raise HTTPException(status_code=404, detail="PDF is not ready or has expired.")
    pdf_path = settings.outputs_dir / job_id / raw["pdf_filename"]
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found on storage.")
    clean_title = "".join(c if c.isalnum() or c in (" ", "-", "_") else "" for c in raw["video_title"]).strip()
    download_name = f"{clean_title[:50].replace(' ', '_') or 'Lecture'}_Slides.pdf"
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=download_name,
    )


@app.get("/api/jobs/{job_id}/slides/{image_name}")
def get_slide_image(job_id: str, image_name: str) -> FileResponse:
    # Prevent path traversal
    safe_name = Path(image_name).name
    img_path = settings.outputs_dir / job_id / safe_name
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Slide image not found.")
    return FileResponse(path=str(img_path), media_type="image/jpeg")
