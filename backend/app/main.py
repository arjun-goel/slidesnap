from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import List, Optional
import zipfile

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse

from .config import QUALITY_PRESETS, settings
from .database import (
    count_recent_jobs_for_client,
    create_job_record,
    get_job_response,
    get_raw_job,
    get_raw_job_slides,
    init_db,
    list_jobs,
    request_job_cancellation,
    reset_job_for_retry,
)
from .models import CreateJobRequest, JobResponse, QualityMode, SourceType
from .pdf_generator import compile_lecture_pdf
from .worker import enqueue_job, start_background_cleanup_daemon

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Streaming Low-Memory Computer Vision Lecture Slide & Chalkboard Extractor API ($0 Student Cloud Tier)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
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


def _clean_filename(title: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in (" ", "-", "_") else "" for c in title).strip()
    return cleaned[:50].replace(" ", "_") or "Lecture"


def _parse_slide_filter(slides_param: Optional[str]) -> Optional[set[int]]:
    if not slides_param:
        return None
    selected: set[int] = set()
    for part in slides_param.split(","):
        part = part.strip()
        if part.isdigit():
            selected.add(int(part))
    return selected or None


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
        "engine": "streaming_pre_transition_v2",
        "max_concurrent_workers": settings.max_concurrent_workers,
        "retention_hours": settings.retention_hours,
        "quality_presets": {
            k: {
                "label": v.label,
                "resolution": f"{v.target_width}x{v.target_height}",
                "base_fps": v.base_fps,
            }
            for k, v in QUALITY_PRESETS.items()
        },
    }


@app.post("/api/jobs", response_model=JobResponse, status_code=201)
def create_job(payload: CreateJobRequest, request: Request) -> JobResponse:
    fp = _enforce_rate_limit(request)

    if payload.source_type == SourceType.YOUTUBE and not payload.source_url:
        raise HTTPException(status_code=400, detail="A valid YouTube lecture URL is required when source_type='youtube'.")

    fallback_titles = {
        SourceType.DEMO_LECTURE: "Demo Lecture",
        SourceType.PROGRESSIVE_BOARD_DEMO: "Progressive Board Demo",
    }
    default_title = payload.custom_title or fallback_titles.get(payload.source_type, "YouTube Lecture")

    job_id = create_job_record(
        client_fingerprint=fp,
        source_type=payload.source_type,
        source_url=payload.source_url,
        upload_path=None,
        video_title=default_title,
        quality_mode=payload.quality_mode,
        enhance_contrast=payload.enhance_contrast,
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
    quality_mode: QualityMode = Form(QualityMode.BALANCED),
    enhance_contrast: bool = Form(False),
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
        quality_mode=quality_mode,
        enhance_contrast=enhance_contrast,
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
def download_job_pdf(
    job_id: str,
    slides: Optional[str] = Query(default=None, description="Optional comma-separated slide numbers for curated PDF export"),
) -> FileResponse:
    raw = get_raw_job(job_id)
    if not raw or not raw.get("pdf_filename"):
        raise HTTPException(status_code=404, detail="PDF is not ready or has expired.")

    job_dir = settings.outputs_dir / job_id
    selected_nums = _parse_slide_filter(slides)
    download_name = f"{_clean_filename(raw['video_title'])}_Slides.pdf"

    if selected_nums:
        all_rows = get_raw_job_slides(job_id)
        chosen_rows = [r for r in all_rows if int(r["slide_number"]) in selected_nums]
        img_paths = [job_dir / r["image_filename"] for r in chosen_rows if (job_dir / r["image_filename"]).exists()]
        chosen_rows = [r for r in chosen_rows if (job_dir / r["image_filename"]).exists()]
        if not chosen_rows:
            raise HTTPException(status_code=400, detail="None of the selected slides were found.")

        sig = hashlib.md5(",".join(str(r["slide_number"]) for r in chosen_rows).encode()).hexdigest()[:8]
        curated_pdf = job_dir / f"SlideSnap_Curated_{sig}.pdf"
        compile_lecture_pdf(
            pdf_path=curated_pdf,
            video_title=raw["video_title"],
            source_label=raw.get("source_url") or raw["source_type"],
            slides=chosen_rows,
            slide_image_paths=img_paths,
            quality_mode=str(raw.get("quality_mode") or "balanced"),
        )
        return FileResponse(
            path=str(curated_pdf),
            media_type="application/pdf",
            filename=f"{_clean_filename(raw['video_title'])}_Curated_{len(chosen_rows)}Slides.pdf",
        )

    pdf_path = job_dir / raw["pdf_filename"]
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found on storage.")
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=download_name,
    )


@app.get("/api/jobs/{job_id}/markdown")
def download_job_markdown(
    job_id: str,
    slides: Optional[str] = Query(default=None),
) -> PlainTextResponse:
    raw = get_raw_job(job_id)
    if not raw:
        raise HTTPException(status_code=404, detail="Job not found.")
    rows = get_raw_job_slides(job_id)
    selected_nums = _parse_slide_filter(slides)
    if selected_nums:
        rows = [r for r in rows if int(r["slide_number"]) in selected_nums]

    src_url = raw.get("source_url") or ""
    is_yt = "youtube.com" in src_url or "youtu.be" in src_url

    lines = [
        f"# {raw['video_title']} — SlideSnap Study Guide",
        "",
        f"- **Extracted Slides**: {len(rows)}",
        f"- **Quality Preset**: `{raw.get('quality_mode', 'balanced')}`",
        f"- **Source**: {src_url or raw['source_type']}",
        "",
        "---",
        "",
    ]

    for r in rows:
        s_num = r["slide_number"]
        t_start = int(r["timestamp_start_sec"])
        t_end = int(r["timestamp_end_sec"])
        mm_s, ss_s = divmod(t_start, 60)
        mm_e, ss_e = divmod(t_end, 60)
        ts_label = f"{mm_s:02d}:{ss_s:02d} – {mm_e:02d}:{ss_e:02d}"
        if is_yt and "watch?v=" in src_url:
            ts_link = f"[{ts_label}]({src_url}&t={t_start}s)"
        else:
            ts_link = f"`{ts_label}`"

        mode_badge = (
            f"Reconstructed (`{r['reconstruction_method']}`)"
            if r["was_reconstructed"]
            else "Unobstructed Pre-Transition Frame"
        )
        lines.extend(
            [
                f"## Slide {s_num} ({ts_link})",
                f"- **Capture Method**: {mode_badge}",
                f"- **Quality Score**: {float(r['final_quality_score']) * 100:.1f}% | **Sharpness**: {float(r['sharpness_score']):.0f} | **Residual Occlusion**: {float(r['person_occlusion_ratio']) * 100:.2f}%",
                f"- **Extracted Notes / Layout**: {r['ocr_text']}",
                "",
            ]
        )

    filename = f"{_clean_filename(raw['video_title'])}_Notes.md"
    return PlainTextResponse(
        content="\n".join(lines),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/jobs/{job_id}/zip")
def download_job_slides_zip(
    job_id: str,
    slides: Optional[str] = Query(default=None),
) -> StreamingResponse:
    raw = get_raw_job(job_id)
    if not raw:
        raise HTTPException(status_code=404, detail="Job not found.")
    rows = get_raw_job_slides(job_id)
    selected_nums = _parse_slide_filter(slides)
    if selected_nums:
        rows = [r for r in rows if int(r["slide_number"]) in selected_nums]

    job_dir = settings.outputs_dir / job_id
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for r in rows:
            img_p = job_dir / r["image_filename"]
            if img_p.exists():
                zf.write(img_p, arcname=f"Slide_{int(r['slide_number']):02d}.jpg")

    buf.seek(0)
    filename = f"{_clean_filename(raw['video_title'])}_Slides_HD.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/jobs/{job_id}/slides/{image_name}")
def get_slide_image(job_id: str, image_name: str) -> FileResponse:
    safe_name = Path(image_name).name
    img_path = settings.outputs_dir / job_id / safe_name
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Slide image not found.")
    return FileResponse(path=str(img_path), media_type="image/jpeg")
