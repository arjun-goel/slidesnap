from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
import re


class JobStatus(str, Enum):
    QUEUED = "queued"
    INGESTING = "ingesting"
    SAMPLING_FRAMES = "sampling_frames"
    DETECTING_SLIDES = "detecting_slides"
    REMOVING_OCCLUSIONS = "removing_occlusions"
    COMPILING_PDF = "compiling_pdf"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SourceType(str, Enum):
    YOUTUBE = "youtube"
    DIRECT_UPLOAD = "direct_upload"
    DEMO_LECTURE = "demo_lecture"


YOUTUBE_REGEX = re.compile(
    r"^(https?://)?(www\.|m\.)?(youtube\.com/(watch\?.*v=|live/|shorts/|embed/)|youtu\.be/)[\w\-]{11}"
)


class CreateJobRequest(BaseModel):
    source_type: SourceType = Field(default=SourceType.YOUTUBE)
    source_url: Optional[str] = Field(
        default=None,
        description="Valid YouTube lecture URL (required when source_type='youtube')",
    )
    rights_attested: bool = Field(
        ...,
        description="Student attestation that they own the video or have rights/permission to extract study slides.",
    )
    enable_temporal_inpainting: bool = Field(
        default=True,
        description="Enable multi-frame temporal composite when the presenter obstructs all frames of a slide.",
    )
    enable_ocr_indexing: bool = Field(
        default=True,
        description="Extract key heading/bullet keywords from slides for searchable PDF metadata.",
    )
    custom_title: Optional[str] = Field(default=None, max_length=140)

    @field_validator("rights_attested")
    @classmethod
    def must_attest_rights(cls, v: bool) -> bool:
        if not v:
            raise ValueError(
                "You must confirm copyright ownership, Creative Commons license, or instructor authorization before processing."
            )
        return v

    @field_validator("source_url")
    @classmethod
    def validate_youtube_url(cls, v: Optional[str], info) -> Optional[str]:
        if v is not None:
            v = v.strip()
            if v and not YOUTUBE_REGEX.match(v):
                raise ValueError(
                    "Invalid YouTube URL. Please provide a standard youtube.com/watch?v=... or youtu.be/... lecture link."
                )
        return v


class SlideMetadataResponse(BaseModel):
    id: str
    job_id: str
    slide_number: int
    timestamp_start_sec: float
    timestamp_end_sec: float
    selected_frame_sec: float
    sharpness_score: float
    person_occlusion_ratio: float
    final_quality_score: float
    perceptual_hash: str
    was_reconstructed: bool
    reconstruction_method: str
    ocr_text: str
    image_url: str


class JobResponse(BaseModel):
    id: str
    source_type: SourceType
    source_url: Optional[str]
    video_title: str
    video_duration_sec: int
    rights_attested: bool
    enable_temporal_inpainting: bool
    enable_ocr_indexing: bool
    status: JobStatus
    progress_pct: int
    current_stage_detail: str
    total_candidate_frames: int
    extracted_slide_count: int
    reconstructed_slide_count: int
    pdf_download_url: Optional[str]
    pdf_file_size_bytes: Optional[int]
    error_code: Optional[str]
    error_message: Optional[str]
    retry_count: int
    cancel_requested: bool
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    expires_at: str
    slides: List[SlideMetadataResponse] = Field(default_factory=list)
