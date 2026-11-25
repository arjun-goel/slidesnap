export type JobStatus =
  | "queued"
  | "ingesting"
  | "sampling_frames"
  | "detecting_slides"
  | "removing_occlusions"
  | "compiling_pdf"
  | "completed"
  | "failed"
  | "cancelled";

export type SourceType =
  | "youtube"
  | "direct_upload"
  | "demo_lecture"
  | "progressive_board_demo";

export type QualityMode = "fast" | "balanced" | "high";

export interface SlideMetadata {
  id: string;
  job_id: string;
  slide_number: number;
  timestamp_start_sec: number;
  timestamp_end_sec: number;
  selected_frame_sec: number;
  sharpness_score: number;
  person_occlusion_ratio: number;
  final_quality_score: number;
  content_completeness_score?: number;
  perceptual_hash: string;
  was_reconstructed: boolean;
  reconstruction_method: string;
  ocr_text: string;
  image_url: string;
  occluded_image_url?: string | null;
}

export interface JobResponse {
  id: string;
  source_type: SourceType;
  source_url: string | null;
  video_title: string;
  video_duration_sec: number;
  quality_mode?: QualityMode;
  enhance_contrast?: boolean;
  used_youtube_fallback?: boolean;
  rights_attested: boolean;
  enable_temporal_inpainting: boolean;
  enable_ocr_indexing: boolean;
  status: JobStatus;
  progress_pct: number;
  current_stage_detail: string;
  total_candidate_frames: number;
  extracted_slide_count: number;
  reconstructed_slide_count: number;
  pdf_download_url: string | null;
  markdown_download_url?: string | null;
  zip_download_url?: string | null;
  pdf_file_size_bytes: number | null;
  error_code: string | null;
  error_message: string | null;
  retry_count: number;
  cancel_requested: boolean;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  expires_at: string;
  slides: SlideMetadata[];
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  engine?: string;
  max_concurrent_workers: number;
  retention_hours: number;
}
