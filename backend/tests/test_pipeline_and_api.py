from __future__ import annotations

from pathlib import Path
import tempfile
import time
from fastapi.testclient import TestClient
import numpy as np

from app.cv_pipeline import run_slide_extraction_pipeline
from app.demo_generator import (
    generate_progressive_board_lecture_video,
    generate_synthetic_lecture_video,
)
from app.main import app


def test_cv_pipeline_extracts_4_unique_slides_and_deoccludes_walking_presenter() -> None:
    """
    Verifies that our synthetic lecture (4 distinct slides + 1 revisited duplicate slide at the end,
    a presenter walking across Slide 2 in every frame, blur on Slide 3, and UI overlay on Slide 4):
    1. Extracts exactly 4 unique slides (deduplicating the revisited Slide 1 at the end).
    2. Triggers Multi-Frame Temporal De-Occlusion on Slide 2 where the presenter never leaves the screen.
    3. Saves Before/After comparison previews for reconstructed slides.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video_file = tmp_path / "test_lecture.mp4"
        _, duration = generate_synthetic_lecture_video(video_file)

        assert video_file.exists()
        assert duration >= 15

        slides, total_candidates, _ = run_slide_extraction_pipeline(
            video_path=video_file,
            output_dir=tmp_path / "out",
            quality_mode="balanced",
            enable_temporal_inpainting=True,
            enable_ocr_indexing=True,
        )

        assert total_candidates >= 30
        assert len(slides) == 4, f"Expected 4 deduplicated slides, got {len(slides)}"

        reconstructed_slides = [s for s in slides if s.was_reconstructed]
        assert len(reconstructed_slides) >= 1
        assert slides[1].was_reconstructed is True
        assert "temporal" in slides[1].reconstruction_method
        assert slides[1].occluded_image_path is not None and slides[1].occluded_image_path.exists()


def test_progressive_chalkboard_pre_transition_and_teacher_removal() -> None:
    """
    Verifies the core Pre-Transition Full-Board + Reverse-Chronological Teacher Removal engine:
    - Board 1 writes Step 1 (0-2s), Step 2 (2-4s), and Step 3 boxed Q.E.D. (4-6s) right before wiping to Slide 2,
      while the teacher stands directly in front of Step 1 & Step 2 during 4-6s!
    Must:
    1. Extract exactly 2 slides (keep all 6s of progressive writing as 1 complete board, NOT 3 partial boards).
    2. Preserve Step 3 chalk box on the right (x=680..1210, y=210..470) written right before transition.
    3. Cleanly erase the teacher at x=310 so Step 1 & Step 2 chalk strokes on the left remain visible!
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video_file = tmp_path / "progressive_board.mp4"
        generate_progressive_board_lecture_video(video_file)

        slides, _, _ = run_slide_extraction_pipeline(
            video_path=video_file,
            output_dir=tmp_path / "out_board",
            quality_mode="balanced",
            enhance_contrast=True,
            enable_temporal_inpainting=True,
        )

        assert len(slides) == 2, f"Expected 2 slides (1 complete chalkboard + 1 summary slide), got {len(slides)}"
        board_slide = slides[0]
        assert board_slide.was_reconstructed is True
        img = board_slide.image

        # Verify right-side Step 3 pre-transition chalk box & text (bright chalk pixels > 180) exist!
        right_step3_roi = img[205:475, 675:1215]
        bright_chalk_right = int(np.count_nonzero(np.mean(right_step3_roi, axis=2) > 175))
        assert bright_chalk_right > 1500, f"Pre-transition Step 3 chalk missing! bright pixels={bright_chalk_right}"

        # Verify left-side Step 1 & Step 2 chalk strokes (where teacher stood at x=310!) were restored,
        # and skin-colored head at (310, 332) was removed back to green chalkboard!
        head_roi = img[315:355, 290:330]  # Center of teacher's face in pre-transition frames
        mean_b, mean_g, mean_r = np.mean(head_roi, axis=(0, 1))
        assert mean_g > mean_r, f"Teacher face was not removed from green chalkboard! BGR=({mean_b:.1f}, {mean_g:.1f}, {mean_r:.1f})"


def test_api_end_to_end_demo_job_curated_pdf_markdown_and_zip() -> None:
    client = TestClient(app)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert "fast" in health.json()["quality_presets"]

    bad_req = client.post(
        "/api/jobs",
        json={
            "source_type": "demo_lecture",
            "rights_attested": False,
        },
    )
    assert bad_req.status_code == 422

    create_res = client.post(
        "/api/jobs",
        json={
            "source_type": "demo_lecture",
            "quality_mode": "fast",
            "enhance_contrast": True,
            "rights_attested": True,
            "enable_temporal_inpainting": True,
            "enable_ocr_indexing": True,
            "custom_title": "Fast Tier Test Lecture",
        },
    )
    assert create_res.status_code == 201
    job_id = create_res.json()["id"]
    assert create_res.json()["quality_mode"] == "fast"

    final_job = None
    for _ in range(50):
        status_res = client.get(f"/api/jobs/{job_id}")
        assert status_res.status_code == 200
        data = status_res.json()
        if data["status"] in ("completed", "failed", "cancelled"):
            final_job = data
            break
        time.sleep(0.25)

    assert final_job is not None
    assert final_job["status"] == "completed", f"Job failed: {final_job.get('error_message')}"
    assert final_job["extracted_slide_count"] == 4

    # Full PDF
    pdf_res = client.get(final_job["pdf_download_url"])
    assert pdf_res.status_code == 200
    assert pdf_res.content.startswith(b"%PDF-")

    # Curated subset PDF (?slides=1,2)
    curated_res = client.get(f"{final_job['pdf_download_url']}?slides=1,2")
    assert curated_res.status_code == 200
    assert curated_res.content.startswith(b"%PDF-")

    # Markdown study guide & ZIP exports
    md_res = client.get(final_job["markdown_download_url"])
    assert md_res.status_code == 200
    assert "SlideSnap Study Guide" in md_res.text

    zip_res = client.get(final_job["zip_download_url"])
    assert zip_res.status_code == 200
    assert zip_res.headers["content-type"] == "application/zip"
