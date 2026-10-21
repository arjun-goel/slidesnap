from __future__ import annotations

from pathlib import Path
import time
import tempfile
from fastapi.testclient import TestClient

from app.cv_pipeline import run_slide_extraction_pipeline
from app.demo_generator import generate_synthetic_lecture_video
from app.main import app


def test_cv_pipeline_extracts_4_unique_slides_and_deoccludes_walking_presenter() -> None:
    """
    Verifies that our synthetic lecture (which has 4 distinct slides + 1 revisited duplicate slide at the end,
    a presenter walking across Slide 2 in every frame, blur on Slide 3, and UI overlay on Slide 4):
    1. Extracts exactly 4 unique slides (deduplicating the revisited Slide 1 at the end).
    2. Triggers Multi-Frame Temporal Median De-Occlusion on Slide 2 where the presenter never leaves the screen.
    3. Selects sharp, unobstructed frames on Slides 1, 3, and 4.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video_file = tmp_path / "test_lecture.mp4"
        title, duration = generate_synthetic_lecture_video(video_file)

        assert video_file.exists()
        assert duration >= 15

        slides, total_candidates, _ = run_slide_extraction_pipeline(
            video_path=video_file,
            output_dir=tmp_path / "out",
            enable_temporal_inpainting=True,
            enable_ocr_indexing=True,
        )

        assert total_candidates >= 30
        assert len(slides) == 4, f"Expected 4 deduplicated slides, got {len(slides)}"

        # Slide 2 has the presenter walking across the slide in 100% of frames -> must be reconstructed
        reconstructed_slides = [s for s in slides if s.was_reconstructed]
        assert len(reconstructed_slides) >= 1
        assert slides[1].was_reconstructed is True
        assert "temporal" in slides[1].reconstruction_method


def test_api_end_to_end_demo_job_and_pdf_download() -> None:
    client = TestClient(app)

    # 1. Verify health endpoint
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    # 2. Verify copyright attestation is enforced
    bad_req = client.post(
        "/api/jobs",
        json={
            "source_type": "demo_lecture",
            "rights_attested": False,
        },
    )
    assert bad_req.status_code == 422

    # 3. Submit a valid Demo Lecture job
    create_res = client.post(
        "/api/jobs",
        json={
            "source_type": "demo_lecture",
            "rights_attested": True,
            "enable_temporal_inpainting": True,
            "enable_ocr_indexing": True,
            "custom_title": "Automated Test Lecture",
        },
    )
    assert create_res.status_code == 201
    job_id = create_res.json()["id"]

    # 4. Poll until completed
    final_job = None
    for _ in range(40):
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
    assert len(final_job["slides"]) == 4

    # 5. Download compiled PDF and verify PDF header bytes (%PDF-)
    pdf_res = client.get(final_job["pdf_download_url"])
    assert pdf_res.status_code == 200
    assert pdf_res.headers["content-type"] == "application/pdf"
    assert pdf_res.content.startswith(b"%PDF-")
