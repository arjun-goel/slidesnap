from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, A4
from reportlab.pdfgen import canvas

from .cv_pipeline import ExtractedSlide


def _format_timestamp(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def compile_lecture_pdf(
    *,
    pdf_path: Path,
    video_title: str,
    source_label: str,
    slides: List[ExtractedSlide],
    slide_image_paths: List[Path],
) -> int:
    """
    Compiles extracted clean slide images into a high-resolution, searchable Landscape A4 PDF.
    Includes:
    - Cover Header on Page 1 with lecture metadata & copyright/rights attestation notice
    - High-DPI centered slide canvas per page
    - Timestamp badges (e.g., 00:04 - 00:08), Sharpness & Occlusion metrics
    - Explicit '[RECONSTRUCTED SLIDE - Multi-Frame Temporal De-Occlusion]' badge whenever fallback was used
    """
    page_w, page_h = landscape(A4)  # 841.89 x 595.27 pt
    c = canvas.Canvas(str(pdf_path), pagesize=(page_w, page_h))
    c.setTitle(f"{video_title} - SlideSnap Notes")
    c.setAuthor("SlideSnap CV Slide Extractor")
    c.setSubject("Extracted Lecture Slides with Presenter De-Occlusion")

    total_slides = len(slides)
    reconstructed_count = sum(1 for s in slides if s.was_reconstructed)
    generated_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    for idx, (slide, img_path) in enumerate(zip(slides, slide_image_paths)):
        # Background
        c.setFillColor(colors.HexColor("#F8FAFC"))
        c.rect(0, 0, page_w, page_h, stroke=0, fill=1)

        # Top Metadata Bar
        c.setFillColor(colors.HexColor("#0F172A"))
        c.rect(0, page_h - 46, page_w, 46, stroke=0, fill=1)

        c.setFillColor(colors.HexColor("#F8FAFC"))
        c.setFont("Helvetica-Bold", 11)
        truncated_title = video_title[:75] + ("..." if len(video_title) > 75 else "")
        c.drawString(24, page_h - 28, f"Slide {slide.slide_number} of {total_slides}  |  {truncated_title}")

        # Timestamp Pill
        ts_range = f"{_format_timestamp(slide.timestamp_start_sec)} - {_format_timestamp(slide.timestamp_end_sec)} (Captured @ {_format_timestamp(slide.selected_frame_sec)})"
        c.setFont("Helvetica", 9.5)
        c.setFillColor(colors.HexColor("#94A3B8"))
        c.drawRightString(page_w - 24, page_h - 28, ts_range)

        # Reconstructed Warning/Transparency Badge (Requirement: Clearly label reconstructed slides)
        if slide.was_reconstructed:
            c.setFillColor(colors.HexColor("#FEF3C7"))
            c.roundRect(24, page_h - 74, 385, 22, 4, stroke=0, fill=1)
            c.setStrokeColor(colors.HexColor("#F59E0B"))
            c.roundRect(24, page_h - 74, 385, 22, 4, stroke=1, fill=0)
            c.setFillColor(colors.HexColor("#92400E"))
            c.setFont("Helvetica-Bold", 8.5)
            c.drawString(
                34,
                page_h - 66,
                f"RECONSTRUCTED SLIDE ({slide.reconstruction_method.replace('_', ' ').upper()})",
            )
        else:
            c.setFillColor(colors.HexColor("#DCFCE7"))
            c.roundRect(24, page_h - 74, 240, 22, 4, stroke=0, fill=1)
            c.setFillColor(colors.HexColor("#166534"))
            c.setFont("Helvetica-Bold", 8.5)
            c.drawString(34, page_h - 66, "CLEAN DIRECT FRAME (UNOBSTRUCTED)")

        # Quality Metrics on right sub-header
        c.setFillColor(colors.HexColor("#475569"))
        c.setFont("Helvetica", 8.5)
        occ_pct = slide.person_occlusion_ratio * 100.0
        c.drawRightString(
            page_w - 24,
            page_h - 66,
            f"Quality Score: {slide.final_quality_score * 100:.1f}%  •  Sharpness: {slide.sharpness_score:.0f}  •  Residual Occlusion: {occ_pct:.2f}%  •  dHash: {slide.perceptual_hash[:10]}",
        )

        # Main Slide Image (16:9 aspect ratio fitted cleanly)
        avail_w = page_w - 48
        avail_h = page_h - 140
        target_w = avail_w
        target_h = target_w * (9.0 / 16.0)
        if target_h > avail_h:
            target_h = avail_h
            target_w = target_h * (16.0 / 9.0)

        img_x = (page_w - target_w) / 2.0
        img_y = 52 + (avail_h - target_h) / 2.0

        # Subtle border shadow around slide
        c.setStrokeColor(colors.HexColor("#CBD5E1"))
        c.setLineWidth(1)
        c.rect(img_x - 1, img_y - 1, target_w + 2, target_h + 2, stroke=1, fill=0)
        c.drawImage(str(img_path), img_x, img_y, width=target_w, height=target_h, preserveAspectRatio=True)

        # Bottom OCR / Searchable Text Layer & Attribution Footer
        c.setFillColor(colors.HexColor("#E2E8F0"))
        c.rect(0, 0, page_w, 42, stroke=0, fill=1)

        c.setFillColor(colors.HexColor("#334155"))
        c.setFont("Helvetica", 8)
        ocr_line = slide.ocr_text[:135] + ("..." if len(slide.ocr_text) > 135 else "")
        c.drawString(24, 24, f"Indexed Content: {ocr_line}")

        c.setFillColor(colors.HexColor("#64748B"))
        c.setFont("Helvetica", 7.5)
        c.drawString(
            24,
            10,
            f"Generated by SlideSnap ({generated_date}) • User-attested educational/authorized content • Reconstructed slides: {reconstructed_count}/{total_slides}",
        )
        c.drawRightString(page_w - 24, 10, f"Page {idx + 1} / {total_slides}")

        c.showPage()

    c.save()
    return pdf_path.stat().st_size
