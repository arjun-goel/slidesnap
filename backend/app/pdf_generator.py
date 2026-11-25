from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Sequence, Union

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas

from .config import QUALITY_PRESETS
from .cv_pipeline import ExtractedSlide


def _format_timestamp(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def _attr(obj: Union[ExtractedSlide, dict[str, Any]], key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def compile_lecture_pdf(
    *,
    pdf_path: Path,
    video_title: str,
    source_label: str,
    slides: Sequence[Union[ExtractedSlide, dict[str, Any]]],
    slide_image_paths: List[Path],
    quality_mode: str = "balanced",
) -> int:
    """
    Compiles extracted clean slide images into a high-resolution, searchable Landscape A4 PDF.
    Supports both full pipeline `ExtractedSlide` objects and curated subsets from SQLite dict rows.
    """
    page_w, page_h = landscape(A4)  # 841.89 x 595.27 pt
    c = canvas.Canvas(str(pdf_path), pagesize=(page_w, page_h))
    c.setTitle(f"{video_title} - SlideSnap Notes")
    c.setAuthor("SlideSnap CV Slide Extractor")
    c.setSubject("Extracted Lecture Slides with Pre-Transition Board Lock & Presenter De-Occlusion")

    preset = QUALITY_PRESETS.get(quality_mode, QUALITY_PRESETS["balanced"])
    total_slides = len(slides)
    reconstructed_count = sum(1 for s in slides if bool(_attr(s, "was_reconstructed", False)))
    generated_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    for idx, (slide, img_path) in enumerate(zip(slides, slide_image_paths)):
        slide_num = int(_attr(slide, "slide_number", idx + 1))
        ts_start = float(_attr(slide, "timestamp_start_sec", 0.0))
        ts_end = float(_attr(slide, "timestamp_end_sec", 0.0))
        ts_sel = float(_attr(slide, "selected_frame_sec", 0.0))
        was_rec = bool(_attr(slide, "was_reconstructed", False))
        rec_method = str(_attr(slide, "reconstruction_method", "single_clean_frame"))
        qual_score = float(_attr(slide, "final_quality_score", 0.9))
        sharp_score = float(_attr(slide, "sharpness_score", 100.0))
        occ_ratio = float(_attr(slide, "person_occlusion_ratio", 0.0))
        comp_score = float(_attr(slide, "content_completeness_score", 1.0))
        phash = str(_attr(slide, "perceptual_hash", ""))
        ocr_text = str(_attr(slide, "ocr_text", ""))

        # Background
        c.setFillColor(colors.HexColor("#F8FAFC"))
        c.rect(0, 0, page_w, page_h, stroke=0, fill=1)

        # Top Metadata Bar
        c.setFillColor(colors.HexColor("#0F172A"))
        c.rect(0, page_h - 46, page_w, 46, stroke=0, fill=1)

        c.setFillColor(colors.HexColor("#F8FAFC"))
        c.setFont("Helvetica-Bold", 11)
        truncated_title = video_title[:70] + ("..." if len(video_title) > 70 else "")
        c.drawString(24, page_h - 28, f"Slide {idx + 1} of {total_slides} (#{slide_num})  |  {truncated_title}")

        # Timestamp Pill
        ts_range = f"{_format_timestamp(ts_start)} - {_format_timestamp(ts_end)} (Locked @ {_format_timestamp(ts_sel)})"
        c.setFont("Helvetica", 9.5)
        c.setFillColor(colors.HexColor("#94A3B8"))
        c.drawRightString(page_w - 24, page_h - 28, ts_range)

        # Transparency Badge
        if was_rec:
            c.setFillColor(colors.HexColor("#FEF3C7"))
            c.roundRect(24, page_h - 74, 392, 22, 4, stroke=0, fill=1)
            c.setStrokeColor(colors.HexColor("#F59E0B"))
            c.roundRect(24, page_h - 74, 392, 22, 4, stroke=1, fill=0)
            c.setFillColor(colors.HexColor("#92400E"))
            c.setFont("Helvetica-Bold", 8.2)
            c.drawString(
                34,
                page_h - 66,
                f"RECONSTRUCTED SLIDE ({rec_method.replace('_', ' ').upper()})",
            )
        else:
            c.setFillColor(colors.HexColor("#DCFCE7"))
            c.roundRect(24, page_h - 74, 275, 22, 4, stroke=0, fill=1)
            c.setFillColor(colors.HexColor("#166534"))
            c.setFont("Helvetica-Bold", 8.2)
            badge_txt = (
                "PRE-TRANSITION FULL BOARD (UNOBSTRUCTED)"
                if "pre_transition" in rec_method
                else "CLEAN DIRECT FRAME (UNOBSTRUCTED)"
            )
            c.drawString(34, page_h - 66, badge_txt)

        # Quality Metrics on right sub-header
        c.setFillColor(colors.HexColor("#475569"))
        c.setFont("Helvetica", 8.2)
        occ_pct = occ_ratio * 100.0
        c.drawRightString(
            page_w - 24,
            page_h - 66,
            f"Board Completeness: {comp_score * 100:.0f}%  •  Quality: {qual_score * 100:.1f}%  •  Sharpness: {sharp_score:.0f}  •  Occlusion: {occ_pct:.2f}%  •  {preset.label}",
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

        c.setStrokeColor(colors.HexColor("#CBD5E1"))
        c.setLineWidth(1)
        c.rect(img_x - 1, img_y - 1, target_w + 2, target_h + 2, stroke=1, fill=0)
        c.drawImage(str(img_path), img_x, img_y, width=target_w, height=target_h, preserveAspectRatio=True)

        # Bottom OCR / Searchable Text Layer & Attribution Footer
        c.setFillColor(colors.HexColor("#E2E8F0"))
        c.rect(0, 0, page_w, 42, stroke=0, fill=1)

        c.setFillColor(colors.HexColor("#334155"))
        c.setFont("Helvetica", 8)
        ocr_line = ocr_text[:135] + ("..." if len(ocr_text) > 135 else "")
        c.drawString(24, 24, f"Indexed Content: {ocr_line}  [dHash: {phash[:10]}]")

        c.setFillColor(colors.HexColor("#64748B"))
        c.setFont("Helvetica", 7.5)
        c.drawString(
            24,
            10,
            f"Generated by SlideSnap v2.0 ({generated_date}) • Pre-Transition & Teacher De-Occlusion Engine • Reconstructed: {reconstructed_count}/{total_slides}",
        )
        c.drawRightString(page_w - 24, 10, f"Page {idx + 1} / {total_slides}")

        c.showPage()

    c.save()
    return pdf_path.stat().st_size
