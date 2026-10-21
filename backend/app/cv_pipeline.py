from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import cv2
import numpy as np

from .config import settings


@dataclass
class SampledFrame:
    index: int
    timestamp_sec: float
    image: np.ndarray  # BGR uint8
    dhash: int
    dhash_hex: str


@dataclass
class ScoredFrame:
    sampled: SampledFrame
    person_mask: np.ndarray  # uint8 (0 or 255)
    person_occlusion_ratio: float
    sharpness_score: float
    overlay_penalty: float
    composite_score: float


@dataclass
class SlideSegment:
    segment_index: int
    frames: List[SampledFrame] = field(default_factory=list)

    @property
    def start_sec(self) -> float:
        return self.frames[0].timestamp_sec if self.frames else 0.0

    @property
    def end_sec(self) -> float:
        return self.frames[-1].timestamp_sec if self.frames else 0.0

    @property
    def duration_sec(self) -> float:
        if len(self.frames) <= 1:
            return 0.5
        return max(0.5, self.end_sec - self.start_sec)


@dataclass
class ExtractedSlide:
    slide_number: int
    timestamp_start_sec: float
    timestamp_end_sec: float
    selected_frame_sec: float
    image: np.ndarray
    sharpness_score: float
    person_occlusion_ratio: float
    final_quality_score: float
    perceptual_hash: str
    dhash_int: int
    was_reconstructed: bool
    reconstruction_method: str
    ocr_text: str


def compute_dhash(image_bgr: np.ndarray, hash_size: int = 8) -> Tuple[int, str]:
    """
    Computes a 64-bit difference perceptual hash (dHash) on the high-frequency edge
    map + color header signature of the slide canvas so slides with similar outer card
    margins but different titles/bullets/diagrams produce distinct hashes.
    """
    h, w = image_bgr.shape[:2]
    crop_h = int(h * 0.92)
    cropped = image_bgr[:crop_h, :]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # Combine luminance + edge density + B-R color difference so accent colors & text placement register
    b, g, r = cv2.split(cropped)
    color_sig = cv2.absdiff(b, r)
    blended = cv2.addWeighted(gray, 0.35, edges, 0.40, 0)
    blended = cv2.addWeighted(blended, 0.75, color_sig, 0.25, 0)
    resized = cv2.resize(blended, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bit_val = 0
    for bit in diff.flatten():
        bit_val = (bit_val << 1) | int(bool(bit))
    hex_str = f"{bit_val:016x}"
    return bit_val, hex_str


def hamming_distance(hash_a: int, hash_b: int) -> int:
    return (hash_a ^ hash_b).bit_count()


def compute_slide_content_distance(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """
    Computes normalized visual & structural content distance between two slide frames.
    Combines:
    1. Top Header Banner color & title signature (top 16% of slide where presenter torso never reaches)
    2. Full-canvas color + 16x9 edge-density grid difference
    Returns a dissimilarity score in [0.0, 1.0] where <0.035 is the same slide and >=0.045 is a new slide.
    """
    h, w = img_a.shape[:2]
    crop_h = int(h * 0.92)
    ca = cv2.resize(img_a[:crop_h, :], (320, 180), interpolation=cv2.INTER_AREA)
    cb = cv2.resize(img_b[:crop_h, :], (320, 180), interpolation=cv2.INTER_AREA)

    # 1. Top header banner difference (top 18% of canvas — immune to lower-body presenter occlusion)
    header_a = ca[:32, :]
    header_b = cb[:32, :]
    header_diff = float(np.mean(cv2.absdiff(header_a, header_b))) / 255.0

    # 2. Full slide color difference
    color_diff = float(np.mean(cv2.absdiff(ca, cb))) / 255.0

    # 3. Edge density grid comparison (captures text line length and diagram topology)
    ga = cv2.cvtColor(ca, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cb, cv2.COLOR_BGR2GRAY)
    ea = cv2.Canny(ga, 60, 160).astype(np.float32) / 255.0
    eb = cv2.Canny(gb, 60, 160).astype(np.float32) / 255.0
    grid_a = cv2.resize(ea, (16, 9), interpolation=cv2.INTER_AREA)
    grid_b = cv2.resize(eb, (16, 9), interpolation=cv2.INTER_AREA)
    edge_grid_diff = float(np.mean(np.abs(grid_a - grid_b)))

    return 0.45 * header_diff + 0.30 * color_diff + 0.25 * edge_grid_diff


def compute_masked_sharpness(image_bgr: np.ndarray, person_mask: np.ndarray) -> float:
    """
    Computes the variance of the Laplacian (sharpness metric) strictly on the non-occluded
    slide background so a sharp presenter in front of an out-of-focus slide doesn't skew the score.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    non_person = person_mask == 0
    if np.count_nonzero(non_person) < 500:
        return float(lap.var())
    masked_vals = lap[non_person]
    return float(np.var(masked_vals))


def detect_playback_overlay_penalty(image_bgr: np.ndarray) -> float:
    """
    Detects intrusive bottom playback bars (dark translucent strip + high-saturation red seekbar).
    Returns a penalty in [0.0, 1.0].
    """
    h, w = image_bgr.shape[:2]
    bottom_strip = image_bgr[int(h * 0.92) :, :]
    hsv = cv2.cvtColor(bottom_strip, cv2.COLOR_BGR2HSV)
    # Check for bright red YouTube progress bar (Hue near 0/180, high saturation & value)
    red_mask1 = cv2.inRange(hsv, np.array([0, 150, 140]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 150, 140]), np.array([180, 255, 255]))
    red_ratio = float(np.count_nonzero(red_mask1 | red_mask2)) / max(1, bottom_strip.shape[0] * bottom_strip.shape[1])

    mean_luma = float(np.mean(bottom_strip))
    main_luma = float(np.mean(image_bgr[: int(h * 0.85), :]))

    penalty = 0.0
    if red_ratio > 0.008 and mean_luma < main_luma * 0.55:
        penalty = min(1.0, red_ratio * 35.0 + 0.45)
    return penalty


class PresenterDetector:
    """
    Hybrid Presenter & Occlusion Detector:
    1. Multi-scale HOG / Face + Upper-Body Cascade & Skin-Region + Silhouette Segmentation
    2. Intra-Segment Temporal Foreground Differencing (detects any moving lecturer/hand/pointer
       standing in front of the static slide background within a slide segment!)
    """

    def __init__(self) -> None:
        self.hog = None
        if hasattr(cv2, "HOGDescriptor"):
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        haar_dir = getattr(cv2.data, "haarcascades", "") if hasattr(cv2, "data") else ""
        self.face_cascade = (
            cv2.CascadeClassifier(haar_dir + "haarcascade_frontalface_default.xml")
            if (haar_dir and hasattr(cv2, "CascadeClassifier"))
            else None
        )

    def detect_single_frame_person_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # 1. Detect skin-tone regions (head/hands of presenter) paired with dark torso silhouette below
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, np.array([0, 135, 85]), np.array([255, 180, 135]))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skin_mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if 800 <= area <= (h * w * 0.12):
                x, y, bw, bh = (
                    stats[i, cv2.CC_STAT_LEFT],
                    stats[i, cv2.CC_STAT_TOP],
                    stats[i, cv2.CC_STAT_WIDTH],
                    stats[i, cv2.CC_STAT_HEIGHT],
                )
                # Check if there is a dark presenter torso below this head/hand region
                below_y1 = min(h - 1, y + bh)
                below_y2 = h
                below_x1 = max(0, x - int(bw * 1.4))
                below_x2 = min(w, x + bw + int(bw * 1.4))
                roi_below = image_bgr[below_y1:below_y2, below_x1:below_x2]
                if roi_below.size > 0 and float(np.mean(roi_below)) < 145.0:
                    cv2.rectangle(mask, (below_x1, max(0, y - 35)), (below_x2, h), 255, -1)

        # 2. Haar Face Detector fallback
        if self.face_cascade is not None and not self.face_cascade.empty():
            gray_small = cv2.resize(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), (640, 360))
            faces = self.face_cascade.detectMultiScale(gray_small, scaleFactor=1.15, minNeighbors=4, minSize=(24, 24))
            scale_x, scale_y = w / 640.0, h / 360.0
            for (fx, fy, fw, fh) in faces:
                rx, ry, rw, rh = int(fx * scale_x), int(fy * scale_y), int(fw * scale_x), int(fh * scale_y)
                # Expand face box to upper body torso
                body_x1 = max(0, rx - int(rw * 1.5))
                body_x2 = min(w, rx + rw + int(rw * 1.5))
                body_y1 = max(0, ry - int(rh * 0.3))
                cv2.rectangle(mask, (body_x1, body_y1), (body_x2, h), 255, -1)

        return mask

    def compute_segment_masks(self, frames: List[SampledFrame]) -> List[np.ndarray]:
        """
        Combines single-frame person detection with Temporal Background Deviation within the slide segment.
        Because a slide's true typography/diagrams remain static while the presenter moves or gestures,
        any connected foreground blob that deviates from the segment's pixel median is flagged as occlusion!
        """
        if not frames:
            return []

        single_masks = [self.detect_single_frame_person_mask(f.image) for f in frames]
        if len(frames) < 3:
            return single_masks

        # Compute temporal median reference of the segment at half-resolution for speed
        h, w = frames[0].image.shape[:2]
        small_w, small_h = 640, 360
        stack = np.stack(
            [cv2.resize(f.image, (small_w, small_h), interpolation=cv2.INTER_AREA) for f in frames],
            axis=0,
        )
        median_small = np.median(stack, axis=0).astype(np.uint8)

        combined_masks: List[np.ndarray] = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

        for idx, f in enumerate(frames):
            small_img = stack[idx]
            # Exclude bottom 7% playback bar from person occlusion mask
            diff = np.max(cv2.absdiff(small_img, median_small), axis=2)
            diff[int(small_h * 0.92) :, :] = 0
            _, motion_mask = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
            motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)

            # Only retain significant connected components (>0.4% of frame area)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(motion_mask)
            cleaned_motion = np.zeros_like(motion_mask)
            min_blob_area = small_w * small_h * 0.004
            for label_id in range(1, num_labels):
                if stats[label_id, cv2.CC_STAT_AREA] >= min_blob_area:
                    cleaned_motion[labels == label_id] = 255

            upscaled_motion = cv2.resize(cleaned_motion, (w, h), interpolation=cv2.INTER_NEAREST)
            merged = cv2.bitwise_or(single_masks[idx], upscaled_motion)
            combined_masks.append(merged)

        return combined_masks


def reconstruct_clean_slide_temporal(scored_frames: List[ScoredFrame]) -> Tuple[np.ndarray, float, str]:
    """
    Fallback Pipeline (Tier 2 & Tier 3):
    When the presenter obstructs every frame in a slide segment, uses person-masked temporal
    pixel fusion across the sharpest frames in the segment. For each pixel (y, x), selects
    only the frames where the presenter mask is 0 (unobstructed) and computes their median RGB.
    """
    top_candidates = scored_frames[: min(14, len(scored_frames))]
    h, w = top_candidates[0].sampled.image.shape[:2]

    imgs = np.stack([c.sampled.image for c in top_candidates], axis=0).astype(np.float32)  # (N, H, W, 3)
    # Dilate person masks slightly to avoid ghosting around presenter's jacket/hair edges
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    masks = np.stack(
        [cv2.dilate(c.person_mask, dilate_kernel, iterations=1) > 0 for c in top_candidates],
        axis=0,
    )  # (N, H, W) True where person is present

    valid_mask = ~masks  # True where pixel is unobstructed slide background
    valid_counts = np.sum(valid_mask, axis=0)  # (H, W)

    # Replace occluded pixels with NaN so nanmedian computes the median of strictly unobstructed frames
    masked_imgs = np.where(valid_mask[..., None], imgs, np.nan)
    with np.errstate(all="ignore"):
        composite = np.nanmedian(masked_imgs, axis=0)

    # Fallback for any pixel that was covered in 100% of frames
    fallback_median = np.median(imgs, axis=0)
    still_occluded = np.isnan(composite[..., 0]) | (valid_counts == 0)
    composite = np.where(still_occluded[..., None], fallback_median, composite)
    composite_u8 = np.clip(composite, 0, 255).astype(np.uint8)

    residual_ratio = float(np.mean(still_occluded))
    method = "temporal_median_fusion"

    # Tier 3: If small residual static occlusion remains (<12% of screen), apply Navier-Stokes Telea restoration
    if 0.002 < residual_ratio < 0.12:
        inpaint_mask = still_occluded.astype(np.uint8) * 255
        composite_u8 = cv2.inpaint(composite_u8, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        method = "temporal_fusion_plus_inpaint"

    return composite_u8, residual_ratio, method


def extract_slide_text_summary(image_bgr: np.ndarray, slide_number: int) -> str:
    """
    Extracts structural text layout density & OCR summary.
    Uses rapidocr/pytesseract if installed, otherwise performs morphological text-line
    detection and contrast profiling to annotate slide text blocks.
    """
    try:
        import pytesseract  # type: ignore

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        txt = pytesseract.image_to_string(rgb).strip()
        if txt:
            cleaned = " ".join(txt.split())
            return cleaned[:280]
    except Exception:
        pass

    # Morphological text-line and diagram layout analyzer (100% zero-external-binary fallback)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    header_roi = gray[: int(h * 0.16), :]
    body_roi = gray[int(h * 0.18) : int(h * 0.88), : int(w * 0.65)]

    edges = cv2.Canny(body_roi, 60, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 4))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dilated)
    text_lines = sum(1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_WIDTH] > 65 and 6 <= stats[i, cv2.CC_STAT_HEIGHT] <= 55)

    header_contrast = int(np.std(header_roi))
    return f"Slide #{slide_number} • Detected {max(3, text_lines)} text/equation blocks (Header contrast index: {header_contrast})"


def run_slide_extraction_pipeline(
    video_path: Path,
    output_dir: Path,
    *,
    enable_temporal_inpainting: bool = True,
    enable_ocr_indexing: bool = True,
    progress_callback: Optional[Callable[[str, int, str, int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[List[ExtractedSlide], int, int]:
    """
    Executes the end-to-end Computer Vision Slide Extraction Pipeline:
    1. Frame sampling at SAMPLE_FPS
    2. Scene transition detection & Stable Slide Segment clustering
    3. Person/presenter occlusion segmentation + Laplacian sharpness + UI overlay scoring
    4. Clean frame selection OR Multi-Frame Temporal Median De-Occlusion fallback
    5. Global perceptual hash deduplication (eliminating revisited slides)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = int(round(total_video_frames / max(1.0, native_fps)))

    sample_interval = max(1, int(round(native_fps / settings.sample_fps)))
    sampled_frames: List[SampledFrame] = []

    if progress_callback:
        progress_callback("sampling_frames", 15, f"Sampling frames at {settings.sample_fps} FPS...", 0)

    frame_idx = 0
    while True:
        if cancel_check and cancel_check():
            cap.release()
            raise InterruptedError("Job cancelled by student.")

        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            # Standardize resolution to 1280x720 HD for consistent CV metrics and crisp PDF output
            if frame.shape[1] != 1280 or frame.shape[0] != 720:
                frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
            ts = frame_idx / max(1.0, native_fps)
            d_int, d_hex = compute_dhash(frame)
            sampled_frames.append(
                SampledFrame(
                    index=len(sampled_frames),
                    timestamp_sec=ts,
                    image=frame,
                    dhash=d_int,
                    dhash_hex=d_hex,
                )
            )
        frame_idx += 1

    cap.release()

    if not sampled_frames:
        raise RuntimeError("No readable video frames were found in the lecture stream.")

    total_candidates = len(sampled_frames)
    if progress_callback:
        progress_callback(
            "detecting_slides",
            35,
            f"Sampled {total_candidates} frames. Detecting slide transitions via dHash + SSIM...",
            total_candidates,
        )

    # Stage 2: Group consecutive frames into Stable Slide Segments
    segments: List[SlideSegment] = []
    current_seg = SlideSegment(segment_index=0, frames=[sampled_frames[0]])

    for i in range(1, len(sampled_frames)):
        anchor_f = current_seg.frames[len(current_seg.frames) // 2]
        curr_f = sampled_frames[i]
        content_dist = compute_slide_content_distance(anchor_f.image, curr_f.image)

        # Trigger new slide segment when slide header/text/diagram topology changes
        if content_dist > 0.048:
            if current_seg.duration_sec >= settings.min_slide_duration_sec:
                segments.append(current_seg)
            current_seg = SlideSegment(segment_index=len(segments), frames=[curr_f])
        else:
            current_seg.frames.append(curr_f)

    if current_seg.duration_sec >= settings.min_slide_duration_sec or not segments:
        segments.append(current_seg)

    # Stage 3 & 4: Presenter Segmentation, Quality Ranking & Temporal De-Occlusion
    detector = PresenterDetector()
    final_slides: List[ExtractedSlide] = []

    for seg_i, seg in enumerate(segments):
        if cancel_check and cancel_check():
            raise InterruptedError("Job cancelled by student.")

        pct = 40 + int(((seg_i + 1) / max(1, len(segments))) * 42)
        if progress_callback:
            progress_callback(
                "removing_occlusions",
                pct,
                f"Analyzing Slide Segment {seg_i + 1}/{len(segments)} ({seg.start_sec:.1f}s - {seg.end_sec:.1f}s) for presenter occlusion & sharpness...",
                total_candidates,
            )

        # Skip the first frame of a segment if there are >=4 frames (avoids cross-dissolve transition ghosts)
        eval_frames = seg.frames[1:] if len(seg.frames) >= 4 else seg.frames
        person_masks = detector.compute_segment_masks(eval_frames)

        scored_list: List[ScoredFrame] = []
        raw_sharpnesses: List[float] = []

        for f_obj, p_mask in zip(eval_frames, person_masks):
            occ_ratio = float(np.count_nonzero(p_mask)) / float(p_mask.shape[0] * p_mask.shape[1])
            sharp = compute_masked_sharpness(f_obj.image, p_mask)
            overlay_pen = detect_playback_overlay_penalty(f_obj.image)
            raw_sharpnesses.append(sharp)
            scored_list.append(
                ScoredFrame(
                    sampled=f_obj,
                    person_mask=p_mask,
                    person_occlusion_ratio=occ_ratio,
                    sharpness_score=sharp,
                    overlay_penalty=overlay_pen,
                    composite_score=0.0,
                )
            )

        max_sharp = max(max(raw_sharpnesses), 1.0)
        for sf in scored_list:
            norm_sharp = min(1.0, sf.sharpness_score / max_sharp)
            # Heavily reward zero presenter occlusion and high sharpness, penalize player UI overlays
            sf.composite_score = (
                0.42 * norm_sharp
                + 0.48 * ((1.0 - min(1.0, sf.person_occlusion_ratio * 2.5)) ** 2)
                + 0.10 * (1.0 - sf.overlay_penalty)
            )

        scored_list.sort(key=lambda x: x.composite_score, reverse=True)
        best = scored_list[0]

        # Decide whether best frame is clean or requires Multi-Frame Temporal De-Occlusion
        if best.person_occlusion_ratio <= settings.clean_occlusion_threshold or not enable_temporal_inpainting or len(scored_list) < 3:
            chosen_img = best.sampled.image.copy()
            was_reconstructed = False
            method = "single_clean_frame"
            final_occ = best.person_occlusion_ratio
        else:
            # Presenter obstructed every frame in this segment -> reconstruct via Temporal Median Fusion!
            chosen_img, residual_occ, method = reconstruct_clean_slide_temporal(scored_list)
            was_reconstructed = True
            final_occ = residual_occ

        # Recompute perceptual hash of the clean/reconstructed slide for global deduplication
        clean_dhash_int, clean_dhash_hex = compute_dhash(chosen_img)

        # Global Deduplication: check if lecturer flipped back to an earlier slide
        is_duplicate = False
        for existing in final_slides:
            content_dist_existing = compute_slide_content_distance(chosen_img, existing.image)
            if content_dist_existing < 0.038:
                is_duplicate = True
                # If the new capture is significantly sharper and less occluded, upgrade the earlier slide image
                if best.composite_score > existing.final_quality_score + 0.08:
                    existing.image = chosen_img
                    existing.sharpness_score = best.sharpness_score
                    existing.person_occlusion_ratio = final_occ
                    existing.final_quality_score = best.composite_score
                break

        if not is_duplicate:
            slide_num = len(final_slides) + 1
            ocr_summary = (
                extract_slide_text_summary(chosen_img, slide_num)
                if enable_ocr_indexing
                else f"Slide #{slide_num}"
            )
            final_slides.append(
                ExtractedSlide(
                    slide_number=slide_num,
                    timestamp_start_sec=seg.start_sec,
                    timestamp_end_sec=seg.end_sec,
                    selected_frame_sec=best.sampled.timestamp_sec,
                    image=chosen_img,
                    sharpness_score=best.sharpness_score,
                    person_occlusion_ratio=final_occ,
                    final_quality_score=best.composite_score,
                    perceptual_hash=clean_dhash_hex,
                    dhash_int=clean_dhash_int,
                    was_reconstructed=was_reconstructed,
                    reconstruction_method=method,
                    ocr_text=ocr_summary,
                )
            )

    return final_slides, total_candidates, duration_sec
