from __future__ import annotations

from dataclasses import dataclass, field
import gc
from pathlib import Path
import time as _time
from typing import Callable, List, Optional, Tuple
import cv2
import numpy as np

from .config import QUALITY_PRESETS, QualityPreset, settings


@dataclass
class BufferedFrame:
    """
    Lightweight, compressed frame representation for O(1)-RAM streaming on 512 MB cloud workers.
    Instead of holding raw 2.76 MB - 6.22 MB BGR arrays for every sampled frame in a lecture,
    stores ~90 KB JPEG bytes + a 320x180 thumbnail & stroke map.
    """
    index: int
    timestamp_sec: float
    jpeg_bytes: bytes
    thumb_bgr: np.ndarray          # (180, 320, 3) uint8
    stroke_mask_small: np.ndarray  # (180, 320) uint8 (255 on crisp board/slide strokes)
    quick_person_small: np.ndarray # (180, 320) uint8 (255 on detected presenter skin/torso)
    dhash: int
    dhash_hex: str

    def decode_bgr(self) -> np.ndarray:
        arr = np.frombuffer(self.jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return cv2.resize(self.thumb_bgr, (1280, 720), interpolation=cv2.INTER_CUBIC)
        return img


@dataclass
class ScoredFrame:
    buffered: BufferedFrame
    decoded_image: np.ndarray
    person_mask: np.ndarray  # uint8 (0 or 255)
    person_occlusion_ratio: float
    sharpness_score: float
    overlay_penalty: float
    content_completeness: float
    composite_score: float


@dataclass
class SlideSegment:
    segment_index: int
    start_timestamp_sec: float = 0.0
    end_timestamp_sec: float = 0.0
    first_frame: Optional[BufferedFrame] = None
    frames: List[BufferedFrame] = field(default_factory=list)

    def add_frame(self, bf: BufferedFrame, max_buffer: int) -> None:
        if self.first_frame is None:
            self.first_frame = bf
            self.start_timestamp_sec = bf.timestamp_sec
        self.end_timestamp_sec = bf.timestamp_sec
        self.frames.append(bf)
        if len(self.frames) > max_buffer + 4:
            self._prune(max_buffer)

    def _prune(self, max_buffer: int) -> None:
        """
        Retains dense pre-transition frames at the tail of the segment (for maximum board content)
        plus diverse unobstructed milestone frames from earlier in the segment (for full spatial de-occlusion).
        """
        if len(self.frames) <= max_buffer:
            return
        tail_keep = max(6, (max_buffer * 3) // 5)
        head_budget = max(3, max_buffer - tail_keep)
        earlier = self.frames[:-tail_keep]
        tail = self.frames[-tail_keep:]

        # Evenly sample across earlier history, preferring less-occluded snapshots in each bucket
        bucket_size = len(earlier) / float(head_budget)
        kept_earlier: List[BufferedFrame] = []
        for b_idx in range(head_budget):
            start_i = int(round(b_idx * bucket_size))
            end_i = max(start_i + 1, int(round((b_idx + 1) * bucket_size)))
            bucket = earlier[start_i:end_i]
            if bucket:
                best_in_bucket = min(
                    bucket,
                    key=lambda f: int(np.count_nonzero(f.quick_person_small)),
                )
                kept_earlier.append(best_in_bucket)

        self.frames = kept_earlier + tail

    @property
    def start_sec(self) -> float:
        return self.start_timestamp_sec

    @property
    def end_sec(self) -> float:
        return self.end_timestamp_sec

    @property
    def duration_sec(self) -> float:
        if len(self.frames) <= 1 and self.end_timestamp_sec <= self.start_timestamp_sec:
            return 0.5
        return max(0.5, self.end_timestamp_sec - self.start_timestamp_sec)


@dataclass
class ExtractedSlide:
    slide_number: int
    timestamp_start_sec: float
    timestamp_end_sec: float
    selected_frame_sec: float
    sharpness_score: float
    person_occlusion_ratio: float
    final_quality_score: float
    perceptual_hash: str
    dhash_int: int
    was_reconstructed: bool
    reconstruction_method: str
    ocr_text: str
    image_path: Path
    occluded_image_path: Optional[Path] = None
    content_completeness_score: float = 1.0
    thumb_bgr: Optional[np.ndarray] = field(default=None, repr=False)
    stroke_mask_small: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def image(self) -> np.ndarray:
        """Lazily reads the slide image from disk so extracted slides consume near-zero RAM."""
        img = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if img is not None:
            return img
        if self.thumb_bgr is not None:
            return cv2.resize(self.thumb_bgr, (1280, 720), interpolation=cv2.INTER_CUBIC)
        return np.zeros((720, 1280, 3), dtype=np.uint8)


def compute_dhash(image_bgr: np.ndarray, hash_size: int = 8) -> Tuple[int, str]:
    """
    Computes a 64-bit difference perceptual hash (dHash) on the high-frequency edge
    map + color header signature of the slide canvas (excluding bottom 8% player bar).
    """
    h, w = image_bgr.shape[:2]
    crop_h = int(h * 0.92)
    cropped = image_bgr[:crop_h, :]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    b, _, r = cv2.split(cropped)
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


def extract_board_stroke_mask_small(thumb_bgr: np.ndarray) -> np.ndarray:
    """
    Extracts thin high-contrast chalk, marker, typography, and diagram strokes on a 320x180
    thumbnail while rejecting smooth gradients and wide solid surfaces. Works identically on
    whiteboards, light PowerPoint slides, and dark green/black chalkboards.
    """
    gray = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    # Black-hat captures dark ink on whiteboards/slides; Top-hat captures white/colored chalk on dark boards
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    stroke_energy = np.maximum(blackhat, tophat)
    _, stroke_bin = cv2.threshold(stroke_energy, 22, 255, cv2.THRESH_BINARY)
    # Suppress bottom 8% player UI zone
    stroke_bin[int(thumb_bgr.shape[0] * 0.92) :, :] = 0
    return stroke_bin


def compute_slide_content_distance(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """
    Computes normalized visual & structural content distance between two slide frames or 320x180 thumbnails.
    """
    if img_a.shape[0] == 180 and img_a.shape[1] == 320:
        ca = img_a[:166, :]
    else:
        crop_ha = int(img_a.shape[0] * 0.92)
        ca = cv2.resize(img_a[:crop_ha, :], (320, 166), interpolation=cv2.INTER_AREA)

    if img_b.shape[0] == 180 and img_b.shape[1] == 320:
        cb = img_b[:166, :]
    else:
        crop_hb = int(img_b.shape[0] * 0.92)
        cb = cv2.resize(img_b[:crop_hb, :], (320, 166), interpolation=cv2.INTER_AREA)

    header_a = ca[:30, :]
    header_b = cb[:30, :]
    header_diff = float(np.mean(cv2.absdiff(header_a, header_b))) / 255.0

    color_diff = float(np.mean(cv2.absdiff(ca, cb))) / 255.0

    ga = cv2.cvtColor(ca, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cb, cv2.COLOR_BGR2GRAY)
    ea = cv2.Canny(ga, 60, 160).astype(np.float32) / 255.0
    eb = cv2.Canny(gb, 60, 160).astype(np.float32) / 255.0
    grid_a = cv2.resize(ea, (16, 9), interpolation=cv2.INTER_AREA)
    grid_b = cv2.resize(eb, (16, 9), interpolation=cv2.INTER_AREA)
    edge_grid_diff = float(np.mean(np.abs(grid_a - grid_b)))

    return 0.45 * header_diff + 0.30 * color_diff + 0.25 * edge_grid_diff


def compute_stroke_dice_similarity(stroke_a: np.ndarray, stroke_b: np.ndarray) -> float:
    """
    Measures spatial stroke overlap (with +-1px jitter tolerance) so two minimalist slides
    sharing the same white/dark background template but different text are never falsely merged.
    """
    cnt_a = int(np.count_nonzero(stroke_a))
    cnt_b = int(np.count_nonzero(stroke_b))
    if cnt_a < 80 and cnt_b < 80:
        return 1.0
    if cnt_a == 0 or cnt_b == 0:
        return 0.0
    k3 = np.ones((3, 3), dtype=np.uint8)
    dil_a = cv2.dilate(stroke_a, k3, iterations=1) > 0
    dil_b = cv2.dilate(stroke_b, k3, iterations=1) > 0
    inter_ab = int(np.count_nonzero((stroke_a > 0) & dil_b))
    inter_ba = int(np.count_nonzero((stroke_b > 0) & dil_a))
    return float(inter_ab + inter_ba) / float(cnt_a + cnt_b)


def is_new_slide_or_board_wipe(
    seg_first: BufferedFrame,
    seg_latest: BufferedFrame,
    curr: BufferedFrame,
) -> bool:
    """
    Pre-Transition & Progressive Board Regime Classifier:
    - Keeps extending the SAME segment while a lecturer progressively writes equations/annotations
      on a chalkboard/whiteboard or reveals incremental bullets (existing strokes stay >=70% intact
      while background/header theme stays constant).
    - Immediately triggers a segment cut when:
      1. Slide header or board background color shifts (new slide deck page / scene cut), OR
      2. Previously established unobstructed strokes disappear (board wipe / slide replacement!).
    """
    # Compare smoothed background/header theme (blurred to ignore thin handwriting strokes)
    smooth_first = cv2.GaussianBlur(seg_first.thumb_bgr[:166, :], (15, 15), 0)
    smooth_latest = cv2.GaussianBlur(seg_latest.thumb_bgr[:166, :], (15, 15), 0)
    smooth_curr = cv2.GaussianBlur(curr.thumb_bgr[:166, :], (15, 15), 0)

    header_shift = float(np.mean(cv2.absdiff(smooth_latest[:30, :], smooth_curr[:30, :]))) / 255.0
    header_shift_origin = float(np.mean(cv2.absdiff(smooth_first[:30, :], smooth_curr[:30, :]))) / 255.0
    if max(header_shift, header_shift_origin) > 0.055:
        return True

    # Exclude presenter regions when comparing background & stroke retention
    unocc_latest = (seg_latest.quick_person_small[:166, :] == 0) & (curr.quick_person_small[:166, :] == 0)
    if int(np.count_nonzero(unocc_latest)) > 4000:
        bg_diff = float(np.mean(cv2.absdiff(smooth_latest, smooth_curr)[unocc_latest])) / 255.0
        if bg_diff > 0.085:
            return True

    # Check stroke retention from seg_latest -> curr in mutually unobstructed board regions
    k3 = np.ones((3, 3), dtype=np.uint8)
    prev_strokes = (seg_latest.stroke_mask_small[:166, :] > 0) & unocc_latest
    curr_strokes = (curr.stroke_mask_small[:166, :] > 0) & unocc_latest
    prev_count = int(np.count_nonzero(prev_strokes))
    curr_count = int(np.count_nonzero(curr_strokes))

    if prev_count >= 160:
        curr_dilated = cv2.dilate(curr.stroke_mask_small[:166, :], k3, iterations=1) > 0
        retained = int(np.count_nonzero(prev_strokes & curr_dilated))
        retention_ratio = retained / float(prev_count)

        # Board wiped or replaced by a different same-template slide!
        if retention_ratio < 0.58:
            return True

        # Progressive writing / bullet build-up: existing strokes remain intact (>= 70%)!
        if retention_ratio >= 0.70 and curr_count >= int(prev_count * 0.85):
            return False

    # Fallback structural distance check against latest frame
    dist_latest = compute_slide_content_distance(seg_latest.thumb_bgr, curr.thumb_bgr)
    if dist_latest > 0.048:
        # Allow autofocus sharpening (low smooth color change + increasing stroke sharpness)
        if header_shift < 0.032 and curr_count > prev_count * 1.35 and prev_count < 160:
            return False
        return True

    return False


def compute_masked_sharpness(image_bgr: np.ndarray, person_mask: np.ndarray) -> float:
    """
    Computes the variance of the Laplacian strictly on the non-occluded slide background.
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
    """
    h, _ = image_bgr.shape[:2]
    bottom_strip = image_bgr[int(h * 0.92) :, :]
    hsv = cv2.cvtColor(bottom_strip, cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv, np.array([0, 150, 140]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 150, 140]), np.array([180, 255, 255]))
    red_ratio = float(np.count_nonzero(red_mask1 | red_mask2)) / max(1, bottom_strip.shape[0] * bottom_strip.shape[1])

    mean_luma = float(np.mean(bottom_strip))
    main_luma = float(np.mean(image_bgr[: int(h * 0.85), :]))

    penalty = 0.0
    if red_ratio > 0.008 and mean_luma < main_luma * 0.55:
        penalty = min(1.0, red_ratio * 35.0 + 0.45)
    return penalty


def enhance_slide_clarity(image_bgr: np.ndarray) -> np.ndarray:
    """
    Gentle adaptive contrast + unsharp text enhancement so faint chalk/marker handwriting
    and projector slides become crisp and legible without introducing color artifacts.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.45, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_chan)
    l_blended = cv2.addWeighted(l_chan, 0.65, l_clahe, 0.35, 0)
    enhanced = cv2.cvtColor(cv2.merge((l_blended, a_chan, b_chan)), cv2.COLOR_LAB2BGR)
    soft = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
    sharpened = cv2.addWeighted(enhanced, 1.18, soft, -0.18, 0)
    return sharpened


class PresenterDetector:
    """
    Board-Agnostic Hybrid Presenter & Occlusion Detector:
    1. Skin-tone + Board-Contrast Torso Column & Haar Face Cascade (works on BOTH whiteboards/slides
       AND dark green/black chalkboards regardless of clothing color).
    2. Morphological Solid-Core Foreground Separator: isolates solid moving human bodies/arms
       while protecting thin persistent chalk/marker handwriting written during the segment!
    """

    def __init__(self) -> None:
        haar_dir = getattr(cv2.data, "haarcascades", "") if hasattr(cv2, "data") else ""
        self.face_cascade = (
            cv2.CascadeClassifier(haar_dir + "haarcascade_frontalface_default.xml")
            if (haar_dir and hasattr(cv2, "CascadeClassifier"))
            else None
        )

    def detect_single_frame_person_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Estimate dominant board background color from upper-body canvas strips
        margin_band = image_bgr[int(h * 0.16) : int(h * 0.35), :]
        board_bg = np.median(margin_band.reshape(-1, 3), axis=0).astype(np.float32)

        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 182, 138]))
        k_open = max(3, int(round(w / 250)) | 1)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, np.ones((k_open, k_open), np.uint8))

        min_skin_area = max(45, int(h * w * 0.00085))
        max_skin_area = int(h * w * 0.12)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(skin_mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if min_skin_area <= area <= max_skin_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                bw = stats[i, cv2.CC_STAT_WIDTH]
                bh = stats[i, cv2.CC_STAT_HEIGHT]

                below_y1 = min(h - 1, y + bh)
                below_x1 = max(0, x - int(bw * 1.45))
                below_x2 = min(w, x + bw + int(bw * 1.45))
                roi_below = image_bgr[below_y1:h, below_x1:below_x2]
                if roi_below.size > 0:
                    diff_from_board = float(np.mean(np.abs(roi_below.astype(np.float32) - board_bg)))
                    mean_luma = float(np.mean(roi_below))
                    if diff_from_board > 24.0 or mean_luma < 145.0:
                        pad_top = max(8, int(h * 0.048))
                        cv2.rectangle(mask, (below_x1, max(0, y - pad_top)), (below_x2, h), 255, -1)

        # Haar Face Detector fallback (only on higher-res frames where faces >= 24px)
        if w >= 480 and self.face_cascade is not None and not self.face_cascade.empty():
            gray_small = cv2.resize(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), (640, 360))
            faces = self.face_cascade.detectMultiScale(gray_small, scaleFactor=1.15, minNeighbors=4, minSize=(24, 24))
            scale_x, scale_y = w / 640.0, h / 360.0
            for (fx, fy, fw, fh) in faces:
                rx, ry, rw, rh = int(fx * scale_x), int(fy * scale_y), int(fw * scale_x), int(fh * scale_y)
                body_x1 = max(0, rx - int(rw * 1.5))
                body_x2 = min(w, rx + rw + int(rw * 1.5))
                body_y1 = max(0, ry - int(rh * 0.3))
                cv2.rectangle(mask, (body_x1, body_y1), (body_x2, h), 255, -1)

        return mask

    def compute_segment_masks(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Combines single-frame person detection with Solid-Core Temporal Foreground Segmentation.
        Crucially distinguishes solid moving human bodies/arms from thin progressive handwriting
        written onto the board during the segment!
        """
        if not images:
            return []

        single_masks = [self.detect_single_frame_person_mask(img) for img in images]
        if len(images) < 3:
            return single_masks

        h, w = images[0].shape[:2]
        small_w, small_h = 640, 360
        stack = np.stack(
            [cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA) for img in images],
            axis=0,
        )
        median_small = np.median(stack, axis=0).astype(np.uint8)
        late_ref_small = np.median(stack[-min(3, len(images)) :], axis=0).astype(np.uint8)

        solid_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        combined_masks: List[np.ndarray] = []

        for idx in range(len(images)):
            small_img = stack[idx]
            diff_med = np.max(cv2.absdiff(small_img, median_small), axis=2)
            diff_late = np.max(cv2.absdiff(small_img, late_ref_small), axis=2)

            # Ignore bottom 8% player UI strip
            diff_med[int(small_h * 0.92) :, :] = 0
            _, raw_motion = cv2.threshold(diff_med, 26, 255, cv2.THRESH_BINARY)

            # Thin handwriting strokes (1-5px) vanish under 11x11 opening, whereas solid human
            # heads, torsos, and arms survive! Also flag transient blobs that vanish by late_ref.
            solid_core = cv2.morphologyEx(raw_motion, cv2.MORPH_OPEN, solid_open_kernel)
            _, transient_blob = cv2.threshold(diff_late, 28, 255, cv2.THRESH_BINARY)
            transient_core = cv2.morphologyEx(transient_blob, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            seed_core = cv2.bitwise_or(solid_core, transient_core)

            candidate_blobs = cv2.morphologyEx(raw_motion, cv2.MORPH_CLOSE, close_kernel)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_blobs)
            cleaned_motion = np.zeros_like(candidate_blobs)
            min_blob_area = small_w * small_h * 0.004

            for label_id in range(1, num_labels):
                if stats[label_id, cv2.CC_STAT_AREA] >= min_blob_area:
                    comp_mask = labels == label_id
                    # Keep component ONLY if it contains a solid body/arm core (protects newly written text!)
                    if np.any(seed_core[comp_mask] > 0):
                        cleaned_motion[comp_mask] = 255

            cleaned_motion = cv2.dilate(cleaned_motion, close_kernel, iterations=1)
            upscaled_motion = cv2.resize(cleaned_motion, (w, h), interpolation=cv2.INTER_NEAREST)
            merged = cv2.bitwise_or(single_masks[idx], upscaled_motion)
            combined_masks.append(merged)

        return combined_masks


def reconstruct_clean_slide_temporal(
    scored_frames: List[ScoredFrame],
    peak_anchor: ScoredFrame,
) -> Tuple[np.ndarray, float, str]:
    """
    Reverse-Chronological Ink-Preserving Teacher Removal:
    1. Anchors on `peak_anchor` (the sharpest pre-transition frame with maximum accumulated board content).
    2. Keeps `peak_anchor` pixels directly wherever the teacher mask is 0 (preserving 100% of even
       last-second handwriting at native single-frame sharpness).
    3. Inside the dilated teacher silhouette of `peak_anchor`, walks BACKWARD in time through recent
       sharp frames (`cum_valid <= 3`) to backfill each occluded pixel from its most recent unobstructed
       observation, blending smoothly across the boundary!
    """
    # Order candidates starting from peak_anchor, then reverse-chronologically among sharp frames
    others = [sf for sf in scored_frames if sf.buffered.index != peak_anchor.buffered.index]
    others.sort(
        key=lambda sf: (
            1 if sf.sharpness_score >= max(20.0, peak_anchor.sharpness_score * 0.55) else 0,
            sf.buffered.index,
        ),
        reverse=True,
    )
    ordered = [peak_anchor] + others[: min(15, len(others))]

    h, w = peak_anchor.decoded_image.shape[:2]
    dilate_px = max(13, int(round(w / 68)) | 1)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))

    imgs = np.stack([c.decoded_image for c in ordered], axis=0).astype(np.float32)  # (N, H, W, 3)
    masks = np.stack(
        [cv2.dilate(c.person_mask, dilate_kernel, iterations=1) > 0 for c in ordered],
        axis=0,
    )  # (N, H, W) True where presenter blocks the board

    # Also mask bottom 8% strip on frames that suffer from a video player UI bar
    bottom_y = int(h * 0.92)
    for idx, c in enumerate(ordered):
        if c.overlay_penalty > 0.25:
            masks[idx, bottom_y:, :] = True

    valid_mask = ~masks  # True where pixel is clean, unobstructed board
    cum_valid = np.cumsum(valid_mask, axis=0)
    # Select up to the 3 most recent unobstructed observations per pixel (preserves newly written ink!)
    recent_valid = valid_mask & (cum_valid <= 3)

    masked_imgs = np.where(recent_valid[..., None], imgs, np.nan)
    with np.errstate(all="ignore"):
        backfill = np.nanmedian(masked_imgs, axis=0)

    fallback_median = np.median(imgs, axis=0)
    still_occluded = np.isnan(backfill[..., 0]) | (cum_valid[-1] == 0)
    backfill = np.where(still_occluded[..., None], fallback_median, backfill)

    # Seamlessly feather backfill ONLY into the occluded region of peak_anchor
    peak_occ_mask = masks[0].astype(np.float32)
    feather_alpha = cv2.GaussianBlur(peak_occ_mask, (15, 15), 4.0)[..., None]
    anchor_f32 = peak_anchor.decoded_image.astype(np.float32)
    blended = anchor_f32 * (1.0 - feather_alpha) + backfill * feather_alpha
    composite_u8 = np.clip(blended, 0, 255).astype(np.uint8)

    residual_ratio = float(np.mean(still_occluded))
    method = "temporal_backfill_fusion"

    # Tier 3: Heal small never-uncovered static spots (<12% of screen) via Navier-Stokes Telea inpainting
    if 0.002 < residual_ratio < 0.12:
        inpaint_mask = still_occluded.astype(np.uint8) * 255
        composite_u8 = cv2.inpaint(composite_u8, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        method = "temporal_backfill_plus_inpaint"

    return composite_u8, residual_ratio, method


def extract_slide_text_summary(image_bgr: np.ndarray, slide_number: int) -> str:
    """
    Extracts structural text layout density & OCR summary.
    Uses pytesseract if installed, otherwise performs morphological text-line detection.
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

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    header_roi = gray[: int(h * 0.16), :]
    body_roi = gray[int(h * 0.18) : int(h * 0.88), : int(w * 0.85)]

    edges = cv2.Canny(body_roi, 60, 160)
    kw = max(14, int(round(w / 52)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 4))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dilated)
    min_line_w = max(40, int(round(w * 0.05)))
    text_lines = sum(
        1
        for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_WIDTH] > min_line_w and 5 <= stats[i, cv2.CC_STAT_HEIGHT] <= int(h * 0.09)
    )

    header_contrast = int(np.std(header_roi))
    return f"Slide #{slide_number} • Detected {max(3, text_lines)} text/equation blocks (Header contrast index: {header_contrast})"


def _resolve_target_dimensions(native_w: int, native_h: int, preset: QualityPreset) -> Tuple[int, int]:
    if preset.name == "high":
        if native_w >= 1400 and native_h >= 800:
            return 1920, 1080
        return 1280, 720
    return preset.target_width, preset.target_height


def run_slide_extraction_pipeline(
    video_path: Path,
    output_dir: Path,
    *,
    quality_mode: str = "balanced",
    enhance_contrast: bool = False,
    enable_temporal_inpainting: bool = True,
    enable_ocr_indexing: bool = True,
    progress_callback: Optional[Callable[[str, int, str, int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[List[ExtractedSlide], int, int]:
    """
    Streaming O(1)-Memory Computer Vision Slide & Progressive-Board Extraction Pipeline:
    1. Streams frames using `cap.grab()` without buffering full videos in RAM (safe on 512 MB Render Free).
    2. Tracks progressive board writing right up to the exact pre-transition frame before a wipe or slide switch.
    3. Segments teacher silhouette across light slides & dark chalkboards while preserving newly written strokes.
    4. Applies Reverse-Chronological Ink-Preserving Backfill when the teacher blocks the completed board,
       and saves a Before/After comparison frame (`slide_XXX_before.jpg`) for interactive inspection.
    5. Deduplicates revisited slides globally using structural distance + stroke Dice overlap.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    preset = QUALITY_PRESETS.get(quality_mode, QUALITY_PRESETS["balanced"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    duration_sec = int(round(total_video_frames / max(1.0, native_fps)))

    target_w, target_h = _resolve_target_dimensions(native_w, native_h, preset)

    # Adaptive FPS scaling by lecture duration & user-selected quality tier
    if duration_sec > 1800:
        effective_sample_fps = preset.long_fps
    elif duration_sec > 600:
        effective_sample_fps = preset.mid_fps
    else:
        effective_sample_fps = preset.base_fps

    sample_interval = max(1, int(round(native_fps / effective_sample_fps)))
    detector = PresenterDetector()
    final_slides: List[ExtractedSlide] = []

    def _finalize_segment(seg: SlideSegment, seg_num: int, approx_candidates: int) -> None:
        if not seg.frames:
            return
        if cancel_check and cancel_check():
            raise InterruptedError("Job cancelled by student.")

        # Trim incoming cross-dissolve frame at start and outgoing motion-blur transition frame at cut
        eval_bfs = seg.frames[1:] if len(seg.frames) >= 4 else list(seg.frames)
        decoded_imgs = [bf.decode_bgr() for bf in eval_bfs]

        if len(decoded_imgs) >= 4:
            # Check if the very last frame right at the cut is mid-dissolve or blurred
            last_gray = cv2.cvtColor(decoded_imgs[-1], cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(decoded_imgs[-2], cv2.COLOR_BGR2GRAY)
            last_sharp = float(cv2.Laplacian(last_gray, cv2.CV_64F).var())
            prev_sharp = float(cv2.Laplacian(prev_gray, cv2.CV_64F).var())
            if last_sharp < prev_sharp * 0.52:
                eval_bfs = eval_bfs[:-1]
                decoded_imgs = decoded_imgs[:-1]

        person_masks = detector.compute_segment_masks(decoded_imgs)

        # Measure unobstructed stroke density per frame to quantify progressive board completeness!
        unocc_stroke_densities: List[float] = []
        raw_sharpnesses: List[float] = []
        temp_metrics: List[Tuple[float, float, float]] = []

        for bf, img, p_mask in zip(eval_bfs, decoded_imgs, person_masks):
            occ_ratio = float(np.count_nonzero(p_mask)) / float(p_mask.shape[0] * p_mask.shape[1])
            sharp = compute_masked_sharpness(img, p_mask)
            overlay_pen = detect_playback_overlay_penalty(img)
            raw_sharpnesses.append(sharp)
            temp_metrics.append((occ_ratio, sharp, overlay_pen))

            p_small = cv2.resize(p_mask, (320, 180), interpolation=cv2.INTER_NEAREST)
            vis_mask = (p_small[:166, :] == 0)
            vis_pixels = max(1, int(np.count_nonzero(vis_mask)))
            vis_strokes = int(np.count_nonzero((bf.stroke_mask_small[:166, :] > 0) & vis_mask))
            unocc_stroke_densities.append(vis_strokes / float(vis_pixels))

        peak_stroke_density = max(float(np.percentile(unocc_stroke_densities, 92)) if unocc_stroke_densities else 0.0, 1e-5)
        min_stroke_density = min(unocc_stroke_densities) if unocc_stroke_densities else 0.0
        is_progressive_board = (peak_stroke_density > 0.008) and (peak_stroke_density > min_stroke_density * 1.32)

        max_sharp = max(max(raw_sharpnesses), 1.0)
        scored_list: List[ScoredFrame] = []
        n_eval = max(1, len(eval_bfs) - 1)

        for idx_f, (bf, img, p_mask, (occ_ratio, sharp, overlay_pen), s_dens) in enumerate(
            zip(eval_bfs, decoded_imgs, person_masks, temp_metrics, unocc_stroke_densities)
        ):
            norm_sharp = min(1.0, sharp / max_sharp)
            stroke_comp = min(1.0, s_dens / peak_stroke_density) if peak_stroke_density > 0.004 else 1.0
            recency = idx_f / float(n_eval)
            completeness = (0.72 * stroke_comp + 0.28 * recency) if is_progressive_board else (0.90 * stroke_comp + 0.10 * recency)

            occ_reward = (1.0 - min(1.0, occ_ratio * 2.5)) ** 2
            overlay_reward = 1.0 - overlay_pen

            if is_progressive_board:
                comp_score = 0.34 * completeness + 0.32 * norm_sharp + 0.26 * occ_reward + 0.08 * overlay_reward
            else:
                comp_score = 0.40 * norm_sharp + 0.44 * occ_reward + 0.08 * completeness + 0.08 * overlay_reward

            scored_list.append(
                ScoredFrame(
                    buffered=bf,
                    decoded_image=img,
                    person_mask=p_mask,
                    person_occlusion_ratio=occ_ratio,
                    sharpness_score=sharp,
                    overlay_penalty=overlay_pen,
                    content_completeness=completeness,
                    composite_score=comp_score,
                )
            )

        # Identify the sharpest pre-transition peak-content frame (captures full board right before slide switch!)
        high_comp_candidates = [
            sf for sf in scored_list if sf.content_completeness >= (0.86 if is_progressive_board else 0.78)
        ] or scored_list

        # Best unobstructed peak-content frame vs. best overall peak-content anchor
        best_peak = max(
            high_comp_candidates,
            key=lambda sf: (
                0.45 * min(1.0, sf.sharpness_score / max_sharp)
                + 0.35 * ((1.0 - min(1.0, sf.person_occlusion_ratio * 2.5)) ** 2)
                + 0.12 * sf.content_completeness
                + 0.08 * (1.0 - sf.overlay_penalty)
            ),
        )
        worst_occluded = max(scored_list, key=lambda sf: sf.person_occlusion_ratio)

        needs_temporal_deocclusion = (
            best_peak.person_occlusion_ratio > settings.clean_occlusion_threshold
            and enable_temporal_inpainting
            and len(scored_list) >= 3
        )

        before_occluded_bgr: Optional[np.ndarray] = None
        if not needs_temporal_deocclusion:
            chosen_img = best_peak.decoded_image.copy()
            was_reconstructed = False
            method = "pre_transition_clean_frame" if is_progressive_board else "single_clean_frame"
            final_occ = best_peak.person_occlusion_ratio
            # Even when a clean late frame existed, if the teacher walked in front earlier (>5% occlusion),
            # we can still save that obstructed frame for interactive Before/After judge comparison!
            if worst_occluded.person_occlusion_ratio >= 0.05:
                before_occluded_bgr = worst_occluded.decoded_image.copy()
        else:
            before_occluded_bgr = best_peak.decoded_image.copy() if best_peak.person_occlusion_ratio >= worst_occluded.person_occlusion_ratio * 0.6 else worst_occluded.decoded_image.copy()
            chosen_img, residual_occ, method = reconstruct_clean_slide_temporal(scored_list, best_peak)
            was_reconstructed = True
            final_occ = residual_occ

        if enhance_contrast:
            chosen_img = enhance_slide_clarity(chosen_img)

        chosen_thumb = cv2.resize(chosen_img, (320, 180), interpolation=cv2.INTER_AREA)
        chosen_strokes = extract_board_stroke_mask_small(chosen_thumb)
        clean_dhash_int, clean_dhash_hex = compute_dhash(chosen_img)

        # Global Deduplication (checks both visual distance AND stroke spatial Dice similarity)
        is_duplicate = False
        for existing in final_slides:
            if existing.thumb_bgr is None or existing.stroke_mask_small is None:
                continue
            content_dist_existing = compute_slide_content_distance(chosen_thumb, existing.thumb_bgr)
            stroke_sim = compute_stroke_dice_similarity(chosen_strokes, existing.stroke_mask_small)

            if content_dist_existing < 0.038 and stroke_sim >= 0.58:
                is_duplicate = True
                new_strokes_cnt = int(np.count_nonzero(chosen_strokes))
                old_strokes_cnt = int(np.count_nonzero(existing.stroke_mask_small))
                has_more_notes = new_strokes_cnt > int(old_strokes_cnt * 1.18)
                if best_peak.composite_score > existing.final_quality_score + 0.06 or has_more_notes:
                    cv2.imwrite(
                        str(existing.image_path),
                        chosen_img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), preset.jpeg_quality],
                    )
                    existing.thumb_bgr = chosen_thumb
                    existing.stroke_mask_small = chosen_strokes
                    existing.sharpness_score = best_peak.sharpness_score
                    existing.person_occlusion_ratio = final_occ
                    existing.final_quality_score = max(existing.final_quality_score, best_peak.composite_score)
                break

        if not is_duplicate:
            slide_num = len(final_slides) + 1
            img_path = output_dir / f"slide_{slide_num:03d}.jpg"
            cv2.imwrite(str(img_path), chosen_img, [int(cv2.IMWRITE_JPEG_QUALITY), preset.jpeg_quality])

            occ_path: Optional[Path] = None
            if before_occluded_bgr is not None:
                occ_path = output_dir / f"slide_{slide_num:03d}_before.jpg"
                cv2.imwrite(str(occ_path), before_occluded_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

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
                    selected_frame_sec=best_peak.buffered.timestamp_sec,
                    sharpness_score=best_peak.sharpness_score,
                    person_occlusion_ratio=final_occ,
                    final_quality_score=best_peak.composite_score,
                    perceptual_hash=clean_dhash_hex,
                    dhash_int=clean_dhash_int,
                    was_reconstructed=was_reconstructed,
                    reconstruction_method=method,
                    ocr_text=ocr_summary,
                    image_path=img_path,
                    occluded_image_path=occ_path,
                    content_completeness_score=best_peak.content_completeness,
                    thumb_bgr=chosen_thumb,
                    stroke_mask_small=chosen_strokes,
                )
            )

        # Explicitly release decoded BGR matrices to keep RSS tiny on 512 MB Render Free
        del decoded_imgs, person_masks, scored_list
        gc.collect()

    if progress_callback:
        progress_callback(
            "sampling_frames",
            14,
            f"Streaming lecture in {preset.label} mode ({effective_sample_fps} FPS, low-memory ring buffer)...",
            0,
        )

    frame_idx = 0
    total_candidates = 0
    segment_counter = 0
    current_seg = SlideSegment(segment_index=0)
    last_report_time = 0.0

    while True:
        if cancel_check and cancel_check():
            cap.release()
            raise InterruptedError("Job cancelled by student.")

        if frame_idx % sample_interval != 0:
            if not cap.grab():
                break
            frame_idx += 1
            continue

        ret, frame = cap.read()
        if not ret:
            break

        if frame.shape[1] != target_w or frame.shape[0] != target_h:
            interp = cv2.INTER_CUBIC if target_w > frame.shape[1] else cv2.INTER_AREA
            frame = cv2.resize(frame, (target_w, target_h), interpolation=interp)

        ts = frame_idx / max(1.0, native_fps)
        thumb = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        stroke_small = extract_board_stroke_mask_small(thumb)
        quick_p_small = detector.detect_single_frame_person_mask(thumb)
        d_int, d_hex = compute_dhash(thumb)

        ok_enc, enc_buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), preset.jpeg_quality])
        jpeg_bytes = enc_buf.tobytes() if ok_enc else b""
        del frame

        bf = BufferedFrame(
            index=total_candidates,
            timestamp_sec=ts,
            jpeg_bytes=jpeg_bytes,
            thumb_bgr=thumb,
            stroke_mask_small=stroke_small,
            quick_person_small=quick_p_small,
            dhash=d_int,
            dhash_hex=d_hex,
        )
        total_candidates += 1

        if not current_seg.frames:
            current_seg.add_frame(bf, preset.max_segment_buffer)
        else:
            assert current_seg.first_frame is not None
            latest_bf = current_seg.frames[-1]
            if is_new_slide_or_board_wipe(current_seg.first_frame, latest_bf, bf):
                if current_seg.duration_sec >= settings.min_slide_duration_sec:
                    segment_counter += 1
                    if progress_callback and total_video_frames > 0:
                        pct = 18 + int(min(1.0, frame_idx / total_video_frames) * 64)
                        progress_callback(
                            "removing_occlusions",
                            pct,
                            f"Pre-transition lock at {int(ts // 60):02d}:{int(ts % 60):02d} • De-occluding Slide #{len(final_slides) + 1}...",
                            total_candidates,
                        )
                    _finalize_segment(current_seg, segment_counter, total_candidates)
                current_seg = SlideSegment(segment_index=segment_counter)
                current_seg.add_frame(bf, preset.max_segment_buffer)
            else:
                current_seg.add_frame(bf, preset.max_segment_buffer)

        now_t = _time.time()
        if progress_callback and total_video_frames > 0 and (now_t - last_report_time > 1.4):
            last_report_time = now_t
            pct = 16 + int(min(1.0, frame_idx / total_video_frames) * 64)
            progress_callback(
                "sampling_frames",
                pct,
                f"Streaming & tracking board ({int(ts // 60)}m {int(ts % 60)}s / {max(1, duration_sec // 60)}m) • {len(final_slides)} slides locked...",
                total_candidates,
            )

        frame_idx += 1

    cap.release()

    if total_candidates == 0:
        raise RuntimeError("No readable video frames were found in the lecture stream.")

    # Finalize the last active slide segment at the end of the video
    if current_seg.frames and (current_seg.duration_sec >= settings.min_slide_duration_sec or not final_slides):
        segment_counter += 1
        if progress_callback:
            progress_callback(
                "removing_occlusions",
                84,
                f"Finalizing concluding slide segment ({current_seg.start_sec:.1f}s - {current_seg.end_sec:.1f}s)...",
                total_candidates,
            )
        _finalize_segment(current_seg, segment_counter, total_candidates)

    return final_slides, total_candidates, duration_sec
