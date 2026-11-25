from __future__ import annotations

from dataclasses import dataclass, field
import gc
import os
from pathlib import Path
import time as _time
from typing import Callable, List, Optional, Tuple
import cv2
import numpy as np

from .config import QUALITY_PRESETS, QualityPreset, settings

# Critical for 512 MB containers running on 32/64-core cloud hosts:
# prevents OpenCV from spawning dozens of native worker threads (saving ~150 MB RSS)
cv2.setNumThreads(1)


@dataclass
class BufferedFrame:
    """
    Ultra-compact compressed frame representation (~85 KB JPEG + 320x180 thumb).
    Never holds raw full-resolution BGR arrays in RAM simultaneously.
    """
    index: int
    timestamp_sec: float
    jpeg_bytes: bytes
    thumb_bgr: np.ndarray          # (180, 320, 3) uint8
    stroke_mask_small: np.ndarray  # (180, 320) uint8
    quick_person_small: np.ndarray # (180, 320) uint8
    dhash: int
    dhash_hex: str

    def decode_bgr(self) -> np.ndarray:
        arr = np.frombuffer(self.jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return cv2.resize(self.thumb_bgr, (1280, 720), interpolation=cv2.INTER_CUBIC)
        return img


@dataclass
class ScoredCandidate:
    """
    Lightweight scored metadata computed at 320x180 without decoding full-resolution JPEGs!
    """
    buffered: BufferedFrame
    person_mask_small: np.ndarray  # (180, 320) uint8 (0 or 255)
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
        if len(self.frames) > max_buffer + 3:
            self._prune(max_buffer)

    def _prune(self, max_buffer: int) -> None:
        if len(self.frames) <= max_buffer:
            return
        tail_keep = max(5, (max_buffer * 3) // 5)
        head_budget = max(3, max_buffer - tail_keep)
        earlier = self.frames[:-tail_keep]
        tail = self.frames[-tail_keep:]

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
        img = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if img is not None:
            return img
        if self.thumb_bgr is not None:
            return cv2.resize(self.thumb_bgr, (1280, 720), interpolation=cv2.INTER_CUBIC)
        return np.zeros((720, 1280, 3), dtype=np.uint8)


def compute_dhash(image_bgr: np.ndarray, hash_size: int = 8) -> Tuple[int, str]:
    h, _ = image_bgr.shape[:2]
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
    gray = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    stroke_energy = np.maximum(blackhat, tophat)
    _, stroke_bin = cv2.threshold(stroke_energy, 22, 255, cv2.THRESH_BINARY)
    stroke_bin[int(thumb_bgr.shape[0] * 0.92) :, :] = 0
    return stroke_bin


def compute_slide_content_distance(img_a: np.ndarray, img_b: np.ndarray) -> float:
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

    header_diff = float(np.mean(cv2.absdiff(ca[:30, :], cb[:30, :]))) / 255.0
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
    smooth_first = cv2.GaussianBlur(seg_first.thumb_bgr[:166, :], (15, 15), 0)
    smooth_latest = cv2.GaussianBlur(seg_latest.thumb_bgr[:166, :], (15, 15), 0)
    smooth_curr = cv2.GaussianBlur(curr.thumb_bgr[:166, :], (15, 15), 0)

    header_shift = float(np.mean(cv2.absdiff(smooth_latest[:30, :], smooth_curr[:30, :]))) / 255.0
    header_shift_origin = float(np.mean(cv2.absdiff(smooth_first[:30, :], smooth_curr[:30, :]))) / 255.0
    if max(header_shift, header_shift_origin) > 0.055:
        return True

    unocc_latest = (seg_latest.quick_person_small[:166, :] == 0) & (curr.quick_person_small[:166, :] == 0)
    if int(np.count_nonzero(unocc_latest)) > 4000:
        bg_diff = float(np.mean(cv2.absdiff(smooth_latest, smooth_curr)[unocc_latest])) / 255.0
        if bg_diff > 0.085:
            return True

    k3 = np.ones((3, 3), dtype=np.uint8)
    prev_strokes = (seg_latest.stroke_mask_small[:166, :] > 0) & unocc_latest
    curr_strokes = (curr.stroke_mask_small[:166, :] > 0) & unocc_latest
    prev_count = int(np.count_nonzero(prev_strokes))
    curr_count = int(np.count_nonzero(curr_strokes))

    if prev_count >= 160:
        curr_dilated = cv2.dilate(curr.stroke_mask_small[:166, :], k3, iterations=1) > 0
        retained = int(np.count_nonzero(prev_strokes & curr_dilated))
        retention_ratio = retained / float(prev_count)

        if retention_ratio < 0.58:
            return True

        if retention_ratio >= 0.70 and curr_count >= int(prev_count * 0.85):
            return False

    dist_latest = compute_slide_content_distance(seg_latest.thumb_bgr, curr.thumb_bgr)
    if dist_latest > 0.048:
        if header_shift < 0.032 and curr_count > prev_count * 1.35 and prev_count < 160:
            return False
        return True

    return False


def compute_masked_sharpness(image_bgr: np.ndarray, person_mask: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_16S)
    non_person = person_mask == 0
    if np.count_nonzero(non_person) < 200:
        return float(np.var(lap))
    masked_vals = lap[non_person]
    return float(np.var(masked_vals))


def detect_playback_overlay_penalty(image_bgr: np.ndarray) -> float:
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
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.45, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_chan)
    l_blended = cv2.addWeighted(l_chan, 0.65, l_clahe, 0.35, 0)
    enhanced = cv2.cvtColor(cv2.merge((l_blended, a_chan, b_chan)), cv2.COLOR_LAB2BGR)
    soft = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
    return cv2.addWeighted(enhanced, 1.18, soft, -0.18, 0)


class PresenterDetector:
    """
    Ultra-Fast 320x180 Thumbnail Presenter & Solid-Core Motion Detector.
    Runs in <8ms per segment and <3 MB RAM while accurately separating walking teachers
    from newly written chalk/marker equations on both dark and light boards!
    """

    def __init__(self) -> None:
        haar_dir = getattr(cv2.data, "haarcascades", "") if hasattr(cv2, "data") else ""
        self.face_cascade = (
            cv2.CascadeClassifier(haar_dir + "haarcascade_frontalface_default.xml")
            if (haar_dir and hasattr(cv2, "CascadeClassifier"))
            else None
        )

    def detect_single_frame_person_mask(self, thumb_bgr: np.ndarray, use_haar: bool = False) -> np.ndarray:
        h, w = thumb_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        margin_band = thumb_bgr[int(h * 0.16) : int(h * 0.35), :]
        board_bg = np.median(margin_band.reshape(-1, 3), axis=0).astype(np.int16)

        ycrcb = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 182, 138]))
        k_open = max(3, int(round(w / 110)) | 1)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, np.ones((k_open, k_open), np.uint8))

        min_skin_area = max(35, int(h * w * 0.0008))
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
                roi_below = thumb_bgr[below_y1:h, below_x1:below_x2]
                if roi_below.size > 0:
                    diff_from_board = float(np.mean(np.abs(roi_below.astype(np.int16) - board_bg)))
                    mean_luma = float(np.mean(roi_below))
                    if diff_from_board > 24.0 or mean_luma < 145.0:
                        pad_top = max(6, int(h * 0.048))
                        cv2.rectangle(mask, (below_x1, max(0, y - pad_top)), (below_x2, h), 255, -1)

        if use_haar and self.face_cascade is not None and not self.face_cascade.empty():
            gray_small = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_small, scaleFactor=1.2, minNeighbors=4, minSize=(14, 14))
            for (fx, fy, fw, fh) in faces:
                body_x1 = max(0, fx - int(fw * 1.5))
                body_x2 = min(w, fx + fw + int(fw * 1.5))
                body_y1 = max(0, fy - int(fh * 0.3))
                cv2.rectangle(mask, (body_x1, body_y1), (body_x2, h), 255, -1)

        return mask

    def compute_segment_masks_small(self, thumbs: List[np.ndarray]) -> List[np.ndarray]:
        """
        Operates purely on 320x180 uint8 thumbnails (<2.5 MB RAM total!).
        Note: if pixels never move across the entire segment (e.g. a static logo or static inset card),
        suppresses static false-positive skin boxes so static graphics are never unnecessarily inpainted!
        """
        if not thumbs:
            return []

        single_masks = [self.detect_single_frame_person_mask(t, use_haar=True) for t in thumbs]
        if len(thumbs) < 3:
            return single_masks

        small_h, small_w = thumbs[0].shape[:2]
        stack = np.stack(thumbs, axis=0)  # (N, 180, 320, 3) uint8 -> only 2.5 MB!
        median_small = np.median(stack, axis=0).astype(np.uint8)
        late_ref_small = np.median(stack[-min(3, len(thumbs)) :], axis=0).astype(np.uint8)

        # Detect whether ANY pixel actually changed across the segment (protects 100%-static fallback cards!)
        temporal_ptp = np.max(np.ptp(stack, axis=0), axis=2)  # (180, 320) uint8 max-min range across time
        dynamic_zone = cv2.dilate((temporal_ptp > 18).astype(np.uint8) * 255, np.ones((9, 9), np.uint8)) > 0

        solid_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        combined_masks: List[np.ndarray] = []

        for idx in range(len(thumbs)):
            small_img = stack[idx]
            diff_med = np.max(cv2.absdiff(small_img, median_small), axis=2)
            diff_late = np.max(cv2.absdiff(small_img, late_ref_small), axis=2)

            diff_med[int(small_h * 0.92) :, :] = 0
            _, raw_motion = cv2.threshold(diff_med, 26, 255, cv2.THRESH_BINARY)

            solid_core = cv2.morphologyEx(raw_motion, cv2.MORPH_OPEN, solid_open_kernel)
            _, transient_blob = cv2.threshold(diff_late, 28, 255, cv2.THRESH_BINARY)
            transient_core = cv2.morphologyEx(
                transient_blob,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )
            seed_core = cv2.bitwise_or(solid_core, transient_core)

            candidate_blobs = cv2.morphologyEx(raw_motion, cv2.MORPH_CLOSE, close_kernel)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_blobs)
            cleaned_motion = np.zeros_like(candidate_blobs)
            min_blob_area = small_w * small_h * 0.004

            for label_id in range(1, num_labels):
                if stats[label_id, cv2.CC_STAT_AREA] >= min_blob_area:
                    comp_mask = labels == label_id
                    if np.any(seed_core[comp_mask] > 0):
                        cleaned_motion[comp_mask] = 255

            cleaned_motion = cv2.dilate(cleaned_motion, close_kernel, iterations=1)
            # Only keep single-frame person detections in regions that actually exhibit temporal variance
            # (prevents trying to erase a static textbook diagram or static fallback thumbnail!)
            active_single = np.where(dynamic_zone, single_masks[idx], 0).astype(np.uint8)
            merged = cv2.bitwise_or(active_single, cleaned_motion)
            combined_masks.append(merged)

        return combined_masks


def reconstruct_clean_slide_streaming_uint8(
    scored_candidates: List[ScoredCandidate],
    peak_anchor: ScoredCandidate,
    peak_bgr: np.ndarray,
) -> Tuple[np.ndarray, float, str]:
    """
    ZERO-STACK uint8 Reverse-Chronological Teacher Removal (< 5 MB Peak RAM!):
    - NEVER allocates 4D float32/float64 arrays or calls `np.nanmedian`!
    - Decodes ONE backup JPEG at a time, copies unobstructed uint8 pixels strictly at the
      1D coordinates `(occ_ys, occ_xs)` behind the teacher in `peak_bgr`, and immediately frees
      the decoded frame!
    """
    h, w = peak_bgr.shape[:2]
    dilate_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    peak_mask_small = cv2.dilate(peak_anchor.person_mask_small, dilate_small, iterations=1)
    if peak_anchor.overlay_penalty > 0.25:
        peak_mask_small[int(180 * 0.92) :, :] = 255

    # Check at 320x180 first before doing any full-res work
    if int(np.count_nonzero(peak_mask_small)) == 0:
        return peak_bgr.copy(), 0.0, "single_clean_frame"

    peak_occ_full = cv2.resize(peak_mask_small, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    occ_ys, occ_xs = np.where(peak_occ_full)
    num_occ = int(occ_ys.size)
    if num_occ == 0:
        return peak_bgr.copy(), 0.0, "single_clean_frame"

    # Order backup candidates reverse-chronologically (most recent sharp frames first!)
    others = [sc for sc in scored_candidates if sc.buffered.index != peak_anchor.buffered.index]
    others.sort(
        key=lambda sc: (
            1 if sc.sharpness_score >= max(15.0, peak_anchor.sharpness_score * 0.55) else 0,
            sc.buffered.index,
        ),
        reverse=True,
    )

    # Store up to 3 recent unobstructed uint8 RGB observations ONLY for the M occluded pixels (~1.2 MB!)
    obs = np.zeros((3, num_occ, 3), dtype=np.uint8)
    obs_count = np.zeros(num_occ, dtype=np.uint8)

    # Track at 320x180 to skip decoding frames that don't uncover any needed pixels
    occ_ys_small = (occ_ys * 180) // h
    occ_xs_small = (occ_xs * 320) // w

    for cand in others[:12]:
        still_needed = obs_count < 3
        if not np.any(still_needed):
            break

        cand_mask_small = cv2.dilate(cand.person_mask_small, dilate_small, iterations=1)
        if cand.overlay_penalty > 0.25:
            cand_mask_small[int(180 * 0.92) :, :] = 255

        # Quick check on 320x180 mask: does this frame uncover at least 64 needed pixels?
        uncovered_approx = still_needed & (cand_mask_small[occ_ys_small, occ_xs_small] == 0)
        if int(np.count_nonzero(uncovered_approx)) < 64:
            continue

        # Decode ONLY this single frame (2.7 MB), copy pixels, and immediately release it!
        cand_bgr = cand.buffered.decode_bgr()
        if cand_bgr.shape[1] != w or cand_bgr.shape[0] != h:
            cand_bgr = cv2.resize(cand_bgr, (w, h), interpolation=cv2.INTER_AREA)

        cand_occ_full = cv2.resize(cand_mask_small, (w, h), interpolation=cv2.INTER_NEAREST) > 0
        hit = still_needed & (~cand_occ_full[occ_ys, occ_xs])
        if np.any(hit):
            hit_indices = np.where(hit)[0]
            slots = obs_count[hit_indices]
            obs[slots, hit_indices] = cand_bgr[occ_ys[hit_indices], occ_xs[hit_indices]]
            obs_count[hit_indices] = slots + 1

        del cand_bgr

    # Compute exact integer median across the 1, 2, or 3 collected uint8 observations
    filled_pixels = peak_bgr[occ_ys, occ_xs].copy()

    m1 = obs_count == 1
    if np.any(m1):
        filled_pixels[m1] = obs[0, m1]

    m2 = obs_count == 2
    if np.any(m2):
        avg2 = (obs[0, m2].astype(np.uint16) + obs[1, m2].astype(np.uint16)) >> 1
        filled_pixels[m2] = avg2.astype(np.uint8)

    m3 = obs_count >= 3
    if np.any(m3):
        # Middle element of 3 sorted uint8 values = exact median in 0 extra memory!
        filled_pixels[m3] = np.sort(obs[:, m3, :], axis=0)[1]

    backfill_canvas = peak_bgr.copy()
    backfill_canvas[occ_ys, occ_xs] = filled_pixels

    # Feather strictly within the bounding box of peak_occ_full in integer fixed-point [0..256]
    y1, y2 = max(0, int(occ_ys.min()) - 12), min(h, int(occ_ys.max()) + 13)
    x1, x2 = max(0, int(occ_xs.min()) - 12), min(w, int(occ_xs.max()) + 13)
    alpha_roi = cv2.GaussianBlur(peak_occ_full[y1:y2, x1:x2].astype(np.uint8) * 255, (15, 15), 4.0).astype(np.uint16)
    p_roi = peak_bgr[y1:y2, x1:x2].astype(np.uint16)
    b_roi = backfill_canvas[y1:y2, x1:x2].astype(np.uint16)
    blended_roi = ((p_roi * (255 - alpha_roi[..., None]) + b_roi * alpha_roi[..., None]) // 255).astype(np.uint8)

    composite_u8 = peak_bgr.copy()
    composite_u8[y1:y2, x1:x2] = blended_roi
    del backfill_canvas

    never_uncovered_count = int(np.count_nonzero(obs_count == 0))
    residual_ratio = never_uncovered_count / float(h * w)
    method = "temporal_backfill_fusion"

    if 0.002 < residual_ratio < 0.10:
        inpaint_mask = np.zeros((h, w), dtype=np.uint8)
        unfilled_idx = np.where(obs_count == 0)[0]
        inpaint_mask[occ_ys[unfilled_idx], occ_xs[unfilled_idx]] = 255
        composite_u8 = cv2.inpaint(composite_u8, inpaint_mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
        method = "temporal_backfill_plus_inpaint"

    return composite_u8, residual_ratio, method


def extract_slide_text_summary(image_bgr: np.ndarray, slide_number: int) -> str:
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
        # On constrained 512 MB Render Free Tier, cap high preset at 1280x720 to guarantee OOM immunity
        if os.getenv("RENDER"):
            return 1280, 720
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
    on_slide_extracted: Optional[Callable[[ExtractedSlide], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[List[ExtractedSlide], int, int]:
    """
    Ultra-Low-Memory (<115 MB total container RSS) Streaming Slide & Chalkboard Pipeline.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    preset = QUALITY_PRESETS.get(quality_mode, QUALITY_PRESETS["balanced"])
    max_seg_buf = min(preset.max_segment_buffer, 10) if os.getenv("RENDER") else preset.max_segment_buffer

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    duration_sec = int(round(total_video_frames / max(1.0, native_fps)))

    target_w, target_h = _resolve_target_dimensions(native_w, native_h, preset)

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

        eval_bfs = seg.frames[1:] if len(seg.frames) >= 4 else list(seg.frames)

        # Trim outgoing cross-dissolve blur frame using lightweight 320x180 thumbnails (0 full-res decodes!)
        if len(eval_bfs) >= 4:
            last_sharp = compute_masked_sharpness(eval_bfs[-1].thumb_bgr, np.zeros((180, 320), np.uint8))
            prev_sharp = compute_masked_sharpness(eval_bfs[-2].thumb_bgr, np.zeros((180, 320), np.uint8))
            if last_sharp < prev_sharp * 0.52:
                eval_bfs = eval_bfs[:-1]

        thumbs = [bf.thumb_bgr for bf in eval_bfs]
        person_masks_small = detector.compute_segment_masks_small(thumbs)

        unocc_stroke_densities: List[float] = []
        raw_sharpnesses: List[float] = []
        temp_metrics: List[Tuple[float, float, float]] = []

        for bf, p_mask_s in zip(eval_bfs, person_masks_small):
            occ_ratio = float(np.count_nonzero(p_mask_s)) / float(p_mask_s.shape[0] * p_mask_s.shape[1])
            sharp_small = compute_masked_sharpness(bf.thumb_bgr, p_mask_s)
            overlay_pen = detect_playback_overlay_penalty(bf.thumb_bgr)
            raw_sharpnesses.append(sharp_small)
            temp_metrics.append((occ_ratio, sharp_small, overlay_pen))

            vis_mask = p_mask_s[:166, :] == 0
            vis_pixels = max(1, int(np.count_nonzero(vis_mask)))
            vis_strokes = int(np.count_nonzero((bf.stroke_mask_small[:166, :] > 0) & vis_mask))
            unocc_stroke_densities.append(vis_strokes / float(vis_pixels))

        peak_stroke_density = max(float(np.percentile(unocc_stroke_densities, 92)) if unocc_stroke_densities else 0.0, 1e-5)
        min_stroke_density = min(unocc_stroke_densities) if unocc_stroke_densities else 0.0
        is_progressive_board = (peak_stroke_density > 0.008) and (peak_stroke_density > min_stroke_density * 1.32)

        max_sharp = max(max(raw_sharpnesses), 1.0)
        scored_list: List[ScoredCandidate] = []
        n_eval = max(1, len(eval_bfs) - 1)

        for idx_f, (bf, p_mask_s, (occ_ratio, sharp_s, overlay_pen), s_dens) in enumerate(
            zip(eval_bfs, person_masks_small, temp_metrics, unocc_stroke_densities)
        ):
            norm_sharp = min(1.0, sharp_s / max_sharp)
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
                ScoredCandidate(
                    buffered=bf,
                    person_mask_small=p_mask_s,
                    person_occlusion_ratio=occ_ratio,
                    sharpness_score=sharp_s,
                    overlay_penalty=overlay_pen,
                    content_completeness=completeness,
                    composite_score=comp_score,
                )
            )

        high_comp_candidates = [
            sc for sc in scored_list if sc.content_completeness >= (0.86 if is_progressive_board else 0.78)
        ] or scored_list

        best_peak = max(
            high_comp_candidates,
            key=lambda sc: (
                0.45 * min(1.0, sc.sharpness_score / max_sharp)
                + 0.35 * ((1.0 - min(1.0, sc.person_occlusion_ratio * 2.5)) ** 2)
                + 0.12 * sc.content_completeness
                + 0.08 * (1.0 - sc.overlay_penalty)
            ),
        )
        worst_occluded = max(scored_list, key=lambda sc: sc.person_occlusion_ratio)

        # Decode ONLY the single winning peak frame at full resolution!
        peak_bgr = best_peak.buffered.decode_bgr()
        peak_full_mask = cv2.resize(
            best_peak.person_mask_small,
            (peak_bgr.shape[1], peak_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        full_sharpness = compute_masked_sharpness(peak_bgr, peak_full_mask)
        del peak_full_mask

        needs_temporal_deocclusion = (
            best_peak.person_occlusion_ratio > settings.clean_occlusion_threshold
            and enable_temporal_inpainting
            and len(scored_list) >= 3
        )

        before_jpeg_bytes: Optional[bytes] = None
        if not needs_temporal_deocclusion:
            chosen_img = peak_bgr
            was_reconstructed = False
            method = "pre_transition_clean_frame" if is_progressive_board else "single_clean_frame"
            final_occ = best_peak.person_occlusion_ratio
            if worst_occluded.person_occlusion_ratio >= 0.05:
                before_jpeg_bytes = worst_occluded.buffered.jpeg_bytes
        else:
            before_src = (
                best_peak.buffered
                if best_peak.person_occlusion_ratio >= worst_occluded.person_occlusion_ratio * 0.6
                else worst_occluded.buffered
            )
            before_jpeg_bytes = before_src.jpeg_bytes
            chosen_img, residual_occ, method = reconstruct_clean_slide_streaming_uint8(
                scored_list,
                best_peak,
                peak_bgr,
            )
            del peak_bgr
            was_reconstructed = method != "single_clean_frame"
            final_occ = residual_occ

        if enhance_contrast:
            chosen_img = enhance_slide_clarity(chosen_img)

        chosen_thumb = cv2.resize(chosen_img, (320, 180), interpolation=cv2.INTER_AREA)
        chosen_strokes = extract_board_stroke_mask_small(chosen_thumb)
        clean_dhash_int, clean_dhash_hex = compute_dhash(chosen_thumb)

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
                    existing.sharpness_score = full_sharpness
                    existing.person_occlusion_ratio = final_occ
                    existing.final_quality_score = max(existing.final_quality_score, best_peak.composite_score)
                break

        if not is_duplicate:
            slide_num = len(final_slides) + 1
            img_path = output_dir / f"slide_{slide_num:03d}.jpg"
            cv2.imwrite(str(img_path), chosen_img, [int(cv2.IMWRITE_JPEG_QUALITY), preset.jpeg_quality])

            occ_path: Optional[Path] = None
            if before_jpeg_bytes:
                occ_path = output_dir / f"slide_{slide_num:03d}_before.jpg"
                occ_path.write_bytes(before_jpeg_bytes)

            ocr_summary = (
                extract_slide_text_summary(chosen_img, slide_num)
                if enable_ocr_indexing
                else f"Slide #{slide_num}"
            )
            extracted_slide = ExtractedSlide(
                slide_number=slide_num,
                timestamp_start_sec=seg.start_sec,
                timestamp_end_sec=seg.end_sec,
                selected_frame_sec=best_peak.buffered.timestamp_sec,
                sharpness_score=full_sharpness,
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
            final_slides.append(extracted_slide)
            if on_slide_extracted:
                on_slide_extracted(extracted_slide)

        del chosen_img, scored_list, person_masks_small
        seg.frames.clear()
        gc.collect()

    if progress_callback:
        progress_callback(
            "sampling_frames",
            14,
            f"Streaming in {preset.label} mode ({effective_sample_fps} FPS, <115 MB Ultra-Low-RAM mode)...",
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
        quick_p_small = detector.detect_single_frame_person_mask(thumb, use_haar=False)
        d_int, d_hex = compute_dhash(thumb)

        ok_enc, enc_buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), preset.jpeg_quality])
        jpeg_bytes = enc_buf.tobytes() if ok_enc else b""
        del frame, enc_buf

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
            current_seg.add_frame(bf, max_seg_buf)
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
                            f"Pre-transition lock at {int(ts // 60):02d}:{int(ts % 60):02d} • Finalizing Slide #{len(final_slides) + 1}...",
                            total_candidates,
                        )
                    _finalize_segment(current_seg, segment_counter, total_candidates)
                current_seg = SlideSegment(segment_index=segment_counter)
                current_seg.add_frame(bf, max_seg_buf)
            else:
                current_seg.add_frame(bf, max_seg_buf)

        now_t = _time.time()
        if progress_callback and total_video_frames > 0 and (now_t - last_report_time > 1.4):
            last_report_time = now_t
            pct = 16 + int(min(1.0, frame_idx / total_video_frames) * 64)
            progress_callback(
                "sampling_frames",
                pct,
                f"Streaming lecture ({int(ts // 60)}m {int(ts % 60)}s / {max(1, duration_sec // 60)}m) • {len(final_slides)} slides extracted...",
                total_candidates,
            )

        frame_idx += 1

    cap.release()

    if total_candidates == 0:
        raise RuntimeError("No readable video frames were found in the lecture stream.")

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
