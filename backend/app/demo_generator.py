from __future__ import annotations

from pathlib import Path
from typing import Tuple
import cv2
import numpy as np


SLIDE_CONTENTS = [
    {
        "title": "Lecture 1: Introduction to Neural Networks",
        "subtitle": "CS-401 Deep Learning • Fall Semester",
        "bullets": [
            "1. Biological vs. Artificial Neurons (Perceptrons)",
            "2. Linear Separability & Activation Functions (ReLU, Sigmoid)",
            "3. Forward Propagation: Matrix Formulation y = f(Wx + b)",
            "4. Why Depth Matters: Hierarchical Feature Learning",
        ],
        "accent": (180, 85, 30),  # BGR
        "diagram": "network",
        "occlusion_mode": "walk_through_then_clear",
    },
    {
        "title": "Slide 2: Loss Functions & Gradient Descent",
        "subtitle": "Optimizing Non-Convex Objective Landscapes",
        "bullets": [
            "• Mean Squared Error (MSE): L = (1/N) * sum((y - y_hat)^2)",
            "• Cross-Entropy Loss for Multi-Class Classification",
            "• Stochastic Gradient Descent (SGD): theta <- theta - lr * grad(L)",
            "• Adaptive Optimizers: Adam, RMSProp & Cosine Annealing",
        ],
        "accent": (45, 130, 60),
        "diagram": "loss_curve",
        "occlusion_mode": "always_occluded_moving",
    },
    {
        "title": "Slide 3: Backpropagation & The Chain Rule",
        "subtitle": "Efficient Reverse-Mode Automatic Differentiation",
        "bullets": [
            "• Local Gradients: Caching intermediate activations during forward pass",
            "• Vanishing & Exploding Gradients in Deep Architectures",
            "• Residual Connections (ResNet): y = F(x) + x identity shortcut",
            "• Layer Normalization & Batch Normalization stability",
        ],
        "accent": (130, 50, 140),
        "diagram": "resblock",
        "occlusion_mode": "brief_appearance",
    },
    {
        "title": "Slide 4: Transformer Attention Mechanism",
        "subtitle": "Scaled Dot-Product Attention & Parallel Sequence Modeling",
        "bullets": [
            "• Queries (Q), Keys (K), and Values (V) Linear Projections",
            "• Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V",
            "• Multi-Head Attention captures diverse syntactic subspaces",
            "• Positional Encodings inject token order into permutation-invariant sets",
        ],
        "accent": (30, 110, 195),
        "diagram": "attention",
        "occlusion_mode": "clear_full",
    },
]


def _render_base_slide(slide_idx: int, spec: dict, width: int = 1280, height: int = 720) -> np.ndarray:
    canvas = np.full((height, width, 3), (248, 250, 252), dtype=np.uint8)

    # Top Header Banner
    accent = spec["accent"]
    cv2.rectangle(canvas, (0, 0), (width, 108), accent, thickness=-1)
    cv2.rectangle(canvas, (0, 108), (width, 114), (210, 180, 90), thickness=-1)

    # Slide Number Pill
    cv2.rectangle(canvas, (width - 170, 24), (width - 36, 78), (255, 255, 255), thickness=-1)
    cv2.putText(
        canvas,
        f"SLIDE 0{slide_idx + 1}",
        (width - 155, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        accent,
        2,
        cv2.LINE_AA,
    )

    # Header Text
    cv2.putText(
        canvas,
        spec["title"],
        (48, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        spec["subtitle"],
        (50, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (230, 238, 245),
        1,
        cv2.LINE_AA,
    )

    # Main Content Card
    cv2.rectangle(canvas, (44, 142), (790, 635), (255, 255, 255), thickness=-1)
    cv2.rectangle(canvas, (44, 142), (790, 635), (215, 222, 232), thickness=2)

    y_cursor = 205
    for bullet in spec["bullets"]:
        cv2.circle(canvas, (76, y_cursor - 8), 7, accent, thickness=-1)
        # Wrap long bullet into 2 lines if needed
        if len(bullet) > 48:
            part1 = bullet[:48]
            part2 = bullet[48:]
            cv2.putText(
                canvas,
                part1,
                (98, y_cursor),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (35, 42, 55),
                2,
                cv2.LINE_AA,
            )
            y_cursor += 34
            cv2.putText(
                canvas,
                part2.strip(),
                (98, y_cursor),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.66,
                (65, 75, 90),
                2,
                cv2.LINE_AA,
            )
            y_cursor += 68
        else:
            cv2.putText(
                canvas,
                bullet,
                (98, y_cursor),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (35, 42, 55),
                2,
                cv2.LINE_AA,
            )
            y_cursor += 92

    # Right-side Diagram Panel
    cv2.rectangle(canvas, (820, 142), (1236, 635), (255, 255, 255), thickness=-1)
    cv2.rectangle(canvas, (820, 142), (1236, 635), (215, 222, 232), thickness=2)
    cv2.putText(
        canvas,
        "Key Architecture / Visual",
        (850, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        accent,
        2,
        cv2.LINE_AA,
    )

    if spec["diagram"] == "network":
        layers = [3, 4, 2]
        xs = [885, 1025, 1165]
        coords = []
        for lx, count in zip(xs, layers):
            layer_pts = []
            for idx in range(count):
                ly = 265 + idx * 95
                layer_pts.append((lx, ly))
            coords.append(layer_pts)
        for i in range(len(coords) - 1):
            for p1 in coords[i]:
                for p2 in coords[i + 1]:
                    cv2.line(canvas, p1, p2, (160, 175, 195), 2, cv2.LINE_AA)
        for layer_pts in coords:
            for pt in layer_pts:
                cv2.circle(canvas, pt, 22, accent, -1, cv2.LINE_AA)
                cv2.circle(canvas, pt, 22, (255, 255, 255), 2, cv2.LINE_AA)

    elif spec["diagram"] == "loss_curve":
        pts = []
        for x in range(860, 1200, 8):
            t = (x - 1030) / 140.0
            y = int(530 - 240 * (t ** 2))
            pts.append([x, y])
        pts_arr = np.array(pts, dtype=np.int32)
        cv2.polylines(canvas, [pts_arr], False, accent, 4, cv2.LINE_AA)
        # Gradient descent steps
        step_x = [885, 930, 975, 1015, 1030]
        for sx in step_x:
            t = (sx - 1030) / 140.0
            sy = int(530 - 240 * (t ** 2))
            cv2.circle(canvas, (sx, sy), 9, (20, 40, 210), -1, cv2.LINE_AA)

    elif spec["diagram"] == "resblock":
        cv2.rectangle(canvas, (945, 245), (1110, 325), accent, 2)
        cv2.putText(canvas, "Weight Layer", (960, 292), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 30, 30), 2)
        cv2.rectangle(canvas, (945, 395), (1110, 475), accent, 2)
        cv2.putText(canvas, "ReLU + Norm", (962, 442), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 30, 30), 2)
        cv2.line(canvas, (1027, 325), (1027, 395), (40, 40, 40), 3)
        # Skip connection arc
        cv2.line(canvas, (1027, 215), (1165, 215), (30, 150, 60), 3)
        cv2.line(canvas, (1165, 215), (1165, 535), (30, 150, 60), 3)
        cv2.line(canvas, (1165, 535), (1027, 535), (30, 150, 60), 3)
        cv2.circle(canvas, (1027, 535), 20, (30, 150, 60), -1)
        cv2.putText(canvas, "+", (1018, 543), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    else:
        for i, label in enumerate(["Input Embedding", "Multi-Head Attn", "Feed Forward", "Softmax Output"]):
            top = 235 + i * 90
            cv2.rectangle(canvas, (905, top), (1155, top + 62), accent, -1)
            cv2.putText(
                canvas,
                label,
                (935, top + 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    # Bottom Lecture Footer
    cv2.putText(
        canvas,
        "Stanford / MIT OpenCourseWare Style Demo • Creative Commons CC-BY 4.0",
        (48, 668),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (120, 130, 145),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _draw_presenter(frame: np.ndarray, center_x: int, pointing: bool = True) -> np.ndarray:
    """Draws a realistic-toned silhouette + skin-toned head/hands of a lecturer standing in front of the slide."""
    out = frame.copy()
    h, w = out.shape[:2]

    # Torso / Jacket (dark charcoal blazer)
    torso_pts = np.array(
        [
            [center_x - 105, h],
            [center_x - 85, 420],
            [center_x + 85, 420],
            [center_x + 105, h],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(out, [torso_pts], (42, 46, 54), cv2.LINE_AA)

    # Collar / Shirt
    shirt_pts = np.array(
        [
            [center_x - 28, 420],
            [center_x + 28, 420],
            [center_x, 495],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(out, [shirt_pts], (220, 225, 235), cv2.LINE_AA)

    # Neck & Head (warm skin tone so HSV/person detector + motion detector lock on)
    cv2.rectangle(out, (center_x - 22, 375), (center_x + 22, 425), (138, 168, 212), -1)
    cv2.ellipse(out, (center_x, 332), (52, 64), 0, 0, 360, (145, 176, 220), -1, cv2.LINE_AA)
    # Hair & Glasses
    cv2.ellipse(out, (center_x, 295), (54, 34), 0, 180, 360, (30, 32, 38), -1, cv2.LINE_AA)
    cv2.rectangle(out, (center_x - 36, 320), (center_x - 8, 340), (30, 30, 30), 2)
    cv2.rectangle(out, (center_x + 8, 320), (center_x + 36, 340), (30, 30, 30), 2)

    if pointing:
        # Raised arm pointing at the slide equations
        arm_pts = np.array(
            [
                [center_x + 70, 445],
                [center_x + 185, 315],
                [center_x + 205, 335],
                [center_x + 85, 480],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(out, [arm_pts], (42, 46, 54), cv2.LINE_AA)
        cv2.circle(out, (center_x + 198, 322), 18, (145, 176, 220), -1, cv2.LINE_AA)

    return out


def _draw_playback_overlay(frame: np.ndarray, progress_ratio: float) -> np.ndarray:
    """Simulates a temporary video player progress bar and control icons at the bottom."""
    out = frame.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - 46), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)
    # Red progress bar
    bar_w = int((w - 40) * max(0.02, min(1.0, progress_ratio)))
    cv2.rectangle(out, (20, h - 38), (w - 20, h - 32), (90, 90, 90), -1)
    cv2.rectangle(out, (20, h - 38), (20 + bar_w, h - 32), (35, 35, 230), -1)
    return out


def generate_synthetic_lecture_video(output_path: Path) -> Tuple[str, int]:
    """
    Generates an 18-second (10 FPS, 1280x720) realistic lecture video containing:
    - 4 distinct slides + 1 revisited duplicate slide at the end (to test global deduplication)
    - Slide 1: Presenter walks in front of slide for the first half, then steps aside (tests clean frame selection)
    - Slide 2: Presenter NEVER leaves the slide, walking left-to-right across the text (tests Multi-Frame Temporal De-Occlusion!)
    - Slide 3: Blurry camera autofocus frames at start, then sharp frames
    - Slide 4: Clean slide with brief player overlay controls
    - Revisited Slide 1 at end (tests perceptual hash deduplication)
    """
    width, height = 1280, 720
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    base_slides = [_render_base_slide(i, spec, width, height) for i, spec in enumerate(SLIDE_CONTENTS)]

    # Sequence of (slide_index, duration_frames, mode)
    timeline = [
        (0, 40, "walk_then_clear"),       # 0.0s - 4.0s: Slide 1
        (1, 45, "always_occluded_walk"),  # 4.0s - 8.5s: Slide 2 (Requires Temporal Median Fusion!)
        (2, 40, "blur_then_sharp"),       # 8.5s - 12.5s: Slide 3
        (3, 35, "clean_with_ui"),         # 12.5s - 16.0s: Slide 4
        (0, 25, "duplicate_revisit"),     # 16.0s - 18.5s: Lecturer flips back to Slide 1 (Should be deduplicated!)
    ]

    total_frames = sum(f for _, f, _ in timeline)
    frame_counter = 0

    for slide_idx, num_frames, mode in timeline:
        base = base_slides[slide_idx]
        for f in range(num_frames):
            frame = base.copy()
            progress = frame_counter / max(1, total_frames)

            # Cross-dissolve transition on first 3 frames
            if f < 3 and slide_idx > 0:
                prev_base = base_slides[slide_idx - 1]
                alpha = (f + 1) / 4.0
                frame = cv2.addWeighted(base, alpha, prev_base, 1.0 - alpha, 0)

            elif mode == "walk_then_clear":
                # First 24 frames: presenter stands at x=420 covering bullet points; last 16 frames: steps off-screen
                if f < 24:
                    px = 340 + int(f * 12)
                    frame = _draw_presenter(frame, px, pointing=True)

            elif mode == "always_occluded_walk":
                # Presenter is ALWAYS on screen in every single frame, walking steadily from x=220 to x=1050
                px = 220 + int((f / max(1, num_frames - 1)) * 830)
                frame = _draw_presenter(frame, px, pointing=(f % 2 == 0))

            elif mode == "blur_then_sharp":
                # First 15 frames suffer heavy camera defocus blur; middle frames have presenter on right; last 12 are crystal clear
                if f < 15:
                    frame = cv2.GaussianBlur(frame, (25, 25), 7.5)
                elif f < 26:
                    frame = _draw_presenter(frame, 980, pointing=False)

            elif mode == "clean_with_ui":
                if f < 12:
                    frame = _draw_playback_overlay(frame, progress)

            elif mode == "duplicate_revisit":
                # Lecturer briefly revisits Slide 1
                pass

            writer.write(frame)
            frame_counter += 1

    writer.release()
    duration_sec = int(round(total_frames / fps))
    return "CS-401 Deep Learning Lecture: Neural Networks, Backprop & Transformers (Demo)", duration_sec


def generate_progressive_board_lecture_video(output_path: Path) -> Tuple[str, int]:
    """
    Generates a progressive chalkboard/whiteboard benchmark video where:
    - Board 1 (Dark Green Chalkboard, 0.0s - 6.0s):
      * 0.0s - 2.0s: Teacher writes Step 1 equation on upper-left (teacher off-screen)
      * 2.0s - 4.0s: Teacher writes Step 2 integral on middle-left while walking in
      * 4.0s - 6.0s: Teacher writes final Step 3 boxed theorem on right right before board wipe,
        WHILE teacher's body stands directly over Step 1 & Step 2!
    - Board 2 (Clean Summary Slide, 6.0s - 9.0s): Sudden board wipe / slide transition.
    Tests that SlideSnap:
      1. Does NOT split Board 1 into 3 partial slides as chalk accumulates.
      2. Captures Step 3 written right before the 6.0s transition.
      3. Simultaneously erases the teacher's body from Step 1 & Step 2 via reverse-chronological backfill!
    """
    width, height = 1280, 720
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Dark green university chalkboard background
    board_bg = np.full((height, width, 3), (42, 68, 46), dtype=np.uint8)
    cv2.rectangle(board_bg, (0, 0), (width, 96), (28, 48, 32), -1)
    cv2.putText(
        board_bg,
        "MATH-302 Chalkboard Derivation: Fundamental Theorem & Residue Proof",
        (42, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.88,
        (238, 246, 238),
        2,
        cv2.LINE_AA,
    )

    for f in range(60):  # 6.0 seconds progressive chalkboard
        canvas = board_bg.copy()
        # Step 1 present from t = 0.0s (x = 70..560, y = 180..290)
        cv2.putText(
            canvas,
            "Step 1: Let f(z) = P(z) / Q(z) on contour C",
            (70, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (245, 248, 240),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "oint_C f(z) dz = 2 * pi * i * sum(Res(f, z_k))",
            (70, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (245, 248, 240),
            2,
            cv2.LINE_AA,
        )

        # Step 2 written at t >= 2.0s (f >= 20) (x = 70..600, y = 360..460)
        if f >= 20:
            cv2.putText(
                canvas,
                "Step 2: Evaluate pole at z_0 = i (order 1):",
                (70, 385),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (245, 250, 210),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "Res(f, i) = lim_{z->i} (z - i) * f(z) = 1 / (2i)",
                (70, 445),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (245, 250, 210),
                2,
                cv2.LINE_AA,
            )

        # Step 3 written right before slide switch at t >= 4.0s (f >= 40) on right half (x = 680..1210)
        if f >= 40:
            cv2.rectangle(canvas, (680, 210), (1210, 470), (240, 245, 190), 3)
            cv2.putText(
                canvas,
                "FINAL RESULT (Pre-Transition):",
                (710, 275),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (255, 255, 220),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "Integral I = pi   [Q.E.D.]",
                (710, 365),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Teacher movement:
        # f < 16: teacher off-screen so Step 1 is clearly visible
        # 16 <= f < 36: teacher stands on right side (x=900) while Step 2 is visible on left
        # f >= 40 (when Step 3 is finally completed!): teacher stands directly at x=310 blocking Step 1 & Step 2!
        if 18 <= f < 36:
            canvas = _draw_presenter(canvas, 920, pointing=False)
        elif f >= 40:
            canvas = _draw_presenter(canvas, 310, pointing=True)

        writer.write(canvas)

    # Board 2: Clean Slide after Chalkboard Wipe (6.0s - 9.0s)
    slide2 = _render_base_slide(3, SLIDE_CONTENTS[3], width, height)
    for _ in range(30):
        writer.write(slide2)

    writer.release()
    return "MATH-302 Progressive Chalkboard Derivation Benchmark", 9

