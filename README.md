---
title: SlideSnap Lecture Slide Extractor
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# SlideSnap — AI & Computer Vision Lecture Slide Extractor ($0 Student Cloud Edition)

**SlideSnap** extracts clean, unobstructed, chronological slide decks (`.pdf`) from lecture recordings and YouTube lectures.

## Features
- **Scene Transition & Slide Regime Clustering**: Uses 64-bit Perceptual Difference Hashing (`dHash`) and fast Structural Similarity (`SSIM`) to detect distinct slides while ignoring cross-dissolve transition animations and bottom player bars.
- **Presenter Detection & Frame Quality Ranking**: Segments the lecturer (face/skin/torso + intra-segment foreground motion) and scores non-occluded background sharpness (Variance of Laplacian) to pick the cleanest frame per slide.
- **Multi-Frame Temporal Median De-Occlusion Fallback**: When a lecturer walks back and forth and *never* steps completely off the slide, SlideSnap aligns frames within the slide interval and computes the person-masked temporal median across unobstructed pixels—cleanly erasing the moving presenter and badging the slide as `RECONSTRUCTED SLIDE` in both the UI and PDF.
- **Global Perceptual Deduplication**: Automatically detects when a lecturer flips back to an earlier slide and eliminates duplicate pages.
- **100% Free Cloud Deployment**: Ready for **Hugging Face Spaces Docker (Free 2 vCPU / 16 GB RAM)**, **Oracle Cloud Free Tier**, or local execution.

## Quick Start (Local)

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 7860 --reload
```

Then open **http://127.0.0.1:7860** in your browser and click **✨ Instant CV Demo** or paste a lecture URL!
