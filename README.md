---
title: SlideSnap Lecture Slide Extractor
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# SlideSnap Studio v2.0 — Pre-Transition Lecture & Chalkboard Slide Extractor

**SlideSnap** converts recorded university lectures, chalkboard/whiteboard derivations, and YouTube OpenCourseWare videos into clean, chronological, unobstructed **Landscape A4 PDF slide decks**, Markdown study guides, and HD image archives.

## Key Innovations (v2.0)
- **Pre-Transition Full-Board Lock**: Tracks morphological chalk/ink stroke persistence (`extract_board_stroke_mask_small`) so progressive whiteboard and green-chalkboard derivations stay in *one* complete segment right up to the exact frame before a board wipe or slide switch.
- **Reverse-Chronological Teacher Erasure**: Anchors on the sharpest pre-transition peak-content frame and walks *backward* in time (`cum_valid <= 3`) strictly inside the segmented presenter silhouette—cleanly erasing walking teachers without erasing equations written moments earlier.
- **Interactive Before / After De-Occlusion Slider**: Exports raw obstructed vs. reconstructed clean slide pairs so students and judges can interactively inspect presenter removal in real time.
- **$O(1)$ Low-Memory Streaming Ring Buffer**: Uses `cap.grab()` frame skipping and compressed in-memory JPEG ring buffers (`~90 KB/frame`) with lazy disk-backed slides, keeping peak RAM under **110 MB** so multi-hour lectures run safely on 512 MB free cloud tiers (Render Free Plan).
- **3-Tier Quality & Speed Selector**: Choose **Turbo Draft (480p)** for ~3× faster cramming, **Balanced HD (720p)**, or **Ultra Crisp (1080p + CLAHE Contrast Boost)** for dense math derivations and code.

## Project Structure
- `frontend/` — **Next.js 15 + TypeScript + Tailwind CSS + Framer Motion** studio deployed on **Vercel**
- `backend/` — **FastAPI + OpenCV + ReportLab + SQLite** streaming CV worker deployed via Docker on **Render**

## Quick Start (Local)

### 1. Backend CV Worker (Port 7860)
```bash
cd backend
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 7860 --reload
```

### 2. Next.js Studio Frontend (Port 3000)
```bash
cd frontend
npm install
NEXT_PUBLIC_API_URL=http://127.0.0.1:7860 npm run dev
```

Made with care by **Arjun** 🎀
