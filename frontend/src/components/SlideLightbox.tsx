"use client";

import React, { useEffect, useState } from "react";
import { ChevronLeft, ChevronRight, ExternalLink, Sparkles, X } from "lucide-react";
import { SlideMetadata } from "@/types/slidesnap";
import { buildYouTubeTimestampUrl, formatTimestamp, resolveAssetUrl } from "@/lib/api";
import { BeforeAfterSlider } from "./BeforeAfterSlider";

interface SlideLightboxProps {
  slides: SlideMetadata[];
  activeIndex: number | null;
  sourceUrl: string | null;
  onClose: () => void;
  onSelectIndex: (idx: number) => void;
}

export function SlideLightbox({
  slides,
  activeIndex,
  sourceUrl,
  onClose,
  onSelectIndex,
}: SlideLightboxProps) {
  const [compareMode, setCompareMode] = useState(true);

  useEffect(() => {
    if (activeIndex === null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft" && activeIndex > 0) onSelectIndex(activeIndex - 1);
      if (e.key === "ArrowRight" && activeIndex < slides.length - 1) onSelectIndex(activeIndex + 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [activeIndex, slides.length, onClose, onSelectIndex]);

  if (activeIndex === null || !slides[activeIndex]) return null;
  const slide = slides[activeIndex];
  const hasBefore = Boolean(slide.occluded_image_url);
  const ytLink = buildYouTubeTimestampUrl(sourceUrl, slide.timestamp_start_sec);

  return (
    <div
      className="fixed inset-0 z-50 bg-slate-950/90 backdrop-blur-xl flex items-center justify-center p-3 sm:p-6"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-6xl rounded-2xl bg-slate-900/95 border border-white/15 shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Top Bar */}
        <div className="px-4 sm:px-6 py-3.5 bg-slate-950/80 border-b border-white/10 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <span className="px-3 py-1 rounded-lg bg-sky-500 text-slate-950 font-extrabold text-xs">
              Slide #{slide.slide_number} / {slides.length}
            </span>
            <span className="text-xs font-mono text-slate-300">
              {formatTimestamp(slide.timestamp_start_sec)} – {formatTimestamp(slide.timestamp_end_sec)}
            </span>
            {ytLink && (
              <a
                href={ytLink}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-xs text-sky-400 hover:text-sky-300 underline"
              >
                <span>Watch @ {formatTimestamp(slide.timestamp_start_sec)}</span>
                <ExternalLink className="w-3 h-3" />
              </a>
            )}
          </div>

          <div className="flex items-center gap-2">
            {hasBefore && (
              <button
                type="button"
                onClick={() => setCompareMode((v) => !v)}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition inline-flex items-center gap-1.5 ${
                  compareMode
                    ? "bg-amber-500/20 border-amber-400/50 text-amber-200"
                    : "bg-slate-800 border-white/10 text-slate-300"
                }`}
              >
                <Sparkles className="w-3.5 h-3.5" />
                {compareMode ? "Before/After Slider ON" : "Compare Before/After"}
              </button>
            )}
            <button
              type="button"
              onClick={onClose}
              className="p-2 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-200 transition"
              aria-label="Close inspector"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Main Image or Interactive Before/After Slider */}
        <div className="relative bg-black">
          {hasBefore && compareMode ? (
            <BeforeAfterSlider
              beforeUrl={resolveAssetUrl(slide.occluded_image_url)}
              afterUrl={resolveAssetUrl(slide.image_url)}
              slideNumber={slide.slide_number}
            />
          ) : (
            <div className="aspect-video w-full flex items-center justify-center">
              <img
                src={resolveAssetUrl(slide.image_url)}
                alt={`Slide ${slide.slide_number}`}
                className="max-h-[72vh] w-full object-contain"
              />
            </div>
          )}

          {activeIndex > 0 && (
            <button
              type="button"
              onClick={() => onSelectIndex(activeIndex - 1)}
              className="absolute left-3 top-1/2 -translate-y-1/2 z-20 p-2.5 rounded-full bg-slate-950/80 hover:bg-sky-500 hover:text-slate-950 border border-white/15 text-white transition shadow-xl"
            >
              <ChevronLeft className="w-5 h-5" />
            </button>
          )}
          {activeIndex < slides.length - 1 && (
            <button
              type="button"
              onClick={() => onSelectIndex(activeIndex + 1)}
              className="absolute right-3 top-1/2 -translate-y-1/2 z-20 p-2.5 rounded-full bg-slate-950/80 hover:bg-sky-500 hover:text-slate-950 border border-white/15 text-white transition shadow-xl"
            >
              <ChevronRight className="w-5 h-5" />
            </button>
          )}
        </div>

        {/* Footer Telemetry */}
        <div className="px-5 py-3.5 bg-slate-950/90 border-t border-white/10 flex flex-wrap items-center justify-between gap-3 text-xs">
          <p className="text-slate-300 truncate max-w-2xl">{slide.ocr_text}</p>
          <div className="flex items-center gap-4 font-mono text-slate-400">
            <span>
              Completeness:{" "}
              <strong className="text-sky-300">
                {((slide.content_completeness_score ?? 1) * 100).toFixed(0)}%
              </strong>
            </span>
            <span>
              Quality:{" "}
              <strong className="text-white">{(slide.final_quality_score * 100).toFixed(1)}%</strong>
            </span>
            <span>
              Sharpness: <strong className="text-white">{slide.sharpness_score.toFixed(0)}</strong>
            </span>
            <span>
              Residual Occlusion:{" "}
              <strong className="text-emerald-400">
                {(slide.person_occlusion_ratio * 100).toFixed(2)}%
              </strong>
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
