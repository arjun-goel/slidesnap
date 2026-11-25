"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { Sparkles, Eye, Wand2 } from "lucide-react";

interface BeforeAfterSliderProps {
  beforeUrl: string;
  afterUrl: string;
  slideNumber: number;
  className?: string;
}

export function BeforeAfterSlider({
  beforeUrl,
  afterUrl,
  slideNumber,
  className = "",
}: BeforeAfterSliderProps) {
  const [splitPct, setSplitPct] = useState(50);
  const [isDragging, setIsDragging] = useState(false);
  const [isAutoSweeping, setIsAutoSweeping] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const updatePosition = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const rel = ((clientX - rect.left) / Math.max(1, rect.width)) * 100;
    setSplitPct(Math.max(4, Math.min(96, rel)));
  }, []);

  useEffect(() => {
    if (!isDragging) return;
    const onMove = (e: MouseEvent) => updatePosition(e.clientX);
    const onTouch = (e: TouchEvent) => {
      if (e.touches[0]) updatePosition(e.touches[0].clientX);
    };
    const onUp = () => setIsDragging(false);

    window.addEventListener("mousemove", onMove);
    window.addEventListener("touchmove", onTouch, { passive: true });
    window.addEventListener("mouseup", onUp);
    window.addEventListener("touchend", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("touchmove", onTouch);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("touchend", onUp);
    };
  }, [isDragging, updatePosition]);

  useEffect(() => {
    if (!isAutoSweeping) return;
    let raf = 0;
    const start = performance.now();
    const duration = 2600;
    const animate = (now: number) => {
      const elapsed = now - start;
      if (elapsed >= duration) {
        setSplitPct(50);
        setIsAutoSweeping(false);
        return;
      }
      const progress = elapsed / duration;
      const wave = 50 + 42 * Math.sin(progress * Math.PI * 2);
      setSplitPct(wave);
      raf = requestAnimationFrame(animate);
    };
    raf = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(raf);
  }, [isAutoSweeping]);

  return (
    <div
      ref={containerRef}
      onMouseDown={(e) => {
        setIsAutoSweeping(false);
        setIsDragging(true);
        updatePosition(e.clientX);
      }}
      onTouchStart={(e) => {
        setIsAutoSweeping(false);
        setIsDragging(true);
        if (e.touches[0]) updatePosition(e.touches[0].clientX);
      }}
      className={`relative aspect-video w-full overflow-hidden bg-slate-950 select-none cursor-ew-resize group ${className}`}
    >
      {/* Base Layer: AFTER (Clean Reconstructed Slide) */}
      <img
        src={afterUrl}
        alt={`Slide ${slideNumber} Clean De-Occluded`}
        className="absolute inset-0 h-full w-full object-contain pointer-events-none"
        draggable={false}
      />

      {/* Clipped Layer: BEFORE (Obstructed Raw Frame with Teacher) */}
      <div
        className="absolute inset-0 overflow-hidden pointer-events-none"
        style={{ clipPath: `inset(0 ${100 - splitPct}% 0 0)` }}
      >
        <img
          src={beforeUrl}
          alt={`Slide ${slideNumber} Obstructed Raw Frame`}
          className="absolute inset-0 h-full w-full object-contain"
          draggable={false}
        />
      </div>

      {/* Labels */}
      <div className="absolute top-2.5 left-2.5 z-10 pointer-events-none">
        <span className="inline-flex items-center gap-1.5 rounded-full bg-rose-950/85 border border-rose-500/40 px-2.5 py-1 text-[10px] font-semibold tracking-wide text-rose-200 backdrop-blur-md shadow">
          <Eye className="w-3 h-3 text-rose-400" />
          BEFORE (Teacher Occluding)
        </span>
      </div>

      <div className="absolute top-2.5 right-2.5 z-10 pointer-events-none">
        <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-950/85 border border-emerald-500/40 px-2.5 py-1 text-[10px] font-semibold tracking-wide text-emerald-200 backdrop-blur-md shadow">
          <Sparkles className="w-3 h-3 text-emerald-400" />
          AFTER (De-Occluded Board)
        </span>
      </div>

      {/* Auto-Sweep Trigger Button */}
      <button
        type="button"
        onMouseDown={(e) => e.stopPropagation()}
        onTouchStart={(e) => e.stopPropagation()}
        onClick={(e) => {
          e.stopPropagation();
          setIsAutoSweeping((v) => !v);
        }}
        className="absolute bottom-2.5 right-2.5 z-20 inline-flex items-center gap-1.5 rounded-lg bg-slate-900/90 hover:bg-indigo-600/90 border border-white/15 px-2.5 py-1 text-[11px] font-medium text-slate-200 backdrop-blur-md transition shadow-lg"
      >
        <Wand2 className="w-3 h-3 text-amber-300" />
        {isAutoSweeping ? "Stop Sweep" : "Auto-Sweep"}
      </button>

      {/* Vertical Split Handle */}
      <div
        className="absolute top-0 bottom-0 z-10 w-0.5 bg-gradient-to-b from-sky-400 via-white to-emerald-400 shadow-[0_0_12px_rgba(56,189,248,0.9)] pointer-events-none"
        style={{ left: `${splitPct}%` }}
      >
        <div className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-slate-950/95 border-2 border-sky-400 shadow-lg shadow-sky-500/40 flex items-center justify-center text-white text-xs font-bold">
          ↔
        </div>
      </div>
    </div>
  );
}
