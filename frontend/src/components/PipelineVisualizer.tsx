"use client";

import React from "react";
import { motion } from "framer-motion";
import {
  CheckCircle2,
  Cpu,
  Download,
  FileText,
  Film,
  Layers,
  Loader2,
  RotateCcw,
  ScanEye,
  Sparkles,
  XCircle,
} from "lucide-react";
import { JobResponse, JobStatus } from "@/types/slidesnap";
import { resolveAssetUrl } from "@/lib/api";

interface PipelineVisualizerProps {
  job: JobResponse | null;
  uploadPct: number | null;
  onCancel: (jobId: string) => void;
  onRetry: (jobId: string) => void;
  onTryDemo: () => void;
}

const STAGES: { key: JobStatus; label: string; desc: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { key: "ingesting", label: "Stream Ingest", desc: "DASH video-only MP4 / upload buffer", icon: Film },
  { key: "sampling_frames", label: "O(1) Ring Sampler", desc: "Fast cap.grab() low-RAM stream", icon: Cpu },
  { key: "detecting_slides", label: "Pre-Transition Lock", desc: "Stroke-persistence board tracking", icon: Layers },
  { key: "removing_occlusions", label: "Teacher Erasure", desc: "Reverse-chronological pixel backfill", icon: ScanEye },
  { key: "compiling_pdf", label: "A4 Landscape PDF", desc: "Vector metadata & OCR indexing", icon: FileText },
];

const STATUS_ORDER: Record<JobStatus, number> = {
  queued: 0,
  ingesting: 1,
  sampling_frames: 2,
  detecting_slides: 3,
  removing_occlusions: 4,
  compiling_pdf: 5,
  completed: 6,
  failed: -1,
  cancelled: -2,
};

export function PipelineVisualizer({
  job,
  uploadPct,
  onCancel,
  onRetry,
  onTryDemo,
}: PipelineVisualizerProps) {
  const currentRank = job ? STATUS_ORDER[job.status] ?? 0 : -99;
  const isRunning = job && !["completed", "failed", "cancelled"].includes(job.status);

  return (
    <div className="glass-panel rounded-2xl p-6 border border-sky-500/25 shadow-2xl relative overflow-hidden">
      <div className="absolute -top-24 -right-24 w-48 h-48 rounded-full bg-sky-500/10 blur-3xl pointer-events-none" />

      <div className="flex items-center justify-between gap-2 mb-3">
        <span className="inline-flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-widest text-sky-400">
          <Sparkles className="w-3.5 h-3.5" />
          Live CV Worker Telemetry
        </span>
        <span
          className={`text-xs font-mono px-2.5 py-0.5 rounded-full border ${
            !job
              ? "bg-slate-900/90 border-slate-700 text-slate-400"
              : job.status === "completed"
              ? "bg-emerald-500/15 border-emerald-500/40 text-emerald-300"
              : job.status === "failed"
              ? "bg-rose-500/15 border-rose-500/40 text-rose-300"
              : job.status === "cancelled"
              ? "bg-slate-800 border-slate-600 text-slate-300"
              : "bg-sky-500/15 border-sky-400/40 text-sky-200"
          }`}
        >
          {uploadPct !== null ? `uploading ${uploadPct}%` : job ? job.status.replace(/_/g, " ") : "standby"}
        </span>
      </div>

      <h2 className="text-base sm:text-lg font-bold text-white truncate">
        {uploadPct !== null
          ? "Streaming Lecture Video to CV Worker..."
          : job
          ? job.video_title
          : "Ready to Extract Clean Study Slides"}
      </h2>
      <p className="text-xs text-slate-400 mt-1 min-h-[2rem] leading-relaxed">
        {uploadPct !== null
          ? `Direct chunked upload in progress (${uploadPct}% sent)...`
          : job
          ? job.error_message || job.current_stage_detail
          : "Paste a YouTube link, upload an MP4, or launch one of the 1-click CV Benchmarks."}
      </p>

      {/* Progress Bar */}
      <div className="mt-4">
        <div className="flex items-center justify-between text-xs font-mono text-slate-300 mb-1.5">
          <span>
            {job?.quality_mode ? `Tier: ${job.quality_mode.toUpperCase()}` : "Streaming O(1) Memory Engine"}
          </span>
          <span className="font-bold text-sky-300">
            {uploadPct !== null ? `${uploadPct}%` : job ? `${job.progress_pct}%` : "0%"}
          </span>
        </div>
        <div className="w-full h-2.5 bg-slate-950/90 rounded-full overflow-hidden border border-white/10">
          <motion.div
            className="h-full bg-gradient-to-r from-sky-400 via-indigo-500 to-emerald-400 bg-[length:200%_100%] animate-shimmer"
            initial={{ width: 0 }}
            animate={{
              width: `${uploadPct !== null ? uploadPct : job ? job.progress_pct : 0}%`,
            }}
            transition={{ duration: 0.35, ease: "easeOut" }}
          />
        </div>
      </div>

      {/* 5-Stage Pipeline Nodes */}
      <div className="mt-5 space-y-2">
        {STAGES.map((stage, idx) => {
          const stageNum = idx + 1;
          const isDone = currentRank > stageNum || job?.status === "completed";
          const isActive = currentRank === stageNum && isRunning;
          const Icon = stage.icon;

          return (
            <div
              key={stage.key}
              className={`flex items-center gap-3 px-3 py-2 rounded-xl border transition-all ${
                isActive
                  ? "bg-sky-500/12 border-sky-400/40 text-white shadow-sm shadow-sky-500/10"
                  : isDone
                  ? "bg-emerald-500/8 border-emerald-500/20 text-slate-200"
                  : "bg-slate-950/45 border-white/5 text-slate-500"
              }`}
            >
              <div
                className={`w-7 h-7 rounded-lg flex items-center justify-center shrink-0 ${
                  isActive
                    ? "bg-sky-500 text-slate-950"
                    : isDone
                    ? "bg-emerald-500/20 text-emerald-400"
                    : "bg-slate-900 text-slate-500"
                }`}
              >
                {isActive ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : isDone ? (
                  <CheckCircle2 className="w-4 h-4" />
                ) : (
                  <Icon className="w-3.5 h-3.5" />
                )}
              </div>
              <div className="min-w-0 flex-1">
                <div className="text-xs font-semibold truncate">{stage.label}</div>
                <div className="text-[11px] text-slate-400 truncate">{stage.desc}</div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Telemetry Counters */}
      <div className="grid grid-cols-3 gap-2.5 mt-5 pt-4 border-t border-white/10 text-center">
        <div className="p-2.5 rounded-xl bg-slate-950/70 border border-white/5">
          <span className="block text-[10px] uppercase tracking-wider text-slate-400">Sampled</span>
          <span className="text-base font-extrabold font-mono text-white">
            {job?.total_candidate_frames ?? 0}
          </span>
        </div>
        <div className="p-2.5 rounded-xl bg-slate-950/70 border border-white/5">
          <span className="block text-[10px] uppercase tracking-wider text-slate-400">Clean Slides</span>
          <span className="text-base font-extrabold font-mono text-emerald-400">
            {job ? job.extracted_slide_count || job.slides?.length || 0 : 0}
          </span>
        </div>
        <div className="p-2.5 rounded-xl bg-slate-950/70 border border-white/5">
          <span className="block text-[10px] uppercase tracking-wider text-slate-400">De-Occluded</span>
          <span className="text-base font-extrabold font-mono text-amber-300">
            {job?.reconstructed_slide_count ?? 0}
          </span>
        </div>
      </div>

      {/* Action Buttons */}
      {job && (
        <div className="mt-5 flex flex-wrap gap-2.5">
          {job.status === "completed" && job.pdf_download_url && (
            <a
              href={resolveAssetUrl(job.pdf_download_url)}
              target="_blank"
              rel="noopener noreferrer"
              className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-emerald-400 to-teal-500 hover:from-emerald-300 hover:to-teal-400 text-slate-950 font-extrabold text-sm flex items-center justify-center gap-2 shadow-lg shadow-emerald-500/25 transition"
            >
              <Download className="w-4 h-4" />
              <span>
                Download Full Slide Deck PDF
                {job.pdf_file_size_bytes ? ` (${Math.round(job.pdf_file_size_bytes / 1024)} KB)` : ""}
              </span>
            </a>
          )}

          {isRunning && (
            <button
              type="button"
              onClick={() => onCancel(job.id)}
              className="py-2 px-4 rounded-xl bg-rose-500/15 hover:bg-rose-500/25 border border-rose-500/35 text-rose-200 text-xs font-semibold inline-flex items-center gap-1.5 transition"
            >
              <XCircle className="w-3.5 h-3.5" />
              Cancel Job
            </button>
          )}

          {(job.status === "failed" || job.status === "cancelled") && (
            <>
              <button
                type="button"
                onClick={() => onRetry(job.id)}
                className="py-2 px-4 rounded-xl bg-sky-500/15 hover:bg-sky-500/25 border border-sky-500/35 text-sky-200 text-xs font-semibold inline-flex items-center gap-1.5 transition"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Retry Job
              </button>
              <button
                type="button"
                onClick={onTryDemo}
                className="py-2 px-4 rounded-xl bg-amber-500/15 hover:bg-amber-500/25 border border-amber-500/35 text-amber-200 text-xs font-semibold inline-flex items-center gap-1.5 transition"
              >
                <Sparkles className="w-3.5 h-3.5" />
                Run Chalkboard CV Demo
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}
