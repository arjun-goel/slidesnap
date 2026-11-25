"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Check,
  CheckSquare,
  ClipboardCopy,
  CloudCog,
  Download,
  ExternalLink,
  FileArchive,
  FileCode2,
  FileText,
  Flame,
  FolderUp,
  Gauge,
  History,
  Maximize2,
  Play,
  RefreshCw,
  Search,
  Server,
  ShieldCheck,
  Sliders,
  Sparkles,
  Square,
  Wand2,
  Youtube,
  Zap,
} from "lucide-react";
import {
  DEFAULT_BACKEND_URL,
  buildYouTubeTimestampUrl,
  cancelJob,
  checkBackendHealth,
  createJsonJob,
  extractYouTubeId,
  fetchJob,
  fetchJobHistory,
  formatTimestamp,
  generateClientMarkdownNotes,
  getBackendBaseUrl,
  resolveAssetUrl,
  retryJob,
  setBackendBaseUrl,
  uploadVideoJobWithProgress,
} from "@/lib/api";
import { HealthResponse, JobResponse, QualityMode, SlideMetadata } from "@/types/slidesnap";
import { BeforeAfterSlider } from "@/components/BeforeAfterSlider";
import { PipelineVisualizer } from "@/components/PipelineVisualizer";
import { SlideLightbox } from "@/components/SlideLightbox";

type InputTab = "youtube" | "upload" | "board_demo" | "slides_demo";

const QUALITY_OPTIONS: {
  id: QualityMode;
  badge: string;
  title: string;
  resolution: string;
  desc: string;
  speedTag: string;
  icon: React.ComponentType<{ className?: string }>;
}[] = [
  {
    id: "fast",
    badge: "FASTEST",
    title: "Turbo Draft",
    resolution: "480p • 1.2 FPS",
    desc: "~3× faster processing & ultra-low cloud memory for long 2h+ lectures.",
    speedTag: "⚡ ~3x Speed",
    icon: Zap,
  },
  {
    id: "balanced",
    badge: "RECOMMENDED",
    title: "Balanced HD",
    resolution: "720p • 2.0 FPS",
    desc: "Crisp equations, full temporal presenter removal & optimal speed.",
    speedTag: "⚖️ Best Balance",
    icon: Gauge,
  },
  {
    id: "high",
    badge: "MAX CLARITY",
    title: "Ultra Crisp",
    resolution: "1080p • 2.5 FPS",
    desc: "Highest stroke precision for dense math derivations & fine code.",
    speedTag: "💎 Pro Detail",
    icon: Flame,
  },
];

const SAMPLE_LECTURES = [
  {
    label: "MIT 6.S191 Deep Learning Intro",
    url: "https://www.youtube.com/watch?v=alfdI7S6wCY",
  },
  {
    label: "3Blue1Brown Neural Networks",
    url: "https://www.youtube.com/watch?v=aircAruvnKk",
  },
];

export default function SlideSnapStudioPage() {
  // Backend Connectivity & Render Cold-Start State
  const [backendUrl, setBackendUrlState] = useState<string>(DEFAULT_BACKEND_URL);
  const [showEndpointModal, setShowEndpointModal] = useState(false);
  const [endpointDraft, setEndpointDraft] = useState(DEFAULT_BACKEND_URL);
  const [backendState, setBackendState] = useState<"waking" | "online" | "offline">("waking");
  const [healthInfo, setHealthInfo] = useState<HealthResponse | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [wakeElapsedSec, setWakeElapsedSec] = useState(0);

  // Submission Form State
  const [activeTab, setActiveTab] = useState<InputTab>("youtube");
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [customTitle, setCustomTitle] = useState("");
  const [qualityMode, setQualityMode] = useState<QualityMode>("balanced");
  const [enhanceContrast, setEnhanceContrast] = useState(false);
  const [enableInpainting, setEnableInpainting] = useState(true);
  const [enableOcr, setEnableOcr] = useState(true);
  const [rightsAttested, setRightsAttested] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [uploadPct, setUploadPct] = useState<number | null>(null);
  const [formError, setFormError] = useState<string | null>(null);

  // Active Job & History State
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeJob, setActiveJob] = useState<JobResponse | null>(null);
  const [jobHistory, setJobHistory] = useState<JobResponse[]>([]);

  // Slide Gallery Curation & Inspector State
  const [selectedSlideNums, setSelectedSlideNums] = useState<Set<number>>(new Set());
  const [compareSlideIds, setCompareSlideIds] = useState<Set<string>>(new Set());
  const [filterReconstructedOnly, setFilterReconstructedOnly] = useState(false);
  const [slideSearch, setSlideSearch] = useState("");
  const [lightboxIdx, setLightboxIdx] = useState<number | null>(null);
  const [copiedMd, setCopiedMd] = useState(false);

  const galleryRef = useRef<HTMLElement>(null);

  // Sync saved backend URL on mount
  useEffect(() => {
    const current = getBackendBaseUrl();
    setBackendUrlState(current);
    setEndpointDraft(current);
  }, []);

  // Smart Health & Cold-Start Waker Loop
  const pingBackend = useCallback(async () => {
    setBackendState((prev) => (prev === "online" ? "online" : "waking"));
    try {
      const { health, latencyMs: ms } = await checkBackendHealth();
      setHealthInfo(health);
      setLatencyMs(ms);
      setBackendState("online");
      const jobs = await fetchJobHistory().catch(() => []);
      setJobHistory(jobs);
      setActiveJobId((prevId) => {
        if (prevId) return prevId;
        const firstCompleted = jobs.find((j) => j.status === "completed" && j.slides?.length > 0);
        if (firstCompleted) {
          setActiveJob(firstCompleted);
          setSelectedSlideNums(new Set(firstCompleted.slides.map((s) => s.slide_number)));
          return firstCompleted.id;
        }
        return null;
      });
    } catch {
      setBackendState("offline");
    }
  }, []);

  useEffect(() => {
    pingBackend();
  }, [pingBackend, backendUrl]);

  useEffect(() => {
    if (backendState !== "waking") {
      setWakeElapsedSec(0);
      return;
    }
    const timer = setInterval(() => setWakeElapsedSec((s) => s + 1), 1000);
    return () => clearInterval(timer);
  }, [backendState]);

  // Poll Active Job Progress
  useEffect(() => {
    if (!activeJobId) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const poll = async () => {
      try {
        const j = await fetchJob(activeJobId);
        if (cancelled) return;
        setActiveJob(j);

        if (j.slides?.length > 0) {
          setSelectedSlideNums((prev) => {
            if (prev.size > 0 && j.slides.some((s) => prev.has(s.slide_number))) return prev;
            return new Set(j.slides.map((s) => s.slide_number));
          });
          // Default Before/After slider ON for reconstructed slides with Before image
          setCompareSlideIds((prev) => {
            if (prev.size > 0) return prev;
            const autoIds = j.slides
              .filter((s) => s.was_reconstructed && s.occluded_image_url)
              .map((s) => s.id);
            return new Set(autoIds);
          });
        }

        if (!["completed", "failed", "cancelled"].includes(j.status)) {
          timer = setTimeout(poll, 1100);
        } else {
          fetchJobHistory()
            .then((list) => !cancelled && setJobHistory(list))
            .catch(() => {});
        }
      } catch {
        if (!cancelled) timer = setTimeout(poll, 2200);
      }
    };

    poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [activeJobId]);

  const ytPreviewId = useMemo(() => extractYouTubeId(youtubeUrl), [youtubeUrl]);
  const supportsV2BoardDemo = !healthInfo || healthInfo.version?.startsWith("2.");

  const handleSelectQuality = (q: QualityMode) => {
    setQualityMode(q);
    if (q === "high") setEnhanceContrast(true);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError(null);

    if (!rightsAttested) {
      setFormError("Please confirm the Copyright & Educational Authorization attestation.");
      return;
    }

    setSubmitting(true);
    setUploadPct(null);

    try {
      let created: JobResponse;
      if (activeTab === "upload") {
        if (!uploadFile) {
          throw new Error("Please drop or select a lecture video (.mp4, .webm, .mkv) first.");
        }
        setUploadPct(1);
        created = await uploadVideoJobWithProgress({
          file: uploadFile,
          quality_mode: qualityMode,
          enhance_contrast: enhanceContrast,
          rights_attested: rightsAttested,
          enable_temporal_inpainting: enableInpainting,
          enable_ocr_indexing: enableOcr,
          custom_title: customTitle,
          onUploadProgress: (pct) => setUploadPct(pct),
        });
      } else {
        if (activeTab === "youtube" && !youtubeUrl.trim()) {
          throw new Error("Please paste a valid YouTube lecture URL or choose a 1-click CV Benchmark tab.");
        }
        const resolvedSourceType =
          activeTab === "youtube"
            ? "youtube"
            : activeTab === "board_demo" && supportsV2BoardDemo
            ? "progressive_board_demo"
            : "demo_lecture";

        created = await createJsonJob({
          source_type: resolvedSourceType,
          source_url: activeTab === "youtube" ? youtubeUrl.trim() : null,
          quality_mode: qualityMode,
          enhance_contrast: enhanceContrast,
          rights_attested: rightsAttested,
          enable_temporal_inpainting: enableInpainting,
          enable_ocr_indexing: enableOcr,
          custom_title: customTitle.trim() || null,
        });
      }

      setSelectedSlideNums(new Set());
      setCompareSlideIds(new Set());
      setActiveJob(created);
      setActiveJobId(created.id);
      setBackendState("online");
      fetchJobHistory().then(setJobHistory).catch(() => {});
    } catch (err) {
      setFormError(err instanceof Error ? err.message : "Could not enqueue lecture job.");
    } finally {
      setSubmitting(false);
      setUploadPct(null);
    }
  };

  const filteredSlides = useMemo<SlideMetadata[]>(() => {
    if (!activeJob?.slides) return [];
    return activeJob.slides.filter((s) => {
      if (filterReconstructedOnly && !s.was_reconstructed) return false;
      if (slideSearch.trim()) {
        const q = slideSearch.toLowerCase();
        const matchText = s.ocr_text.toLowerCase().includes(q);
        const matchNum = `slide ${s.slide_number}`.includes(q) || `#${s.slide_number}`.includes(q);
        return matchText || matchNum;
      }
      return true;
    });
  }, [activeJob, filterReconstructedOnly, slideSearch]);

  const toggleSlideSelection = (num: number) => {
    setSelectedSlideNums((prev) => {
      const next = new Set(prev);
      if (next.has(num)) next.delete(num);
      else next.add(num);
      return next;
    });
  };

  const toggleCompareMode = (slideId: string) => {
    setCompareSlideIds((prev) => {
      const next = new Set(prev);
      if (next.has(slideId)) next.delete(slideId);
      else next.add(slideId);
      return next;
    });
  };

  const curatedQueryParam = useMemo(() => {
    if (!activeJob?.slides?.length) return "";
    if (selectedSlideNums.size === 0 || selectedSlideNums.size === activeJob.slides.length) {
      return "";
    }
    const sorted = Array.from(selectedSlideNums).sort((a, b) => a - b);
    return `?slides=${sorted.join(",")}`;
  }, [activeJob, selectedSlideNums]);

  const handleCopyMarkdown = async () => {
    if (!activeJob?.slides?.length) return;
    const chosen = activeJob.slides.filter(
      (s) => selectedSlideNums.size === 0 || selectedSlideNums.has(s.slide_number)
    );
    const md = generateClientMarkdownNotes(activeJob, chosen);
    await navigator.clipboard.writeText(md);
    setCopiedMd(true);
    setTimeout(() => setCopiedMd(false), 2200);
  };

  const handleSaveEndpoint = (e: React.FormEvent) => {
    e.preventDefault();
    setBackendBaseUrl(endpointDraft);
    const updated = getBackendBaseUrl();
    setBackendUrlState(updated);
    setShowEndpointModal(false);
    setBackendState("waking");
  };

  const totalLectureDuration = useMemo(() => {
    if (!activeJob) return 1;
    const lastSlideEnd = activeJob.slides?.length
      ? Math.max(...activeJob.slides.map((s) => s.timestamp_end_sec))
      : 0;
    return Math.max(activeJob.video_duration_sec || 0, lastSlideEnd, 1);
  }, [activeJob]);

  return (
    <div className="min-h-screen flex flex-col">
      {/* Top Studio Header */}
      <header className="sticky top-0 z-40 border-b border-white/10 bg-slate-950/75 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-sky-400 via-indigo-500 to-emerald-400 p-[1px] shadow-lg shadow-sky-500/25">
              <div className="w-full h-full rounded-[11px] bg-slate-950 flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-sky-400" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-lg tracking-tight text-white">
                  SlideSnap
                </span>
                <span className="hidden sm:inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-sky-500/15 text-sky-300 border border-sky-400/30">
                  Studio v2.0
                </span>
              </div>
              <p className="text-[11px] text-slate-400 hidden md:block">
                Pre-Transition Board Lock • Reverse-Chronological Teacher Removal
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2.5 text-xs">
            {/* Signature Made by Arjun Badge */}
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-gradient-to-r from-pink-500/20 via-rose-500/20 to-indigo-500/20 border border-pink-400/40 text-pink-200 font-semibold shadow-sm shadow-pink-500/10 hover:scale-105 transition-transform">
              <span>🎀</span>
              <span>
                Made by <strong className="text-white">Arjun</strong>
              </span>
            </span>

            {/* Live Cloud Worker Status Pill */}
            <button
              type="button"
              onClick={() => setShowEndpointModal(true)}
              title="Configure CV Worker Endpoint"
              className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full border font-medium transition ${
                backendState === "online"
                  ? "bg-emerald-950/60 border-emerald-500/35 text-emerald-200 hover:border-emerald-400"
                  : backendState === "waking"
                  ? "bg-amber-950/60 border-amber-500/40 text-amber-200"
                  : "bg-rose-950/60 border-rose-500/40 text-rose-200"
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  backendState === "online"
                    ? "bg-emerald-400 animate-pulse"
                    : backendState === "waking"
                    ? "bg-amber-400 animate-ping"
                    : "bg-rose-400"
                }`}
              />
              <span className="hidden sm:inline">
                {backendState === "online"
                  ? `CV Worker Online${latencyMs ? ` · ${latencyMs}ms` : ""}`
                  : backendState === "waking"
                  ? `Waking Render Cloud (${wakeElapsedSec}s)...`
                  : "Worker Sleeping · Retry"}
              </span>
              <CloudCog className="w-3.5 h-3.5 opacity-80" />
            </button>

            <a
              href={`${backendUrl}/docs`}
              target="_blank"
              rel="noopener noreferrer"
              className="hidden lg:inline-flex items-center gap-1 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-white/10 text-slate-300 font-medium transition"
            >
              <span>API Docs</span>
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        </div>
      </header>

      {/* Render Cold-Start Helpful Banner */}
      <AnimatePresence>
        {(backendState === "waking" && wakeElapsedSec >= 3) || backendState === "offline" ? (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="border-b border-amber-500/30 bg-amber-500/10 backdrop-blur-md"
          >
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-2.5 flex flex-wrap items-center justify-between gap-3 text-xs text-amber-200">
              <div className="flex items-center gap-2">
                <Server className="w-4 h-4 text-amber-400 shrink-0" />
                <span>
                  {backendState === "waking"
                    ? `Waking up Render Free-Tier Docker CV container (${wakeElapsedSec}s elapsed — cold boot typically takes ~25–40s on first visit)...`
                    : `Render container is still spinning up or unreachable at ${backendUrl}. Click Wake Ping to reconnect.`}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={pingBackend}
                  className="px-2.5 py-1 rounded-lg bg-amber-400/20 hover:bg-amber-400/30 border border-amber-400/40 font-semibold text-amber-100 inline-flex items-center gap-1 transition"
                >
                  <RefreshCw className="w-3 h-3" />
                  Wake Ping
                </button>
                <button
                  type="button"
                  onClick={() => setShowEndpointModal(true)}
                  className="px-2.5 py-1 rounded-lg bg-slate-900/80 border border-white/15 text-slate-200"
                >
                  Switch Endpoint
                </button>
              </div>
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-10">
        {/* Hero Banner */}
        <section className="text-center max-w-4xl mx-auto space-y-4 pt-2">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="inline-flex flex-wrap items-center justify-center gap-2 px-3.5 py-1.5 rounded-full bg-sky-500/10 border border-sky-400/25 text-xs font-medium text-sky-300"
          >
            <Wand2 className="w-3.5 h-3.5 text-sky-400" />
            <span>Pre-Transition Full-Board Detection</span>
            <span className="text-slate-600">•</span>
            <span>Reverse-Chronological Teacher Erasure</span>
            <span className="text-slate-600">•</span>
            <span>O(1) Low-RAM Cloud Engine</span>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05 }}
            className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white leading-[1.12]"
          >
            Turn Any Lecture Video Into{" "}
            <span className="bg-gradient-to-r from-sky-400 via-indigo-400 to-emerald-400 bg-clip-text text-transparent">
              Unobstructed Study Slides
            </span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-sm sm:text-base text-slate-400 max-w-2xl mx-auto leading-relaxed"
          >
            Captures 100% of progressive chalkboard & slide derivations at the exact moment before
            slide transitions—while seamlessly erasing walking teachers blocking the board.
          </motion.p>
        </section>

        {/* Studio Workspace Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: Input & Quality Control Studio (7 cols) */}
          <motion.section
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.12 }}
            className="lg:col-span-7 glass-panel rounded-2xl p-6 sm:p-7 border border-white/10 shadow-2xl space-y-6"
          >
            {/* 4-Mode Source Selector */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 p-1.5 rounded-xl bg-slate-950/90 border border-white/10">
              {[
                { id: "youtube" as const, label: "YouTube URL", icon: Youtube },
                { id: "upload" as const, label: "Upload MP4", icon: FolderUp },
                { id: "board_demo" as const, label: "Chalkboard Demo", icon: Wand2 },
                { id: "slides_demo" as const, label: "Slide Deck Demo", icon: Sparkles },
              ].map((tab) => {
                const Icon = tab.icon;
                const active = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => {
                      setActiveTab(tab.id);
                      if (tab.id === "board_demo" || tab.id === "slides_demo") {
                        setRightsAttested(true);
                      }
                    }}
                    className={`py-2.5 px-3 rounded-lg text-xs font-bold transition flex items-center justify-center gap-1.5 ${
                      active
                        ? "bg-gradient-to-r from-sky-500 to-indigo-600 text-white shadow-md shadow-sky-500/20"
                        : "text-slate-400 hover:text-white hover:bg-white/5"
                    }`}
                  >
                    <Icon className="w-3.5 h-3.5 shrink-0" />
                    <span className="truncate">{tab.label}</span>
                  </button>
                );
              })}
            </div>

            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Source Panel Content */}
              {activeTab === "youtube" && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-bold uppercase tracking-wider text-slate-300">
                      YouTube Lecture Link
                    </label>
                    <div className="flex items-center gap-1.5">
                      {SAMPLE_LECTURES.map((s) => (
                        <button
                          key={s.url}
                          type="button"
                          onClick={() => setYoutubeUrl(s.url)}
                          className="text-[11px] px-2 py-0.5 rounded-md bg-slate-800/90 hover:bg-slate-700 text-sky-300 border border-sky-500/20 transition"
                        >
                          + {s.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  <input
                    type="url"
                    value={youtubeUrl}
                    onChange={(e) => setYoutubeUrl(e.target.value)}
                    placeholder="https://www.youtube.com/watch?v=..."
                    className="w-full px-4 py-3 rounded-xl bg-slate-950/90 border border-white/15 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500 text-sm font-mono"
                  />

                  {ytPreviewId && (
                    <div className="flex items-center gap-3 p-2.5 rounded-xl bg-slate-950/70 border border-white/10">
                      <img
                        src={`https://i.ytimg.com/vi/${ytPreviewId}/mqdefault.jpg`}
                        alt="YouTube preview"
                        className="w-24 h-14 object-cover rounded-lg border border-white/10 shrink-0"
                      />
                      <div className="text-xs text-slate-300 space-y-0.5">
                        <div className="font-semibold text-white">
                          Detected Video ID: <code className="text-sky-400">{ytPreviewId}</code>
                        </div>
                        <p className="text-[11px] text-slate-400">
                          Streams video-only H.264 DASH matching your selected Quality Tier below.
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {activeTab === "upload" && (
                <div className="space-y-2.5">
                  <label className="block text-xs font-bold uppercase tracking-wider text-slate-300">
                    Direct Lecture Video Upload (.mp4, .mkv, .webm — Up to 500 MB)
                  </label>
                  <label className="flex flex-col items-center justify-center gap-2 p-6 rounded-xl bg-slate-950/70 border-2 border-dashed border-sky-500/35 hover:border-sky-400 cursor-pointer transition text-center">
                    <FolderUp className="w-8 h-8 text-sky-400" />
                    <span className="text-sm font-semibold text-white">
                      {uploadFile ? uploadFile.name : "Click to choose a recorded lecture file"}
                    </span>
                    <span className="text-xs text-slate-400">
                      {uploadFile
                        ? `${(uploadFile.size / (1024 * 1024)).toFixed(2)} MB • Streams directly to CV backend`
                        : "100% immune to YouTube datacenter bot checks • Direct streaming"}
                    </span>
                    <input
                      type="file"
                      accept="video/mp4,video/webm,video/x-matroska,video/quicktime"
                      className="hidden"
                      onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                    />
                  </label>
                </div>
              )}

              {activeTab === "board_demo" && (
                <div className="rounded-xl bg-gradient-to-br from-emerald-950/50 to-slate-950/80 border border-emerald-500/35 p-4 space-y-2">
                  <div className="flex items-center gap-2 text-emerald-300 font-bold text-sm">
                    <Wand2 className="w-4 h-4" />
                    <span>MATH-302 Progressive Chalkboard + Walking Teacher Benchmark</span>
                  </div>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    Synthesizes a university green-chalkboard derivation on the fly: the professor writes{" "}
                    <strong>Step 1 → Step 2 → Step 3 [Q.E.D.]</strong> progressively up to the 6.0s board
                    wipe while standing directly in front of Step 1 & Step 2 at the moment of transition!
                  </p>
                  <p className="text-[11px] text-emerald-300/90 font-medium">
                    ✓ Proves SlideSnap keeps all 3 steps on 1 complete slide AND erases the professor
                    with an interactive Before/After slider!
                  </p>
                </div>
              )}

              {activeTab === "slides_demo" && (
                <div className="rounded-xl bg-gradient-to-br from-amber-950/40 to-slate-950/80 border border-amber-500/35 p-4 space-y-2">
                  <div className="flex items-center gap-2 text-amber-300 font-bold text-sm">
                    <Sparkles className="w-4 h-4" />
                    <span>CS-401 Deep Learning Multi-Hazard Slide Benchmark</span>
                  </div>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    Generates an 18.5s HD lecture with <strong>4 slides + 1 revisited duplicate</strong>:
                    tests clean-frame selection (Slide 1), 100%-occluded walking lecturer removal (Slide 2),
                    camera autofocus blur rejection (Slide 3), red player seekbar rejection (Slide 4), and
                    global revisit deduplication.
                  </p>
                </div>
              )}

              {/* 3-Card Quality & Speed Selector */}
              <div className="space-y-2.5">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                    <Sliders className="w-3.5 h-3.5 text-sky-400" />
                    Capture Quality & Processing Speed Tier
                  </label>
                  <span className="text-[11px] text-slate-400">
                    Optimized for Render 512 MB Cloud
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  {QUALITY_OPTIONS.map((opt) => {
                    const selected = qualityMode === opt.id;
                    const Icon = opt.icon;
                    return (
                      <button
                        key={opt.id}
                        type="button"
                        onClick={() => handleSelectQuality(opt.id)}
                        className={`text-left p-3.5 rounded-xl border transition-all relative ${
                          selected
                            ? "bg-sky-500/15 border-sky-400 shadow-lg shadow-sky-500/10"
                            : "bg-slate-950/65 border-white/10 hover:border-white/25"
                        }`}
                      >
                        <div className="flex items-center justify-between gap-1 mb-1.5">
                          <span
                            className={`text-[10px] font-extrabold tracking-wider px-2 py-0.5 rounded-md ${
                              selected
                                ? "bg-sky-400 text-slate-950"
                                : "bg-slate-800 text-slate-400"
                            }`}
                          >
                            {opt.badge}
                          </span>
                          <Icon
                            className={`w-4 h-4 ${selected ? "text-sky-300" : "text-slate-500"}`}
                          />
                        </div>
                        <div className="font-bold text-sm text-white">{opt.title}</div>
                        <div className="text-[11px] font-mono text-sky-300/90 mt-0.5">
                          {opt.resolution}
                        </div>
                        <p className="text-[11px] text-slate-400 mt-1.5 leading-snug">{opt.desc}</p>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Optional Deck Title & CV Enhancement Toggles */}
              <div className="space-y-3">
                <input
                  type="text"
                  value={customTitle}
                  onChange={(e) => setCustomTitle(e.target.value)}
                  placeholder="Optional Custom PDF Deck Title (e.g., CS229 Lecture 4 — Backpropagation)"
                  className="w-full px-4 py-2.5 rounded-xl bg-slate-950/80 border border-white/10 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500 text-sm"
                />

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                  <label className="flex items-start gap-2.5 p-3 rounded-xl bg-slate-950/60 border border-white/10 cursor-pointer hover:border-white/20 transition">
                    <input
                      type="checkbox"
                      checked={enableInpainting}
                      onChange={(e) => setEnableInpainting(e.target.checked)}
                      className="mt-0.5 rounded bg-slate-900 border-slate-600 text-sky-500"
                    />
                    <div>
                      <span className="block text-xs font-semibold text-white">Teacher Removal</span>
                      <span className="block text-[10px] text-slate-400">
                        Reverse-time pixel backfill
                      </span>
                    </div>
                  </label>

                  <label className="flex items-start gap-2.5 p-3 rounded-xl bg-slate-950/60 border border-white/10 cursor-pointer hover:border-white/20 transition">
                    <input
                      type="checkbox"
                      checked={enhanceContrast}
                      onChange={(e) => setEnhanceContrast(e.target.checked)}
                      className="mt-0.5 rounded bg-slate-900 border-slate-600 text-sky-500"
                    />
                    <div>
                      <span className="block text-xs font-semibold text-white">Contrast Boost</span>
                      <span className="block text-[10px] text-slate-400">
                        CLAHE crisp chalk/ink
                      </span>
                    </div>
                  </label>

                  <label className="flex items-start gap-2.5 p-3 rounded-xl bg-slate-950/60 border border-white/10 cursor-pointer hover:border-white/20 transition">
                    <input
                      type="checkbox"
                      checked={enableOcr}
                      onChange={(e) => setEnableOcr(e.target.checked)}
                      className="mt-0.5 rounded bg-slate-900 border-slate-600 text-sky-500"
                    />
                    <div>
                      <span className="block text-xs font-semibold text-white">OCR Indexing</span>
                      <span className="block text-[10px] text-slate-400">
                        Searchable PDF footer
                      </span>
                    </div>
                  </label>
                </div>
              </div>

              {/* Copyright Attestation */}
              <label className="flex items-start gap-3 p-3.5 rounded-xl bg-indigo-950/35 border border-indigo-500/30 cursor-pointer">
                <input
                  type="checkbox"
                  checked={rightsAttested}
                  onChange={(e) => setRightsAttested(e.target.checked)}
                  className="mt-0.5 rounded bg-slate-900 border-indigo-400 text-indigo-500"
                />
                <span className="text-xs text-slate-300 leading-relaxed">
                  <strong className="text-indigo-300 inline-flex items-center gap-1">
                    <ShieldCheck className="w-3.5 h-3.5" /> Educational Rights Attestation:
                  </strong>{" "}
                  I confirm this video is owned by me, Creative Commons licensed, or authorized by my
                  instructor for personal study slide extraction.
                </span>
              </label>

              {formError && (
                <div className="p-3.5 rounded-xl bg-rose-500/15 border border-rose-500/40 text-xs text-rose-200">
                  {formError}
                </div>
              )}

              <button
                type="submit"
                disabled={submitting}
                className="w-full py-3.5 px-6 rounded-xl font-extrabold text-sm text-white bg-gradient-to-r from-sky-500 via-indigo-600 to-emerald-500 hover:brightness-110 disabled:opacity-60 shadow-xl shadow-sky-500/20 transition flex items-center justify-center gap-2"
              >
                <Play className="w-4 h-4 fill-current" />
                <span>
                  {submitting
                    ? "Dispatching to CV Worker..."
                    : `Extract Clean Slides (${qualityMode.toUpperCase()} Mode) & Compile PDF`}
                </span>
              </button>
            </form>
          </motion.section>

          {/* Right Column: Live CV Pipeline Visualizer (5 cols) */}
          <motion.aside
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.18 }}
            className="lg:col-span-5 space-y-6"
          >
            <PipelineVisualizer
              job={activeJob}
              uploadPct={uploadPct}
              onCancel={async (id) => {
                await cancelJob(id);
                fetchJob(id).then(setActiveJob).catch(() => {});
              }}
              onRetry={async (id) => {
                const j = await retryJob(id);
                setActiveJob(j);
                setActiveJobId(j.id);
              }}
              onTryDemo={() => setActiveTab("board_demo")}
            />

            {/* Algorithmic Advantage Card for Judges */}
            <div className="glass-panel rounded-2xl p-5 border border-white/10 space-y-3">
              <h3 className="text-xs font-extrabold uppercase tracking-widest text-slate-200 flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-amber-300" />
                Why SlideSnap Beats Standard Frame Samplers
              </h3>
              <div className="space-y-2.5 text-xs text-slate-300 leading-relaxed">
                <p>
                  <strong className="text-sky-300">1. Pre-Transition Peak Lock:</strong> Tracks
                  morphological chalk/ink stroke persistence so progressive derivations stay in{" "}
                  <em>one</em> segment right up to the exact frame before a board wipe.
                </p>
                <p>
                  <strong className="text-emerald-300">2. Reverse-Chronological Backfill:</strong>{" "}
                  Anchors on the completed board and walks <em>backward</em> in time only inside the
                  teacher&apos;s silhouette—erasing the teacher without erasing late equations!
                </p>
                <p>
                  <strong className="text-indigo-300">3. O(1) Compressed Ring Buffer:</strong>{" "}
                  Processes multi-hour lectures inside &lt;110 MB RAM on free 512 MB cloud tiers.
                </p>
              </div>
            </div>
          </motion.aside>
        </div>

        {/* Extracted Slide Deck Curation Studio & Interactive Timeline */}
        {activeJob && activeJob.slides && activeJob.slides.length > 0 && (
          <section
            ref={galleryRef}
            className="glass-panel rounded-2xl p-6 sm:p-8 border border-white/10 shadow-2xl space-y-6"
          >
            {/* Studio Header & Multi-Format Export Toolbar */}
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <div className="flex items-center gap-2">
                  <h2 className="text-xl sm:text-2xl font-extrabold text-white">
                    Extracted Clean Slide Deck
                  </h2>
                  <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-bold bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                    {activeJob.slides.length} Unique Slides
                  </span>
                </div>
                <p className="text-xs text-slate-400 mt-1">
                  Drag the split handle on reconstructed slides to inspect Before vs. After teacher
                  de-occlusion, or uncheck slides to build a custom curated PDF.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                {activeJob.pdf_download_url && (
                  <a
                    href={`${resolveAssetUrl(activeJob.pdf_download_url)}${curatedQueryParam}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="py-2.5 px-4 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-extrabold text-xs sm:text-sm inline-flex items-center gap-2 shadow-lg shadow-emerald-500/20 transition"
                  >
                    <FileText className="w-4 h-4" />
                    <span>
                      {curatedQueryParam
                        ? `Download Curated PDF (${selectedSlideNums.size})`
                        : `Download PDF (${activeJob.slides.length})`}
                    </span>
                  </a>
                )}

                <button
                  type="button"
                  onClick={handleCopyMarkdown}
                  className="py-2.5 px-3.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-white/15 text-slate-200 font-semibold text-xs inline-flex items-center gap-1.5 transition"
                >
                  {copiedMd ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-emerald-400" />
                      <span className="text-emerald-300">Copied Markdown!</span>
                    </>
                  ) : (
                    <>
                      <ClipboardCopy className="w-3.5 h-3.5 text-sky-400" />
                      <span>Copy Notes (.MD)</span>
                    </>
                  )}
                </button>

                {activeJob.markdown_download_url && (
                  <a
                    href={`${resolveAssetUrl(activeJob.markdown_download_url)}${curatedQueryParam}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="py-2.5 px-3.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-white/15 text-slate-200 font-semibold text-xs inline-flex items-center gap-1.5 transition"
                  >
                    <FileCode2 className="w-3.5 h-3.5 text-indigo-400" />
                    <span>.MD File</span>
                  </a>
                )}

                {activeJob.zip_download_url && (
                  <a
                    href={`${resolveAssetUrl(activeJob.zip_download_url)}${curatedQueryParam}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="py-2.5 px-3.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-white/15 text-slate-200 font-semibold text-xs inline-flex items-center gap-1.5 transition"
                  >
                    <FileArchive className="w-3.5 h-3.5 text-amber-300" />
                    <span>HD Images (.ZIP)</span>
                  </a>
                )}
              </div>
            </div>

            {activeJob.used_youtube_fallback && (
              <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-xs text-amber-200">
                <strong>Cloud IP Notice:</strong> YouTube blocked direct video stream extraction on
                this datacenter IP, so SlideSnap synthesized structured study cards from public
                oEmbed metadata & high-res snapshots. Upload an `.mp4` directly for full frame-by-frame
                de-occlusion on restricted videos.
              </div>
            )}

            {/* Interactive Lecture Segment Timeline Bar */}
            <div className="p-4 rounded-xl bg-slate-950/75 border border-white/10 space-y-2">
              <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono">
                <span>Interactive Lecture Timeline (00:00)</span>
                <span className="flex items-center gap-3">
                  <span className="inline-flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-emerald-400" /> Direct Clean
                  </span>
                  <span className="inline-flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-amber-400" /> Teacher De-Occluded
                  </span>
                </span>
                <span>{formatTimestamp(totalLectureDuration)}</span>
              </div>
              <div className="relative h-5 w-full rounded-lg bg-slate-900 overflow-hidden border border-white/10 flex">
                {activeJob.slides.map((s, idx) => {
                  const leftPct = (s.timestamp_start_sec / totalLectureDuration) * 100;
                  const widthPct = Math.max(
                    2.5,
                    ((s.timestamp_end_sec - s.timestamp_start_sec) / totalLectureDuration) * 100
                  );
                  return (
                    <button
                      key={s.id}
                      type="button"
                      onClick={() => setLightboxIdx(idx)}
                      style={{ left: `${leftPct}%`, width: `${Math.min(100 - leftPct, widthPct)}%` }}
                      title={`Slide #${s.slide_number} (${formatTimestamp(s.timestamp_start_sec)} – ${formatTimestamp(s.timestamp_end_sec)})`}
                      className={`absolute top-0 bottom-0 border-r border-slate-950/80 transition hover:brightness-125 ${
                        s.was_reconstructed
                          ? "bg-gradient-to-r from-amber-500/80 to-orange-400/80"
                          : "bg-gradient-to-r from-emerald-500/75 to-sky-500/75"
                      }`}
                    />
                  );
                })}
              </div>
            </div>

            {/* Search & Filter Controls */}
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="relative flex-1 min-w-[220px] max-w-md">
                <Search className="w-4 h-4 text-slate-500 absolute left-3.5 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  value={slideSearch}
                  onChange={(e) => setSlideSearch(e.target.value)}
                  placeholder="Filter slides by OCR keywords or slide #..."
                  className="w-full pl-9 pr-4 py-2 rounded-xl bg-slate-950/80 border border-white/10 text-xs text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500"
                />
              </div>

              <div className="flex items-center gap-2 text-xs">
                <button
                  type="button"
                  onClick={() => setFilterReconstructedOnly((v) => !v)}
                  className={`px-3 py-2 rounded-xl border font-semibold transition ${
                    filterReconstructedOnly
                      ? "bg-amber-500/20 border-amber-400/50 text-amber-200"
                      : "bg-slate-950/70 border-white/10 text-slate-300 hover:text-white"
                  }`}
                >
                  ✨ De-Occluded Only ({activeJob.slides.filter((s) => s.was_reconstructed).length})
                </button>
                <button
                  type="button"
                  onClick={() => {
                    if (selectedSlideNums.size === activeJob.slides.length) {
                      setSelectedSlideNums(new Set());
                    } else {
                      setSelectedSlideNums(new Set(activeJob.slides.map((s) => s.slide_number)));
                    }
                  }}
                  className="px-3 py-2 rounded-xl bg-slate-950/70 border border-white/10 text-slate-300 hover:text-white font-semibold transition"
                >
                  {selectedSlideNums.size === activeJob.slides.length ? "Deselect All" : "Select All"}
                </button>
              </div>
            </div>

            {/* Slides Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {filteredSlides.map((s) => {
                const isSelected = selectedSlideNums.has(s.slide_number);
                const hasBefore = Boolean(s.occluded_image_url);
                const showSlider = hasBefore && compareSlideIds.has(s.id);
                const fullIdx = activeJob.slides.findIndex((item) => item.id === s.id);
                const ytUrl = buildYouTubeTimestampUrl(activeJob.source_url, s.timestamp_start_sec);

                return (
                  <motion.div
                    key={s.id}
                    layout
                    initial={{ opacity: 0, scale: 0.98 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className={`rounded-2xl bg-slate-950/85 border overflow-hidden flex flex-col shadow-xl transition ${
                      isSelected ? "border-white/15" : "border-white/5 opacity-60"
                    }`}
                  >
                    {/* Card Header */}
                    <div className="px-4 py-3 bg-slate-900/70 border-b border-white/10 flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2.5">
                        <button
                          type="button"
                          onClick={() => toggleSlideSelection(s.slide_number)}
                          className="text-sky-400 hover:text-sky-300 transition"
                          title="Include/exclude slide in curated export"
                        >
                          {isSelected ? (
                            <CheckSquare className="w-4 h-4" />
                          ) : (
                            <Square className="w-4 h-4 text-slate-500" />
                          )}
                        </button>
                        <span className="px-2.5 py-0.5 rounded-md bg-sky-500 text-slate-950 font-extrabold text-xs">
                          Slide #{s.slide_number}
                        </span>
                        <span className="text-xs font-mono text-slate-400">
                          {formatTimestamp(s.timestamp_start_sec)} –{" "}
                          {formatTimestamp(s.timestamp_end_sec)}
                        </span>
                        {ytUrl && (
                          <a
                            href={ytUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-sky-400 hover:text-sky-300"
                            title="Open exact lecture timestamp on YouTube"
                          >
                            <ExternalLink className="w-3.5 h-3.5" />
                          </a>
                        )}
                      </div>

                      <div className="flex items-center gap-1.5">
                        {hasBefore && (
                          <button
                            type="button"
                            onClick={() => toggleCompareMode(s.id)}
                            className={`px-2.5 py-1 rounded-md text-[11px] font-semibold border transition ${
                              showSlider
                                ? "bg-amber-500/20 border-amber-400/50 text-amber-200"
                                : "bg-slate-800 border-white/10 text-slate-300 hover:text-white"
                            }`}
                          >
                            {showSlider ? "↔ Slider Active" : "Compare Before/After"}
                          </button>
                        )}
                        <button
                          type="button"
                          onClick={() => setLightboxIdx(fullIdx)}
                          className="p-1.5 rounded-md bg-slate-800/90 hover:bg-slate-700 text-slate-300 transition"
                          title="Full-screen HD Inspector"
                        >
                          <Maximize2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>

                    {/* Visual Canvas: Before/After Interactive Slider OR Clean Slide */}
                    {showSlider && s.occluded_image_url ? (
                      <BeforeAfterSlider
                        beforeUrl={resolveAssetUrl(s.occluded_image_url)}
                        afterUrl={resolveAssetUrl(s.image_url)}
                        slideNumber={s.slide_number}
                      />
                    ) : (
                      <div
                        onClick={() => setLightboxIdx(fullIdx)}
                        className="relative aspect-video bg-black cursor-zoom-in group overflow-hidden"
                      >
                        <img
                          src={resolveAssetUrl(s.image_url)}
                          alt={`Slide ${s.slide_number}`}
                          className="w-full h-full object-contain group-hover:scale-[1.01] transition-transform duration-300"
                          loading="lazy"
                        />
                        <div className="absolute bottom-2.5 left-2.5">
                          {s.was_reconstructed ? (
                            <span className="px-2.5 py-1 rounded-md bg-amber-950/90 border border-amber-500/40 text-amber-200 text-[11px] font-semibold backdrop-blur-md">
                              ✨ De-Occluded ({s.reconstruction_method.replace(/_/g, " ")})
                            </span>
                          ) : (
                            <span className="px-2.5 py-1 rounded-md bg-emerald-950/90 border border-emerald-500/40 text-emerald-200 text-[11px] font-semibold backdrop-blur-md">
                              ✓ Clean Pre-Transition Frame
                            </span>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Telemetry & OCR Card Footer */}
                    <div className="p-4 space-y-2.5 text-xs">
                      <div className="flex flex-wrap items-center justify-between gap-2 text-slate-400 font-mono text-[11px]">
                        <span>
                          Board Fullness:{" "}
                          <strong className="text-sky-300">
                            {((s.content_completeness_score ?? 1) * 100).toFixed(0)}%
                          </strong>
                        </span>
                        <span>
                          Quality:{" "}
                          <strong className="text-white">
                            {(s.final_quality_score * 100).toFixed(1)}%
                          </strong>
                        </span>
                        <span>
                          Sharpness:{" "}
                          <strong className="text-white">{s.sharpness_score.toFixed(0)}</strong>
                        </span>
                        <span>
                          Occlusion:{" "}
                          <strong className="text-emerald-400">
                            {(s.person_occlusion_ratio * 100).toFixed(2)}%
                          </strong>
                        </span>
                      </div>

                      <p
                        className="text-slate-300 bg-slate-900/70 px-3 py-2 rounded-lg border border-white/5 truncate"
                        title={s.ocr_text}
                      >
                        🔍 {s.ocr_text}
                      </p>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </section>
        )}

        {/* Recent Lecture Jobs History */}
        <section className="glass-panel rounded-2xl p-6 sm:p-8 border border-white/10">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2.5">
              <History className="w-5 h-5 text-sky-400" />
              <div>
                <h2 className="text-lg font-bold text-white">
                  Recent Cloud Extraction Jobs (24h Auto-Retention)
                </h2>
                <p className="text-xs text-slate-400">
                  Click any job to inspect its extracted slides, Before/After teacher removal, and PDF.
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={() => fetchJobHistory().then(setJobHistory).catch(() => {})}
              className="px-3 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-white/10 text-xs font-semibold text-slate-200 inline-flex items-center gap-1.5 transition"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Refresh
            </button>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs sm:text-sm">
              <thead>
                <tr className="border-b border-white/10 text-slate-400 text-[11px] uppercase tracking-wider">
                  <th className="py-3 px-3">Lecture Title</th>
                  <th className="py-3 px-3">Source / Tier</th>
                  <th className="py-3 px-3">Status</th>
                  <th className="py-3 px-3">Extracted Slides</th>
                  <th className="py-3 px-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 text-slate-200">
                {jobHistory.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="py-8 text-center text-slate-500">
                      No recent jobs on this worker yet. Launch the Chalkboard Demo or paste a YouTube
                      link above!
                    </td>
                  </tr>
                ) : (
                  jobHistory.map((j) => (
                    <tr
                      key={j.id}
                      onClick={() => {
                        setSelectedSlideNums(new Set());
                        setCompareSlideIds(new Set());
                        setActiveJob(j);
                        setActiveJobId(j.id);
                        galleryRef.current?.scrollIntoView({ behavior: "smooth" });
                      }}
                      className={`hover:bg-white/5 cursor-pointer transition ${
                        activeJobId === j.id ? "bg-sky-500/8" : ""
                      }`}
                    >
                      <td className="py-3.5 px-3 font-semibold text-white max-w-xs truncate">
                        {j.video_title}
                      </td>
                      <td className="py-3.5 px-3 font-mono text-xs text-slate-400">
                        {j.source_type} · {(j.quality_mode || "balanced").toUpperCase()}
                      </td>
                      <td className="py-3.5 px-3">
                        <span
                          className={`px-2.5 py-0.5 rounded-full text-xs font-mono ${
                            j.status === "completed"
                              ? "bg-emerald-500/15 text-emerald-300"
                              : j.status === "failed"
                              ? "bg-rose-500/15 text-rose-300"
                              : "bg-sky-500/15 text-sky-300"
                          }`}
                        >
                          {j.status === "completed"
                            ? "completed"
                            : `${j.status} (${j.progress_pct}%)`}
                        </span>
                      </td>
                      <td className="py-3.5 px-3 font-mono text-xs text-slate-300">
                        {j.extracted_slide_count} slides ({j.reconstructed_slide_count} de-occluded)
                      </td>
                      <td className="py-3.5 px-3 text-right space-x-2">
                        {j.pdf_download_url && (
                          <a
                            href={resolveAssetUrl(j.pdf_download_url)}
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()}
                            className="px-2.5 py-1 rounded-lg bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 font-semibold text-xs inline-flex items-center gap-1"
                          >
                            <Download className="w-3 h-3" />
                            PDF
                          </a>
                        )}
                        <span className="px-2.5 py-1 rounded-lg bg-slate-800 text-slate-300 text-xs">
                          Inspect
                        </span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </section>
      </main>

      {/* Signature Footer */}
      <footer className="max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 pb-10 pt-4 flex flex-col sm:flex-row items-center justify-between gap-4 border-t border-white/5 text-xs text-slate-500">
        <div>
          SlideSnap Studio v2.0 • OpenCV Pre-Transition & Reverse-Chronological De-Occlusion Engine
        </div>
        <div className="inline-flex items-center gap-2 px-5 py-2 rounded-full bg-slate-900/90 border border-pink-500/30 shadow-lg shadow-pink-500/10 text-xs sm:text-sm text-slate-300 hover:border-pink-400/60 hover:scale-105 transition-all">
          <span>🎀</span>
          <span>
            Made by <strong className="text-pink-300 font-bold">Arjun</strong>
          </span>
          <span>🎀</span>
        </div>
      </footer>

      {/* Lightbox Modal */}
      <SlideLightbox
        slides={activeJob?.slides || []}
        activeIndex={lightboxIdx}
        sourceUrl={activeJob?.source_url || null}
        onClose={() => setLightboxIdx(null)}
        onSelectIndex={setLightboxIdx}
      />

      {/* Backend Endpoint Configuration Modal */}
      {showEndpointModal && (
        <div
          className="fixed inset-0 z-50 bg-slate-950/85 backdrop-blur-md flex items-center justify-center p-4"
          onClick={() => setShowEndpointModal(false)}
        >
          <form
            onSubmit={handleSaveEndpoint}
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-lg rounded-2xl bg-slate-900 border border-white/15 p-6 shadow-2xl space-y-4"
          >
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <CloudCog className="w-5 h-5 text-sky-400" />
              CV Worker Backend Endpoint
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Connected by default to your Render cloud worker (
              <code className="text-sky-300">{DEFAULT_BACKEND_URL}</code>). You can also point this UI
              to a local worker (<code className="text-slate-300">http://127.0.0.1:7860</code>) or a
              GitHub Student Pack DigitalOcean/Azure container.
            </p>
            <input
              type="url"
              required
              value={endpointDraft}
              onChange={(e) => setEndpointDraft(e.target.value)}
              className="w-full px-4 py-2.5 rounded-xl bg-slate-950 border border-white/15 text-sm font-mono text-white"
            />
            <div className="flex items-center justify-between pt-2">
              <button
                type="button"
                onClick={() => {
                  setEndpointDraft(DEFAULT_BACKEND_URL);
                  setBackendBaseUrl(null);
                  setBackendUrlState(DEFAULT_BACKEND_URL);
                  setShowEndpointModal(false);
                }}
                className="text-xs text-sky-400 hover:underline"
              >
                Reset to Default Render URL
              </button>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setShowEndpointModal(false)}
                  className="px-4 py-2 rounded-xl bg-slate-800 text-xs font-semibold text-slate-300"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 rounded-xl bg-sky-500 hover:bg-sky-400 text-xs font-bold text-slate-950"
                >
                  Save & Reconnect
                </button>
              </div>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}
