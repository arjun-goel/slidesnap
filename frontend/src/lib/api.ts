import { HealthResponse, JobResponse, QualityMode, SlideMetadata, SourceType } from "@/types/slidesnap";

export const DEFAULT_BACKEND_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/+$/, "") ||
  "https://slidesnap-lecture-extractor.onrender.com";

const STORAGE_KEY = "slidesnap_backend_url_override";

export function getBackendBaseUrl(): string {
  if (typeof window !== "undefined") {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved && saved.trim()) {
      return saved.trim().replace(/\/+$/, "");
    }
  }
  return DEFAULT_BACKEND_URL;
}

export function setBackendBaseUrl(url: string | null): void {
  if (typeof window === "undefined") return;
  if (!url || !url.trim() || url.trim().replace(/\/+$/, "") === DEFAULT_BACKEND_URL) {
    window.localStorage.removeItem(STORAGE_KEY);
  } else {
    window.localStorage.setItem(STORAGE_KEY, url.trim().replace(/\/+$/, ""));
  }
}

export function resolveAssetUrl(pathOrUrl: string | null | undefined): string {
  if (!pathOrUrl) return "";
  if (pathOrUrl.startsWith("http://") || pathOrUrl.startsWith("https://")) {
    return pathOrUrl;
  }
  const base = getBackendBaseUrl();
  return `${base}${pathOrUrl.startsWith("/") ? "" : "/"}${pathOrUrl}`;
}

export function extractYouTubeId(url: string): string | null {
  const match = url.match(/(?:v=|youtu\.be\/|embed\/|shorts\/|live\/)([\w-]{11})/);
  return match ? match[1] : null;
}

export function buildYouTubeTimestampUrl(sourceUrl: string | null, seconds: number): string | null {
  if (!sourceUrl) return null;
  const vid = extractYouTubeId(sourceUrl);
  if (!vid) return null;
  return `https://www.youtube.com/watch?v=${vid}&t=${Math.max(0, Math.floor(seconds))}s`;
}

export function formatTimestamp(sec: number): string {
  const safe = Math.max(0, Math.floor(sec || 0));
  const hrs = Math.floor(safe / 3600);
  const mins = Math.floor((safe % 3600) / 60);
  const secs = safe % 60;
  if (hrs > 0) {
    return `${hrs}:${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

export async function checkBackendHealth(signal?: AbortSignal): Promise<{
  health: HealthResponse;
  latencyMs: number;
}> {
  const base = getBackendBaseUrl();
  const t0 = performance.now();
  const res = await fetch(`${base}/api/health`, {
    method: "GET",
    cache: "no-store",
    signal,
  });
  if (!res.ok) {
    throw new Error(`Health check returned HTTP ${res.status}`);
  }
  const health = (await res.json()) as HealthResponse;
  return { health, latencyMs: Math.round(performance.now() - t0) };
}

export async function createJsonJob(params: {
  source_type: SourceType;
  source_url?: string | null;
  quality_mode: QualityMode;
  enhance_contrast: boolean;
  rights_attested: boolean;
  enable_temporal_inpainting: boolean;
  enable_ocr_indexing: boolean;
  custom_title?: string | null;
}): Promise<JobResponse> {
  const base = getBackendBaseUrl();
  const res = await fetch(`${base}/api/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data?.detail
      ? typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail)
      : `Request failed (HTTP ${res.status})`;
    throw new Error(detail);
  }
  return data as JobResponse;
}

export function uploadVideoJobWithProgress(params: {
  file: File;
  quality_mode: QualityMode;
  enhance_contrast: boolean;
  rights_attested: boolean;
  enable_temporal_inpainting: boolean;
  enable_ocr_indexing: boolean;
  custom_title?: string;
  onUploadProgress?: (pct: number) => void;
}): Promise<JobResponse> {
  return new Promise((resolve, reject) => {
    const base = getBackendBaseUrl();
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${base}/api/jobs/upload`);

    xhr.upload.onprogress = (ev) => {
      if (ev.lengthComputable && params.onUploadProgress) {
        params.onUploadProgress(Math.round((ev.loaded / ev.total) * 100));
      }
    };

    xhr.onload = () => {
      try {
        const parsed = JSON.parse(xhr.responseText || "{}");
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(parsed as JobResponse);
        } else {
          const detail = parsed?.detail
            ? typeof parsed.detail === "string"
              ? parsed.detail
              : JSON.stringify(parsed.detail)
            : `Upload failed with HTTP ${xhr.status}`;
          reject(new Error(detail));
        }
      } catch {
        reject(new Error(`Upload failed (HTTP ${xhr.status})`));
      }
    };

    xhr.onerror = () => reject(new Error("Network error while uploading video to CV backend."));

    const formData = new FormData();
    formData.append("video_file", params.file);
    formData.append("rights_attested", String(params.rights_attested));
    formData.append("quality_mode", params.quality_mode);
    formData.append("enhance_contrast", String(params.enhance_contrast));
    formData.append("enable_temporal_inpainting", String(params.enable_temporal_inpainting));
    formData.append("enable_ocr_indexing", String(params.enable_ocr_indexing));
    if (params.custom_title?.trim()) {
      formData.append("custom_title", params.custom_title.trim());
    }
    xhr.send(formData);
  });
}

export class JobNotFoundError extends Error {
  constructor(jobId: string) {
    super(`Job ${jobId} was cleared or the cloud worker restarted.`);
    this.name = "JobNotFoundError";
  }
}

export async function fetchJob(jobId: string): Promise<JobResponse> {
  const base = getBackendBaseUrl();
  const res = await fetch(`${base}/api/jobs/${jobId}`, { cache: "no-store" });
  if (res.status === 404) throw new JobNotFoundError(jobId);
  if (!res.ok) throw new Error(`Job lookup failed (${res.status})`);
  return (await res.json()) as JobResponse;
}

export async function fetchJobHistory(): Promise<JobResponse[]> {
  const base = getBackendBaseUrl();
  const res = await fetch(`${base}/api/jobs`, { cache: "no-store" });
  if (!res.ok) throw new Error(`History lookup failed (${res.status})`);
  return (await res.json()) as JobResponse[];
}

export async function cancelJob(jobId: string): Promise<void> {
  const base = getBackendBaseUrl();
  await fetch(`${base}/api/jobs/${jobId}/cancel`, { method: "POST" });
}

export async function retryJob(jobId: string): Promise<JobResponse> {
  const base = getBackendBaseUrl();
  const res = await fetch(`${base}/api/jobs/${jobId}/retry`, { method: "POST" });
  if (!res.ok) throw new Error("Could not retry job.");
  return (await res.json()) as JobResponse;
}

export function generateClientMarkdownNotes(job: JobResponse, slides: SlideMetadata[]): string {
  const lines: string[] = [
    `# ${job.video_title} — SlideSnap Study Notes`,
    "",
    `- **Extracted Slides**: ${slides.length}`,
    `- **De-Occluded Slides**: ${slides.filter((s) => s.was_reconstructed).length}`,
    `- **Quality Mode**: \`${job.quality_mode || "balanced"}\``,
    job.source_url ? `- **Lecture Source**: ${job.source_url}` : `- **Source Mode**: \`${job.source_type}\``,
    "",
    "---",
    "",
  ];

  for (const s of slides) {
    const tsRange = `${formatTimestamp(s.timestamp_start_sec)} – ${formatTimestamp(s.timestamp_end_sec)}`;
    const ytDeep = buildYouTubeTimestampUrl(job.source_url, s.timestamp_start_sec);
    const tsFormatted = ytDeep ? `[${tsRange}](${ytDeep})` : `\`${tsRange}\``;
    const modeLabel = s.was_reconstructed
      ? `Reconstructed via \`${s.reconstruction_method}\``
      : "Unobstructed Pre-Transition Frame";

    lines.push(`## Slide #${s.slide_number} (${tsFormatted})`);
    lines.push(`- **Method**: ${modeLabel}`);
    lines.push(
      `- **Quality**: ${(s.final_quality_score * 100).toFixed(1)}% · **Sharpness**: ${s.sharpness_score.toFixed(0)} · **Residual Occlusion**: ${(s.person_occlusion_ratio * 100).toFixed(2)}%`
    );
    lines.push(`- **Content Summary**: ${s.ocr_text}`);
    lines.push("");
  }

  return lines.join("\n");
}
