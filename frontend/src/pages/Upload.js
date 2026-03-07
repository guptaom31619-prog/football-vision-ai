import React, { useCallback, useState } from "react";
import UploadBox from "../components/UploadBox";
import DetectionCanvas from "../components/DetectionCanvas";
import { detectImage, detectVideo, BASE_URL } from "../services/detectionApi";

/* ──────────────────────────── tiny helper components ──────────────────────── */

function StatCard({ value, label, color = "text-accent-400" }) {
  return (
    <div className="flex flex-col items-center rounded-xl bg-pitch-800/60 px-3 py-3">
      <p className={`text-xl font-extrabold tabular-nums ${color}`}>{value}</p>
      <p className="mt-0.5 text-[10px] uppercase tracking-wider text-slate-500">{label}</p>
    </div>
  );
}

function SectionHeader({ children }) {
  return (
    <p className="mb-3 flex items-center gap-2 text-[11px] font-bold uppercase tracking-widest text-slate-500">
      <span className="h-px flex-1 bg-pitch-700" />
      {children}
      <span className="h-px flex-1 bg-pitch-700" />
    </p>
  );
}

function PossessionBar({ pctA, pctB }) {
  return (
    <div className="overflow-hidden rounded-full">
      <div className="flex h-5 text-[10px] font-bold leading-5 text-white">
        <div
          className="flex items-center justify-center transition-all duration-500"
          style={{ width: `${pctA}%`, background: "#0891b2" }}
        >
          {pctA > 10 && `${pctA}%`}
        </div>
        <div
          className="flex items-center justify-center transition-all duration-500"
          style={{ width: `${pctB}%`, background: "#ea580c" }}
        >
          {pctB > 10 && `${pctB}%`}
        </div>
      </div>
    </div>
  );
}

function Badge({ team }) {
  const bg = team === "A" ? "#0891b2" : "#ea580c";
  return (
    <span
      className="rounded-full px-2 py-0.5 text-[10px] font-bold uppercase text-white"
      style={{ background: bg }}
    >
      Team {team}
    </span>
  );
}

function DownloadBtn({ href, label }) {
  return (
    <a
      href={href}
      download
      className="inline-flex items-center gap-1.5 rounded-lg bg-accent-500/20 px-3 py-1.5 text-xs font-semibold text-accent-400 transition hover:bg-accent-500/30"
    >
      <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5m0 0l5-5m-5 5V3" />
      </svg>
      {label}
    </a>
  );
}

function PossessionTimeline({ data }) {
  if (!data || data.length === 0) return null;
  const barW = 100 / data.length;
  return (
    <div className="mt-4 border-t border-pitch-700 pt-4">
      <p className="mb-2 text-[10px] font-bold uppercase tracking-widest text-slate-500">
        Possession Timeline
      </p>
      <svg viewBox="0 0 100 30" className="w-full rounded" preserveAspectRatio="none">
        {data.map((d, i) => (
          <g key={i}>
            <rect x={i * barW} y={0} width={barW} height={(d.pct_A / 100) * 30} fill="#0891b2" />
            <rect x={i * barW} y={(d.pct_A / 100) * 30} width={barW} height={(d.pct_B / 100) * 30} fill="#ea580c" />
          </g>
        ))}
      </svg>
      <div className="mt-1 flex justify-between text-[9px] text-slate-600">
        <span>Start</span>
        <span>End</span>
      </div>
    </div>
  );
}

function CoverageMeter({ label, pct, color }) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-16 text-xs text-slate-400">{label}</span>
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-pitch-800">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${Math.min(pct, 100)}%`, background: color }}
        />
      </div>
      <span className="w-10 text-right text-xs font-bold tabular-nums" style={{ color }}>{pct}%</span>
    </div>
  );
}

/* ──────────────────────────────── main page ───────────────────────────────── */

export default function Upload() {
  const [previewUrl, setPreviewUrl]     = useState(null);
  const [detections, setDetections]     = useState([]);
  const [videoResults, setVideoResults] = useState(null);
  const [matchStats, setMatchStats]     = useState(null);
  const [loading, setLoading]           = useState(false);
  const [progress, setProgress]         = useState(null);
  const [error, setError]               = useState(null);
  const [fileType, setFileType]         = useState(null);
  const [annotatedImage, setAnnotatedImage] = useState(null);

  const handleRunDetection = useCallback(async (file) => {
    setError(null);
    setDetections([]);
    setVideoResults(null);
    setMatchStats(null);
    setProgress(null);
    setLoading(true);
    setAnnotatedImage(null);

    const isVideo = file.type.startsWith("video/");
    setFileType(isVideo ? "video" : "image");
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(file));

    try {
      if (isVideo) {
        const result = await detectVideo(file, (p) => setProgress(p));
        setVideoResults(result);
        if (result.stats) setMatchStats(result.stats);
      } else {
        const result = await detectImage(file);
        setDetections(result.detections);
        if (result.stats) setMatchStats(result.stats);
        if (result.annotated_image) setAnnotatedImage(result.annotated_image);
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Detection failed.");
    } finally {
      setLoading(false);
      setProgress(null);
    }
  }, [previewUrl]);

  /* ── derived stats ── */
  const s = matchStats || {};
  const totalDetected = videoResults
    ? videoResults.results.reduce((sum, f) => sum + f.detections.length, 0)
    : detections.length;
  const teamACt     = s.players_team_A ?? detections.filter((d) => d.team === "A").length;
  const teamBCt     = s.players_team_B ?? detections.filter((d) => d.team === "B").length;
  const refsOnField = s.referees_on_field ?? 0;
  const possession  = s.ball_possession ?? null;
  const possA       = s.possession_pct?.A ?? 0;
  const possB       = s.possession_pct?.B ?? 0;
  const ballVis     = s.ball_visibility_pct ?? null;
  const classCounts = s.class_counts ?? {};
  const avgConf     = s.avg_confidence ?? {};
  const coverageA   = s.pitch_coverage_pct?.A ?? 0;
  const coverageB   = s.pitch_coverage_pct?.B ?? 0;
  const atkThirdA   = s.attacking_third_pct?.A ?? 0;
  const atkThirdB   = s.attacking_third_pct?.B ?? 0;
  const possTimeline = s.possession_timeline ?? [];

  const hasResults = detections.length > 0 || videoResults !== null;
  const isVideo    = fileType === "video";

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">

      {/* ── Header ── */}
      <div className="mb-6 flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-extrabold tracking-tight text-white">
            Detection <span className="text-accent-400">Studio</span>
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            Upload an image or video, run YOLOv8 detection, and explore results.
          </p>
        </div>

        {/* Download buttons */}
        {!loading && hasResults && (
          <div className="flex items-center gap-2">
            {isVideo && videoResults?.annotated_video && (
              <DownloadBtn
                href={`${BASE_URL}/outputs/${videoResults.annotated_video}`}
                label="Video"
              />
            )}
            {!isVideo && annotatedImage && (
              <DownloadBtn
                href={`${BASE_URL}/outputs/${annotatedImage}`}
                label="Image"
              />
            )}
            {isVideo && videoResults?.heatmaps?.map((name) => (
              <DownloadBtn
                key={name}
                href={`${BASE_URL}/heatmaps/${name}`}
                label={name.replace(".png", "").replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
              />
            ))}
          </div>
        )}
      </div>

      {/* ═══════════ TOP ROW: upload + media ═══════════════════════════════════ */}
      <div className="grid gap-5 lg:grid-cols-[340px_1fr]">

        {/* Upload box — fixed width on desktop */}
        <div className="flex flex-col gap-4">
          <UploadBox onRunDetection={handleRunDetection} loading={loading} />

          {/* Progress */}
          {loading && (
            <div className="rounded-xl border border-pitch-700 bg-pitch-900 p-4">
              {isVideo && progress ? (
                <>
                  <div className="mb-2 flex items-center justify-between text-xs">
                    <span className="text-slate-400">
                      Frame {progress.frame?.toLocaleString()} / {progress.total?.toLocaleString()}
                    </span>
                    <span className="font-bold text-accent-400">{progress.percent}%</span>
                  </div>
                  <div className="h-2 w-full overflow-hidden rounded-full bg-pitch-800">
                    <div
                      className="h-full rounded-full bg-accent-400 transition-all duration-200"
                      style={{ width: `${progress.percent}%` }}
                    />
                  </div>
                  <p className="mt-1.5 text-[10px] text-slate-500">
                    {progress.sampled} frames sampled · YOLOv8 running
                  </p>
                </>
              ) : (
                <div className="flex items-center gap-2 text-xs text-slate-400">
                  <span className="h-3 w-3 rounded-full border-2 border-accent-400 border-t-transparent animate-spin" />
                  {isVideo ? "Uploading video…" : "Running YOLOv8 inference…"}
                </div>
              )}
            </div>
          )}

          {error && (
            <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400">
              {error}
            </div>
          )}
        </div>

        {/* Media output — image canvas or video player */}
        <div className="min-h-[260px] overflow-hidden rounded-xl border border-pitch-700 bg-pitch-900">
          {!loading && isVideo && videoResults?.annotated_video ? (
            <video
              controls
              autoPlay
              muted
              className="h-full w-full object-contain"
              src={`${BASE_URL}/outputs/${videoResults.annotated_video}`}
            />
          ) : !loading && !isVideo && annotatedImage ? (
            <img
              src={`${BASE_URL}/outputs/${annotatedImage}`}
              alt="Annotated detection"
              className="h-full w-full object-contain"
            />
          ) : fileType === "image" && detections.length > 0 ? (
            <DetectionCanvas imageSrc={previewUrl} detections={detections} />
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-slate-600">
              {loading ? "" : "Your results will appear here"}
            </div>
          )}
        </div>
      </div>

      {/* ═══════════ RESULTS SECTION ═══════════════════════════════════════════ */}
      {!loading && hasResults && (
        <div className="mt-6 space-y-5">

          {/* Snapshot thumbnail (video) */}
          {isVideo && videoResults?.snapshot && (
            <div className="flex items-center gap-3">
              <img
                src={`${BASE_URL}/outputs/${videoResults.snapshot}`}
                alt="Best frame snapshot"
                className="h-20 rounded-lg border border-pitch-700 object-cover"
              />
              <p className="text-xs text-slate-500">Key frame snapshot</p>
            </div>
          )}

          {/* ── Row 1: Key numbers ── */}
          <SectionHeader>Match Overview</SectionHeader>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
            <StatCard value={totalDetected.toLocaleString()} label="Total Detections" />
            {videoResults && (
              <StatCard value={videoResults.frames_sampled} label="Frames Sampled" />
            )}
            <StatCard value={teamACt} label="Team A Players" color="text-cyan-400" />
            <StatCard value={teamBCt} label="Team B Players" color="text-orange-400" />
            {refsOnField > 0 && <StatCard value={refsOnField} label="Referees" color="text-yellow-300" />}
            {ballVis !== null && <StatCard value={`${ballVis}%`} label="Ball Visibility" color="text-fuchsia-400" />}
          </div>

          {/* ── Row 2: Three-column details ── */}
          <div className="grid gap-5 lg:grid-cols-3">

            {/* Col 1: Possession + Timeline */}
            <div className="rounded-xl border border-pitch-700 bg-pitch-900 p-5">
              <p className="mb-4 text-xs font-bold uppercase tracking-widest text-slate-500">
                Ball Possession
              </p>
              {possession ? (
                <>
                  <div className="mb-2 flex items-center justify-between text-sm font-semibold text-white">
                    <span className="flex items-center gap-2">
                      <span className="inline-block h-3 w-3 rounded-sm" style={{ background: "#0891b2" }} />
                      Team A
                    </span>
                    <span className="flex items-center gap-2">
                      Team B
                      <span className="inline-block h-3 w-3 rounded-sm" style={{ background: "#ea580c" }} />
                    </span>
                  </div>
                  <PossessionBar pctA={possA} pctB={possB} />
                  <p className="mt-2 text-center text-xs text-slate-500">
                    Last holder: <span className="font-semibold text-white">{possession}</span>
                  </p>
                  <PossessionTimeline data={possTimeline} />
                </>
              ) : (
                <p className="text-sm text-slate-600">Ball not tracked long enough for possession data.</p>
              )}
            </div>

            {/* Col 2: Pitch Coverage + Attacking Third */}
            <div className="rounded-xl border border-pitch-700 bg-pitch-900 p-5">
              <p className="mb-4 text-xs font-bold uppercase tracking-widest text-slate-500">
                Pitch Coverage
              </p>
              <div className="space-y-3">
                <CoverageMeter label="Team A" pct={coverageA} color="#0891b2" />
                <CoverageMeter label="Team B" pct={coverageB} color="#ea580c" />
              </div>

              <div className="mt-5 border-t border-pitch-700 pt-4">
                <p className="mb-3 text-[10px] font-bold uppercase tracking-widest text-slate-500">
                  Attacking Third Presence
                </p>
                <div className="space-y-3">
                  <CoverageMeter label="Team A" pct={atkThirdA} color="#0891b2" />
                  <CoverageMeter label="Team B" pct={atkThirdB} color="#ea580c" />
                </div>
              </div>
            </div>

            {/* Col 3: Detection class breakdown */}
            <div className="rounded-xl border border-pitch-700 bg-pitch-900 p-5">
              <p className="mb-4 text-xs font-bold uppercase tracking-widest text-slate-500">
                Detections by Class
              </p>
              {Object.keys(classCounts).length > 0 ? (
                <div className="space-y-2">
                  {Object.entries(classCounts)
                    .sort((a, b) => b[1] - a[1])
                    .map(([cls, cnt]) => (
                      <div key={cls} className="flex items-center justify-between text-xs">
                        <span className="capitalize text-slate-300">{cls}</span>
                        <div className="flex items-center gap-3">
                          <span className="text-slate-500">{cnt.toLocaleString()}</span>
                          {avgConf[cls] && (
                            <span className="rounded bg-pitch-800 px-1.5 py-0.5 text-[10px] font-mono text-accent-400">
                              avg {(avgConf[cls] * 100).toFixed(1)}%
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                </div>
              ) : (
                <p className="text-sm text-slate-600">No class data.</p>
              )}

              {/* Image mode: detected objects list */}
              {!isVideo && detections.length > 0 && (
                <div className="mt-5 border-t border-pitch-700 pt-4">
                  <p className="mb-2 text-[10px] font-bold uppercase tracking-widest text-slate-500">
                    Detected Objects
                  </p>
                  <div className="max-h-48 space-y-1.5 overflow-y-auto pr-1">
                    {detections.map((d, i) => (
                      <div
                        key={i}
                        className="flex items-center justify-between rounded-lg bg-pitch-800/60 px-3 py-1.5"
                      >
                        <div className="flex items-center gap-2">
                          <span className="flex h-5 w-5 items-center justify-center rounded-full bg-pitch-700 text-[10px] font-bold text-slate-400">
                            {i + 1}
                          </span>
                          <span className="text-xs capitalize text-slate-200">{d.label}</span>
                          {d.team && <Badge team={d.team} />}
                        </div>
                        <span className="rounded-full bg-accent-500/15 px-2 py-0.5 text-[10px] font-bold text-accent-400">
                          {(d.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* ── Row 3: Heatmaps (video only) ── */}
          {isVideo && videoResults?.heatmaps?.length > 0 && (
            <>
              <SectionHeader>Player Heatmaps</SectionHeader>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {videoResults.heatmaps.map((name) => {
                  const label = name
                    .replace(".png", "")
                    .replace(/_/g, " ")
                    .replace(/\b\w/g, (c) => c.toUpperCase());
                  return (
                    <div
                      key={name}
                      className="overflow-hidden rounded-xl border border-pitch-700 bg-pitch-900"
                    >
                      <img
                        src={`${BASE_URL}/heatmaps/${name}`}
                        alt={label}
                        className="w-full"
                      />
                      <div className="flex items-center justify-between px-3 py-2">
                        <p className="text-xs font-semibold text-slate-400">{label}</p>
                        <DownloadBtn href={`${BASE_URL}/heatmaps/${name}`} label="Save" />
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          )}

          {/* ── Row 4: Detail table (image mode) ── */}
          {!isVideo && detections.length > 0 && (
            <>
              <SectionHeader>Detection Details</SectionHeader>
              <div className="overflow-hidden rounded-xl border border-pitch-700">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-pitch-700 bg-pitch-800">
                      {["#", "Label", "Team", "Confidence", "Bounding Box"].map((h) => (
                        <th
                          key={h}
                          className="px-4 py-2.5 text-left text-[10px] font-bold uppercase tracking-widest text-slate-500"
                        >
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {detections.map((d, i) => (
                      <tr key={i} className="border-b border-pitch-800/50 odd:bg-pitch-950 even:bg-pitch-900">
                        <td className="px-4 py-2 text-slate-600">{i + 1}</td>
                        <td className="px-4 py-2 font-medium capitalize text-slate-200">{d.label}</td>
                        <td className="px-4 py-2">
                          {d.team ? <Badge team={d.team} /> : <span className="text-slate-700">—</span>}
                        </td>
                        <td className="px-4 py-2">
                          <span className="rounded bg-accent-500/15 px-1.5 py-0.5 text-xs font-mono font-semibold text-accent-400">
                            {(d.confidence * 100).toFixed(1)}%
                          </span>
                        </td>
                        <td className="px-4 py-2 font-mono text-xs text-slate-600">
                          {d.bbox.map((v) => Math.round(v)).join(", ")}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
