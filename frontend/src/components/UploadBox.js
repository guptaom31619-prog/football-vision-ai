/**
 * UploadBox.js — File picker with explicit "Run Detection" trigger
 *
 * Flow:
 *  1. User picks or drags an image/video onto the box.
 *  2. The file is validated and a thumbnail preview is shown.
 *  3. User clicks "Run Detection" — this is when onRunDetection(file) fires.
 *  4. While detection is in-flight, the button shows a loading spinner.
 *  5. "Remove" clears the selection so the user can start over.
 *
 * Props:
 *  onRunDetection(file)  — called when user clicks the Run Detection button
 *  accept: string        — comma-separated MIME types
 *  maxSizeMB: number     — max file size (default 100 MB)
 *  loading: boolean      — true while API call is in-flight
 */

import React, { useCallback, useRef, useState } from "react";

const DEFAULT_ACCEPT = "image/jpeg,image/png,video/mp4,video/avi,video/quicktime";
const DEFAULT_MAX_MB = 2048;

export default function UploadBox({
  onRunDetection,
  accept = DEFAULT_ACCEPT,
  maxSizeMB = DEFAULT_MAX_MB,
  loading = false,
}) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver]   = useState(false);
  const [selected, setSelected]   = useState(null);
  const [previewUrl, setPreview]  = useState(null);
  const [error, setError]         = useState(null);

  // ── Validation ────────────────────────────────────────────────────────────
  const validate = useCallback(
    (file) => {
      if (!file) return "No file provided.";
      const allowed = accept.split(",").map((t) => t.trim());
      if (!allowed.includes(file.type)) return `Unsupported type: ${file.type}`;
      if (file.size > maxSizeMB * 1024 * 1024) return `File exceeds ${maxSizeMB} MB.`;
      return null;
    },
    [accept, maxSizeMB]
  );

  // ── Receive a file (from click or drop) ───────────────────────────────────
  const handleFile = useCallback(
    (file) => {
      const err = validate(file);
      if (err) { setError(err); return; }

      setError(null);
      setSelected(file);

      // Generate an object URL for the thumbnail preview
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreview(file.type.startsWith("image/") ? URL.createObjectURL(file) : null);
    },
    [validate, previewUrl]
  );

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragOver(false);
      if (!loading) handleFile(e.dataTransfer.files[0]);
    },
    [loading, handleFile]
  );

  const handleChange = useCallback(
    (e) => handleFile(e.target.files[0]),
    [handleFile]
  );

  // ── Remove selection ──────────────────────────────────────────────────────
  const reset = useCallback(
    (e) => {
      e.stopPropagation();
      setSelected(null);
      setError(null);
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreview(null);
      if (inputRef.current) inputRef.current.value = "";
    },
    [previewUrl]
  );

  // ── Send file to parent for detection ─────────────────────────────────────
  const runDetection = useCallback(
    (e) => {
      e.stopPropagation();
      if (selected && !loading) onRunDetection(selected);
    },
    [selected, loading, onRunDetection]
  );

  const isImage = selected?.type.startsWith("image/");

  return (
    <div className="flex flex-col gap-3">
      {/* ── Drop zone ─────────────────────────────────────────────────────── */}
      <div
        onClick={() => !loading && inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); if (!loading) setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={[
          "relative flex flex-col items-center justify-center rounded-card border-2 border-dashed",
          "transition-all duration-200 px-6 py-8 text-center",
          loading
            ? "opacity-60 cursor-not-allowed border-pitch-700 bg-pitch-900"
            : dragOver
            ? "cursor-pointer border-accent-400 bg-pitch-800"
            : "cursor-pointer border-pitch-700 bg-pitch-900 hover:border-accent-500 hover:bg-pitch-800",
        ].join(" ")}
      >
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          className="hidden"
          onChange={handleChange}
          disabled={loading}
        />

        {selected ? (
          /* ── File selected: show preview + meta ── */
          <div className="flex flex-col items-center gap-2.5">
            {isImage && previewUrl ? (
              // Image thumbnail
              <img
                src={previewUrl}
                alt="preview"
                className="h-32 w-auto rounded-md object-contain border border-pitch-700"
              />
            ) : (
              // Video icon
              <div className="flex h-14 w-14 items-center justify-center rounded-full bg-pitch-800 text-3xl">
                🎬
              </div>
            )}
            <p className="text-sm font-medium text-slate-200 max-w-xs truncate">{selected.name}</p>
            <p className="text-xs text-slate-500">
              {(selected.size / 1024 / 1024).toFixed(2)} MB · Click to change
            </p>
            <button
              onClick={reset}
              className="rounded-md border border-pitch-700 px-3 py-1 text-xs text-slate-400 hover:border-danger-400 hover:text-danger-400 transition-colors"
            >
              Remove
            </button>
          </div>
        ) : (
          /* ── Empty: drop prompt ── */
          <div className="flex flex-col items-center gap-3 pointer-events-none">
            <div className="flex h-14 w-14 items-center justify-center rounded-full bg-pitch-800 text-3xl">
              📁
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-200">
                Drag & drop or <span className="text-accent-400">browse</span>
              </p>
              <p className="mt-1 text-xs text-slate-500">JPEG · PNG · MP4 · No size limit</p>
            </div>
          </div>
        )}
      </div>

      {/* ── Validation error ──────────────────────────────────────────────── */}
      {error && (
        <p className="rounded-md border border-danger-400/30 bg-danger-400/10 px-3 py-2 text-xs text-danger-400">
          {error}
        </p>
      )}

      {/* ── Run Detection button ──────────────────────────────────────────── */}
      {/* Only visible once a file has been selected */}
      {selected && (
        <button
          onClick={runDetection}
          disabled={loading}
          className={[
            "flex w-full items-center justify-center gap-2.5 rounded-card py-3 text-sm font-semibold",
            "transition-all duration-150 active:scale-[0.98]",
            loading
              ? "cursor-not-allowed bg-accent-600 text-white opacity-70"
              : "bg-accent-500 text-white hover:bg-accent-600 shadow-lg shadow-accent-500/20",
          ].join(" ")}
        >
          {loading ? (
            <>
              {/* Spinner */}
              <span className="h-4 w-4 rounded-full border-2 border-white border-t-transparent animate-spin" />
              Running Detection…
            </>
          ) : (
            <>
              ⚡ Run Detection
            </>
          )}
        </button>
      )}
    </div>
  );
}
