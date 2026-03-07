/**
 * DetectionCanvas.js — Image viewer with bounding box overlay
 *
 * Rendering pipeline:
 *  1. An <img> tag displays the uploaded image, scaled to fit the container.
 *  2. A <canvas> of the exact same CSS size sits on top (position: absolute).
 *  3. When detections arrive, we:
 *       a. Measure the rendered image size (may differ from natural resolution).
 *       b. Calculate scale factors: scaleX = renderedW / naturalW
 *       c. Multiply every bbox coordinate by the scale factor before drawing.
 *       d. Draw rectangle + label for each detection.
 *
 * This coordinate scaling is critical — YOLO returns bboxes in the original
 * image pixel space, but the canvas renders at a different (smaller) size.
 *
 * Props:
 *  imageSrc: string       — Object URL of the uploaded image
 *  detections: Array      — [{ label, confidence, bbox: [x1,y1,x2,y2] }, ...]
 */

import React, { useCallback, useEffect, useRef, useState } from "react";

// ── Visual style constants ──────────────────────────────────────────────────
const LINE_WIDTH = 2;
const CORNER_SZ  = 6;          // size of corner accent squares
const FONT_SIZE  = 12;
const FONT       = `bold ${FONT_SIZE}px 'Inter', monospace`;
const LABEL_PAD  = 5;
const LABEL_TEXT = "#f0f9ff";  // sky-50 — label text (all roles)

// Box + label background colour per team / role
const ROLE_COLORS = {
  A:          { box: "#22d3ee", bg: "#083344" },  // Team A — cyan
  B:          { box: "#fb923c", bg: "#3b1005" },  // Team B — orange
  referee:    { box: "#c084fc", bg: "#2e1065" },  // purple
  ball:       { box: "#f8fafc", bg: "#1e293b" },  // white / slate
  goalkeeper: { box: "#facc15", bg: "#3b2a00" },  // yellow
  _default:   { box: "#38bdf8", bg: "#0c1a2e" },  // sky-400 fallback
};

export default function DetectionCanvas({ imageSrc, detections = [] }) {
  const containerRef = useRef(null);
  const imgRef       = useRef(null);
  const canvasRef    = useRef(null);

  // Track the rendered dimensions of the image so we can scale bbox coords
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });

  // ── When the image finishes loading, record its rendered size ─────────────
  const handleImageLoad = useCallback(() => {
    const img = imgRef.current;
    if (!img) return;
    setImgSize({ w: img.clientWidth, h: img.clientHeight });
  }, []);

  // ── Resize observer: keep canvas in sync if container resizes ─────────────
  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    const ro = new ResizeObserver(() => {
      setImgSize({ w: img.clientWidth, h: img.clientHeight });
    });
    ro.observe(img);
    return () => ro.disconnect();
  }, [imageSrc]);

  // ── Draw bounding boxes whenever detections or image size change ───────────
  useEffect(() => {
    const canvas = canvasRef.current;
    const img    = imgRef.current;
    if (!canvas || !img || !imageSrc) return;

    // Match canvas pixel dimensions to the rendered image size
    canvas.width  = imgSize.w;
    canvas.height = imgSize.h;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!detections.length || !imgSize.w || !imgSize.h) return;

    // ── Scale factors: map YOLO pixel coords → canvas pixel coords ──────────
    // YOLO outputs coordinates in the original image's pixel space.
    // The canvas is sized to match the *rendered* image, which may be smaller.
    const scaleX = imgSize.w / img.naturalWidth;
    const scaleY = imgSize.h / img.naturalHeight;

    detections.forEach(({ label, confidence, bbox, team }) => {
      const [rx1, ry1, rx2, ry2] = bbox;

      // Apply scale so boxes align with the displayed image
      const x1 = rx1 * scaleX;
      const y1 = ry1 * scaleY;
      const x2 = rx2 * scaleX;
      const y2 = ry2 * scaleY;
      const bw = x2 - x1;
      const bh = y2 - y1;

      // Pick colour scheme: team assignment overrides role colour
      const colorKey = team || label;
      const { box: boxColor, bg: labelBg } =
        ROLE_COLORS[colorKey] ?? ROLE_COLORS._default;

      // ── Bounding box rectangle ────────────────────────────────────────────
      ctx.strokeStyle = boxColor;
      ctx.lineWidth   = LINE_WIDTH;
      ctx.strokeRect(x1, y1, bw, bh);

      // ── Corner accents (small filled squares at each corner) ──────────────
      ctx.fillStyle = boxColor;
      [
        [x1,             y1            ],   // top-left
        [x2 - CORNER_SZ, y1            ],   // top-right
        [x1,             y2 - CORNER_SZ],   // bottom-left
        [x2 - CORNER_SZ, y2 - CORNER_SZ],  // bottom-right
      ].forEach(([cx, cy]) => ctx.fillRect(cx, cy, CORNER_SZ, CORNER_SZ));

      // ── Label: "player (Team A) 91.2%" ───────────────────────────────────
      const teamSuffix = team ? ` (Team ${team})` : "";
      const caption    = `${label}${teamSuffix}  ${(confidence * 100).toFixed(1)}%`;
      ctx.font = FONT;

      const textMetrics = ctx.measureText(caption);
      const labelW = textMetrics.width + LABEL_PAD * 2;
      const labelH = FONT_SIZE + LABEL_PAD * 2;

      // Position label above the box; clamp to top edge if near viewport top
      const labelY = y1 >= labelH ? y1 - labelH : y1 + bh;

      // Label background pill
      ctx.fillStyle = labelBg;
      ctx.beginPath();
      ctx.roundRect(x1, labelY, labelW, labelH, 3);
      ctx.fill();

      // Label border (same colour as box)
      ctx.strokeStyle = boxColor;
      ctx.lineWidth   = 1;
      ctx.beginPath();
      ctx.roundRect(x1, labelY, labelW, labelH, 3);
      ctx.stroke();

      // Label text
      ctx.fillStyle = LABEL_TEXT;
      ctx.lineWidth = LINE_WIDTH;
      ctx.fillText(caption, x1 + LABEL_PAD, labelY + FONT_SIZE + LABEL_PAD - 2);
    });
  }, [detections, imgSize, imageSrc]);

  return (
    <div
      ref={containerRef}
      className="w-full rounded-card border border-pitch-700 bg-pitch-900 overflow-hidden"
    >
      {imageSrc ? (
        // ── Image + canvas overlay ─────────────────────────────────────────
        <div className="relative w-full">
          {/* Source image — fills the container width, height is auto */}
          <img
            ref={imgRef}
            src={imageSrc}
            alt="Detection source"
            onLoad={handleImageLoad}
            className="block w-full h-auto"
            draggable={false}
          />

          {/* Canvas sits exactly on top of the image */}
          <canvas
            ref={canvasRef}
            style={{ width: imgSize.w, height: imgSize.h }}
            className="absolute top-0 left-0 pointer-events-none"
          />
        </div>
      ) : (
        // ── Empty state ────────────────────────────────────────────────────
        <div className="flex h-64 flex-col items-center justify-center gap-3 text-slate-600">
          <span className="text-4xl">🖼️</span>
          <p className="text-sm">Upload an image to see detections here</p>
        </div>
      )}
    </div>
  );
}
