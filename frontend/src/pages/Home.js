import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { healthCheck } from "../services/detectionApi";

const PIPELINE = [
  {
    step: "01",
    title: "Upload",
    desc: "Drop a football image or full-match video clip.",
    icon: (
      <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
      </svg>
    ),
  },
  {
    step: "02",
    title: "Detect",
    desc: "YOLOv8 finds players, goalkeepers, referees, and the ball.",
    icon: (
      <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
    ),
  },
  {
    step: "03",
    title: "Classify",
    desc: "K-Means jersey-color clustering assigns Team A vs Team B.",
    icon: (
      <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" />
      </svg>
    ),
  },
  {
    step: "04",
    title: "Analyse",
    desc: "Tracking, stats, ball possession, and player heatmaps.",
    icon: (
      <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
  },
];

const TECH = [
  { label: "Model",     value: "YOLOv8m",      color: "#38bdf8" },
  { label: "Tracking",  value: "DeepSort",      color: "#4ade80" },
  { label: "Teams",     value: "K-Means",       color: "#fb923c" },
  { label: "Backend",   value: "FastAPI",       color: "#c084fc" },
  { label: "Frontend",  value: "React",         color: "#38bdf8" },
  { label: "Styling",   value: "Tailwind v4",   color: "#38bdf8" },
];

export default function Home() {
  const navigate = useNavigate();
  const [status, setStatus] = useState("checking");

  useEffect(() => {
    healthCheck()
      .then(() => setStatus("online"))
      .catch(() => setStatus("offline"));
  }, []);

  const dotColor = {
    checking: "bg-slate-500",
    online:   "bg-emerald-400 shadow-[0_0_8px_#4ade80]",
    offline:  "bg-red-400",
  }[status];

  const textColor = {
    checking: "text-slate-400",
    online:   "text-emerald-400",
    offline:  "text-red-400",
  }[status];

  return (
    <div className="mx-auto max-w-5xl px-6 py-14">

      {/* ── Hero ── */}
      <div className="mb-16 text-center">
        <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-pitch-700 bg-pitch-900 px-4 py-1.5 text-xs text-slate-400">
          <span className={`h-2 w-2 rounded-full ${dotColor}`} />
          Backend: <span className={`font-semibold capitalize ${textColor}`}>{status}</span>
        </div>

        <h1 className="mb-4 text-5xl font-extrabold tracking-tight text-white leading-tight">
          Football Vision{" "}
          <span className="bg-gradient-to-r from-accent-400 to-cyan-300 bg-clip-text text-transparent">
            AI
          </span>
        </h1>
        <p className="mx-auto max-w-lg text-base text-slate-400 leading-relaxed">
          Deep learning-powered match analysis. Upload footage to get player detection,
          team classification, tracking, stats, and heatmaps — all in one click.
        </p>

        <button
          onClick={() => navigate("/upload")}
          className="mt-8 inline-flex items-center gap-2.5 rounded-xl bg-accent-500 px-7 py-3.5 text-sm font-bold text-white shadow-lg shadow-accent-500/25 transition hover:bg-accent-600 hover:shadow-accent-500/40 active:scale-95"
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
          </svg>
          Upload & Detect
        </button>
      </div>

      {/* ── How it works ── */}
      <div className="mb-16">
        <p className="mb-6 text-center text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
          How it works
        </p>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {PIPELINE.map(({ step, title, desc, icon }) => (
            <div
              key={step}
              className="group relative rounded-2xl border border-pitch-700 bg-pitch-900 p-5 transition-all hover:border-accent-500/40 hover:bg-pitch-800"
            >
              <div className="mb-4 flex items-center gap-3">
                <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent-500/10 text-accent-400">
                  {icon}
                </span>
                <span className="text-[10px] font-bold tracking-widest text-slate-600">{step}</span>
              </div>
              <h3 className="mb-1.5 text-sm font-bold text-white">{title}</h3>
              <p className="text-xs leading-relaxed text-slate-500">{desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* ── Tech stack ── */}
      <div className="mb-16">
        <p className="mb-6 text-center text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
          Tech Stack
        </p>
        <div className="flex flex-wrap justify-center gap-3">
          {TECH.map(({ label, value, color }) => (
            <div
              key={label}
              className="flex items-center gap-2.5 rounded-full border border-pitch-700 bg-pitch-900 px-4 py-2"
            >
              <span className="h-2 w-2 rounded-full" style={{ background: color }} />
              <span className="text-xs font-semibold text-white">{value}</span>
              <span className="text-[10px] text-slate-500">{label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ── Detection classes ── */}
      <div>
        <p className="mb-6 text-center text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
          Detection Classes
        </p>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {[
            { emoji: "🏃", label: "Player",     color: "#22d3ee" },
            { emoji: "🧤", label: "Goalkeeper",  color: "#facc15" },
            { emoji: "🟨", label: "Referee",     color: "#c084fc" },
            { emoji: "⚽", label: "Ball",        color: "#f8fafc" },
          ].map(({ emoji, label, color }) => (
            <div
              key={label}
              className="flex items-center gap-3 rounded-xl border border-pitch-700 bg-pitch-900 px-4 py-3.5 transition hover:border-pitch-600"
            >
              <span className="text-xl">{emoji}</span>
              <div>
                <p className="text-sm font-semibold" style={{ color }}>{label}</p>
                <p className="text-[10px] text-slate-600">Class detected</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
