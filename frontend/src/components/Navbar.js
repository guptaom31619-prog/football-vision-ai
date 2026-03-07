/**
 * Navbar.js — Persistent top navigation bar
 *
 * Renders the brand logo and navigation links for all top-level pages.
 * Uses NavLink from React Router to highlight the active route.
 * Remains mounted across all page transitions (rendered once in App.js).
 */

import React from "react";
import { NavLink } from "react-router-dom";

const NAV_LINKS = [
  { to: "/", label: "Home" },
  { to: "/upload", label: "Upload" },
];

export default function Navbar() {
  return (
    <header className="sticky top-0 z-50 border-b border-pitch-700 bg-pitch-900/90 backdrop-blur-sm">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-6">

        {/* Brand */}
        <NavLink to="/" className="flex items-center gap-2.5 group">
          {/* Football icon */}
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent-500 text-white font-bold text-sm shadow-lg shadow-accent-500/30">
            ⚽
          </span>
          <span className="text-sm font-bold tracking-wide text-white">
            Football<span className="text-accent-400">Vision</span> AI
          </span>
        </NavLink>

        {/* Nav links */}
        <nav>
          <ul className="flex items-center gap-1">
            {NAV_LINKS.map(({ to, label }) => (
              <li key={to}>
                <NavLink
                  to={to}
                  end={to === "/"}
                  className={({ isActive }) =>
                    [
                      "rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                      isActive
                        ? "bg-pitch-800 text-accent-400"
                        : "text-slate-400 hover:text-slate-100 hover:bg-pitch-800",
                    ].join(" ")
                  }
                >
                  {label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </header>
  );
}
