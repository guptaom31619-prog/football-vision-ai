/**
 * App.js — Root component and route configuration
 *
 * Routes:
 *   /       → Home       (landing page)
 *   /upload → Upload     (image / video detection studio)
 */

import React from "react";
import { Routes, Route } from "react-router-dom";

import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import Upload from "./pages/Upload";

export default function App() {
  return (
    <div className="min-h-screen bg-pitch-950 flex flex-col">
      <Navbar />
      <main className="flex-1">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/upload" element={<Upload />} />
        </Routes>
      </main>
    </div>
  );
}
