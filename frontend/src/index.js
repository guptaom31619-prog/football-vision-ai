/**
 * index.js — React application entry point
 *
 * Mounts the app into #root, wraps it with BrowserRouter for client-side
 * routing, and imports the global Tailwind stylesheet.
 */

import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import "./index.css";

const root = ReactDOM.createRoot(document.getElementById("root"));

root.render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);
