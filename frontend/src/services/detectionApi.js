import axios from "axios";

export const BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 0,
});

/**
 * Send an image file and return detection results.
 */
export async function detectImage(file) {
  const form = new FormData();
  form.append("file", file);

  const { data } = await api.post("/detect/image", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return data;
}

/**
 * Send a video file. The backend streams NDJSON progress lines followed by the
 * final result. We read the stream incrementally so the UI can show real-time
 * progress.
 *
 * @param {File}     file       - Video file
 * @param {function} onProgress - Called with { percent, frame, total, sampled }
 * @returns {Promise<object>}  - Final detection result payload
 */
export async function detectVideo(file, onProgress) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${BASE_URL}/detect/video`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Server returned ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResult = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Split on newlines — each complete line is one JSON object
    const lines = buffer.split("\n");
    // Keep the last (possibly incomplete) chunk in the buffer
    buffer = lines.pop();

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      try {
        const msg = JSON.parse(trimmed);

        if (msg.type === "progress" && onProgress) {
          onProgress(msg);
        } else if (msg.type === "result") {
          finalResult = msg;
        } else if (msg.type === "error") {
          throw new Error(msg.detail || "Video processing failed.");
        }
      } catch (e) {
        if (e instanceof SyntaxError) continue; // skip malformed lines
        throw e;
      }
    }
  }

  if (!finalResult) {
    throw new Error("No result received from server.");
  }

  return finalResult;
}

/**
 * Ping the health check endpoint.
 */
export async function healthCheck() {
  const { data } = await api.get("/");
  return data;
}
