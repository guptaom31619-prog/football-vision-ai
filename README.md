# Football Vision AI

Real-time football player detection, team classification, player tracking, match statistics, and heatmap generation — powered by YOLOv8, DeepSort, and K-Means clustering.

---

## Features

| Feature | Description |
|---------|-------------|
| **Object Detection** | YOLOv8-based detection of players, goalkeepers, referees, and the ball (conf ≥ 0.5, IoU 0.4) |
| **Team Classification** | Stateful jersey-color classifier — K-Means initialises centroids on the first frame, then nearest-centroid assignment locks teams consistently across all subsequent frames (no random flipping) |
| **Player Tracking** | DeepSort multi-object tracker assigns stable IDs across video frames |
| **Match Statistics** | Player counts (median-capped at 11), ball possession %, pitch coverage %, attacking third % |
| **Possession Timeline** | Per-bucket possession chart showing how possession shifts over the match |
| **Heatmaps** | Per-team and combined position heatmaps on a rendered football pitch |
| **Annotated Output** | Bounding-boxed annotated video (H.264) and annotated images returned from API |
| **Download Buttons** | One-click download for annotated video, images, and heatmap PNGs |
| **REST API** | FastAPI backend with image and video endpoints, streaming NDJSON progress |
| **React Dashboard** | Dark sports-analytics UI with upload, progress bar, and rich results |
| **Docker** | Production-ready `docker-compose` setup with nginx + FastAPI |
| **CI/CD** | GitHub Actions pipeline — lint, syntax check, build, structure validation |

---

## Project Structure

```
football-vision-ai/
├── backend/
│   ├── api.py                 # FastAPI server (REST endpoints)
│   ├── detect.py              # YOLO inference + team + tracking pipeline
│   ├── train.py               # YOLOv8 training pipeline
│   ├── tracker.py             # DeepSort player tracking module
│   ├── team_classifier.py     # Stateful jersey-color team classifier (K-Means + centroid lock)
│   ├── stats_engine.py        # Match statistics engine
│   ├── heatmap.py             # Player heatmap generator
│   ├── download_dataset.py    # Kaggle dataset download + YOLO format conversion
│   ├── system_test.py         # End-to-end test script
│   ├── football.yaml          # YOLOv8 dataset configuration
│   ├── ruff.toml              # Python linter config
│   ├── Dockerfile             # Backend container image
│   ├── .dockerignore          # Docker build exclusions
│   ├── requirements.txt       # Python dependencies (pinned versions)
│   └── .env.example           # Environment variable template
├── frontend/
│   ├── src/
│   │   ├── App.js             # Route configuration
│   │   ├── index.js           # React entry point
│   │   ├── tailwind.css       # Tailwind CSS source
│   │   ├── pages/
│   │   │   ├── Home.js        # Landing page
│   │   │   └── Upload.js      # Detection studio (upload + results)
│   │   ├── components/
│   │   │   ├── Navbar.js      # Navigation bar
│   │   │   ├── UploadBox.js   # File upload widget
│   │   │   └── DetectionCanvas.js # Bounding box overlay canvas
│   │   └── services/
│   │       └── detectionApi.js # API client (Axios + fetch stream)
│   ├── public/
│   │   └── index.html         # HTML template
│   ├── Dockerfile             # Frontend container image (nginx)
│   ├── .dockerignore          # Docker build exclusions
│   ├── nginx.conf             # Nginx config for SPA + API proxy
│   ├── package.json
│   ├── package-lock.json
│   └── .env.example           # Frontend env template
├── docker-compose.yml         # Multi-container orchestration
├── .github/workflows/ci.yml   # GitHub Actions CI pipeline
├── Makefile                   # Project commands
├── .gitignore
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- npm

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/football-vision-ai.git
cd football-vision-ai
make install
```

### 2. Configure environment

```bash
cp backend/.env.example backend/.env
# Edit backend/.env — only needed if you want to download the Kaggle dataset
```

### 3. Download dataset and train (optional)

```bash
make download    # Download football dataset from Kaggle
make train       # Train YOLOv8 model (~50 epochs)
```

> The system falls back to a pretrained `yolov8n.pt` if no custom model is found.

### 4. Run the app

```bash
make dev
```

This starts both servers:
- **Backend** → http://localhost:8000
- **Frontend** → http://localhost:3000

---

## Docker

Build and run the entire stack in containers:

```bash
make docker-up     # Build images + start containers
make docker-down   # Stop and remove containers
```

Or directly with Docker Compose:

```bash
docker compose up --build -d
```

| Service | Port | Description |
|---------|------|-------------|
| Frontend | 3000 | nginx serving React build |
| Backend | 8000 | FastAPI + Uvicorn |

The backend container includes `ffmpeg` for H.264 video re-encoding.

---

## Available Commands

```bash
make install        # Install all dependencies (backend + frontend)
make dev            # Start backend + frontend together
make backend        # Start only the FastAPI server
make frontend       # Start only the React dev server
make test           # Run end-to-end system test
make lint           # Run ruff lint on backend code
make train          # Run YOLOv8 training pipeline
make download       # Download dataset from Kaggle
make detect         # Run single-image detection test
make heatmaps       # Open generated heatmaps folder
make kill           # Kill processes on ports 8000 & 3000
make clean          # Remove caches, temp files, outputs
make docker-up      # Build & start Docker containers
make docker-down    # Stop Docker containers
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/detect/image` | Upload image → annotated image + detections + stats |
| `POST` | `/detect/video` | Upload video → NDJSON stream (progress + annotated video + stats + heatmaps) |
| `GET` | `/heatmaps/{filename}` | Serve a generated heatmap PNG |
| `GET` | `/outputs/{filename}` | Serve annotated video (MP4) or image (JPG) |

**Example — image detection:**

```bash
curl -X POST http://localhost:8000/detect/image \
  -F "file=@test_image.jpg"
```

**Response:**

```json
{
  "detections": [
    {
      "label": "player",
      "id": null,
      "team": "A",
      "confidence": 0.92,
      "bbox": [120.5, 45.2, 210.8, 310.6]
    }
  ],
  "stats": {
    "frames_analysed": 1,
    "players_team_A": 6,
    "players_team_B": 5,
    "referees_on_field": 1,
    "ball_possession": "Team A",
    "possession_pct": { "A": 100.0, "B": 0.0 },
    "ball_visibility_pct": 100.0,
    "pitch_coverage_pct": { "A": 12.3, "B": 8.7 },
    "attacking_third_pct": { "A": 0.0, "B": 0.0 },
    "class_counts": { "player": 9, "goalkeeper": 2, "referee": 1, "ball": 1 },
    "avg_confidence": { "player": 0.87, "goalkeeper": 0.82, "referee": 0.76, "ball": 0.91 },
    "possession_timeline": []
  },
  "annotated_image": "annotated_1741347600.jpg"
}
```

---

## Detection Classes

| ID | Class | Max per frame |
|----|-------|---------------|
| 0 | Player | 11 per team |
| 1 | Ball | 1 |
| 2 | Goalkeeper | Counted with players |
| 3 | Referee | 4 |

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Backend** | Python 3.11, FastAPI, Ultralytics YOLOv8, OpenCV, DeepSort, scikit-learn, Matplotlib, imageio-ffmpeg |
| **Frontend** | React 18, Tailwind CSS v4, Axios, React Router v6 |
| **Infrastructure** | Docker, Docker Compose, nginx, GitHub Actions CI |

---

## Environment Variables

> **Important:** Never commit `.env` files. Only `.env.example` templates are tracked in git.

### Backend (`backend/.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `KAGGLE_USERNAME` | For dataset download only | Kaggle API username |
| `KAGGLE_KEY` | For dataset download only | Kaggle API key |
| `RF_API_KEY` | Optional | Roboflow API key (alternative dataset source) |

### Frontend (`frontend/.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `REACT_APP_API_URL` | No | Backend URL (defaults to `http://localhost:8000`) |

---

## Security

- **`.env` files are gitignored** — secrets never enter version control.
- **`.env.example` files contain no real values** — only blank templates.
- **Model weights (`.pt`) are gitignored** — large binary files stay local.
- **Generated outputs are gitignored** — `heatmaps/`, `outputs/`, `test_data/`.
- **Auto-cleanup** — the server deletes output files older than 24 hours on startup.
- **CORS** — currently allows all origins (`*`). Restrict in production.
- For GCP/Vercel/cloud deployment, use platform-level environment variable configuration (e.g., GCP Secret Manager, Vercel Environment Variables) instead of `.env` files.

---

## Deployment

### GCP (Cloud Run / Compute Engine)

1. Push to GitHub (all secrets are gitignored).
2. Set environment variables via GCP Console or `gcloud` CLI.
3. Deploy backend as a Cloud Run service or via `docker compose` on a VM.
4. Set `REACT_APP_API_URL` to your backend URL before building the frontend.

### Vercel (frontend only)

1. Import the `frontend/` directory as a Vercel project.
2. Set `REACT_APP_API_URL` in Vercel Environment Variables.
3. Build command: `npm run build`, output: `build/`.

### General

- Old outputs are auto-cleaned on server startup (files > 24h).
- For production, restrict CORS origins in `api.py`.
- Model weights must be provided separately (train locally or download).

---

## Running Tests

```bash
make test
```

The system test validates:
1. Trained model file exists
2. Local detection pipeline returns correct schema
3. FastAPI server starts and responds
4. `POST /detect/image` returns detections + stats + annotated image
5. `POST /detect/video` returns per-frame results + heatmaps + annotated video

---

## License

MIT
