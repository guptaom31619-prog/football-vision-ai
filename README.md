# Football Vision AI

[![CI](https://github.com/guptaom31619-prog/football-vision-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/guptaom31619-prog/football-vision-ai/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![React 18](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)

**Demo project** — computer-vision pipeline for football match footage.

Upload an image or video clip and get player detection, team classification, tracking IDs, possession stats, and pitch heatmaps.

> Built as a portfolio / learning demo, not a production SaaS deploy.

---

## What it does

| Capability | How |
|---|---|
| **Detection** | YOLOv8 finds players, goalkeepers, referees, and the ball |
| **Team classification** | K-Means on jersey colors with centroid lock across frames |
| **Tracking** | DeepSort assigns stable player IDs |
| **Stats** | Possession %, pitch coverage, attacking-third %, class counts |
| **Heatmaps** | Per-team and combined position heatmaps on a pitch |
| **UI** | React dashboard with upload, live progress, and downloads |

---

## Architecture

```
football-vision-ai/
├── backend/                 # FastAPI + YOLO pipeline
│   ├── api.py               # REST endpoints
│   ├── detect.py            # Inference + team + tracking
│   ├── train.py             # YOLOv8 training
│   ├── tracker.py           # DeepSort wrapper
│   ├── team_classifier.py   # Jersey-color K-Means
│   ├── stats_engine.py      # Match statistics
│   ├── heatmap.py           # Pitch heatmaps
│   ├── download_dataset.py  # Kaggle / Roboflow fetch
│   ├── system_test.py       # End-to-end local test
│   ├── football.yaml        # YOLO dataset config
│   └── requirements.txt
├── frontend/                # React 18 + Tailwind dashboard
│   └── src/
│       ├── pages/           # Home, Upload
│       ├── components/      # Navbar, UploadBox, Canvas
│       └── services/        # API client
├── .github/workflows/ci.yml # Lint + build on every push
├── docker-compose.yml
├── Makefile
└── README.md
```

**Flow:** Upload → YOLOv8 detect → team classify → DeepSort track → stats + heatmaps → annotated media.

---

## Quick start

### Requirements

- Python **3.11+**
- Node.js **20+**
- npm

### Install

```bash
git clone https://github.com/guptaom31619-prog/football-vision-ai.git
cd football-vision-ai
make install
```

### Configure (optional)

Only needed if you want to download a training dataset from Kaggle:

```bash
cp backend/.env.example backend/.env
# Add KAGGLE_USERNAME / KAGGLE_KEY
```

Without a custom model, inference falls back to pretrained `yolov8n.pt`.

### Run locally

```bash
make dev
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |

---

## Docker

```bash
make docker-up      # build + start
make docker-down    # stop
```

---

## Train your own model (optional)

```bash
make download   # fetch dataset (needs Kaggle credentials)
make train      # train YOLOv8 (~50 epochs)
```

Weights land under `backend/runs/` (gitignored).

---

## Makefile commands

```bash
make install        # backend venv + frontend npm install
make dev            # run both servers
make backend        # FastAPI only (:8000)
make frontend       # React only (:3000)
make lint           # ruff on backend
make test           # end-to-end system test (needs sample media)
make train          # YOLO training
make download       # dataset download
make clean          # caches + generated outputs
make docker-up      # compose up
make docker-down    # compose down
```

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/detect/image` | Image → detections + stats + annotated JPG |
| `POST` | `/detect/video` | Video → NDJSON progress + annotated MP4 + heatmaps |
| `GET` | `/heatmaps/{file}` | Heatmap PNG |
| `GET` | `/outputs/{file}` | Annotated media |

```bash
curl -X POST http://localhost:8000/detect/image \
  -F "file=@match_frame.jpg"
```

Interactive docs: http://localhost:8000/docs

---

## Detection classes

| ID | Class |
|---|---|
| 0 | Player |
| 1 | Ball |
| 2 | Goalkeeper |
| 3 | Referee |

---

## Tech stack

| Layer | Stack |
|---|---|
| Backend | Python 3.11, FastAPI, Ultralytics YOLOv8, OpenCV, DeepSort, scikit-learn |
| Frontend | React 18, Tailwind CSS v4, Axios, React Router |
| Tooling | Docker Compose, Makefile, GitHub Actions CI |

---

## CI

Every push / PR to `main` runs:

1. **Backend** — Python syntax, Ruff lint, YAML + requirements validation  
2. **Frontend** — `npm ci`, ESLint, production build  
3. **Structure** — critical files present  

---

## Notes for demos

- Model weights (`.pt`), datasets, `venv/`, `node_modules/`, and generated outputs are **gitignored**.
- Never commit `.env` — only `.env.example` templates are tracked.
- This repo is intentionally a **local demo**. Docker is included for convenience, not as a cloud deploy target.

---

## License

[MIT](LICENSE)
