# =============================================================================
# Makefile — Football Vision AI
#
# Usage (run from project root):
#   make install       — create venv + install backend & frontend deps
#   make dev           — start backend + frontend in parallel
#   make backend       — start FastAPI server on :8000
#   make frontend      — start React dev server on :3000
#   make train         — run YOLOv8 training pipeline
#   make download      — download & prepare Kaggle dataset
#   make test          — run full end-to-end system test
#   make detect        — run single-image detection test
#   make heatmaps      — open the heatmaps folder
#   make kill          — kill processes on ports 8000 & 3000
#   make clean         — remove caches, temp files, outputs
#   make docker-up     — build & start Docker containers
#   make docker-down   — stop Docker containers
# =============================================================================

SHELL        := /bin/bash
BACKEND_DIR  := backend
FRONTEND_DIR := frontend
VENV         := $(BACKEND_DIR)/venv
ACTIVATE     := source $(VENV)/bin/activate

.PHONY: help install install-backend install-frontend \
        dev backend frontend kill kill-backend \
        test train download detect heatmaps clean lint \
        docker docker-up docker-down docker-build

# ---- Default target --------------------------------------------------------

help:
	@echo ""
	@echo "  Football Vision AI — available commands"
	@echo "  ───────────────────────────────────────────"
	@echo "  make install          Install all dependencies (backend + frontend)"
	@echo "  make dev              Start backend & frontend together"
	@echo "  make backend          Start FastAPI server (port 8000)"
	@echo "  make frontend         Start React dev server (port 3000)"
	@echo "  make train            Run YOLOv8 training pipeline"
	@echo "  make download         Download & prepare Kaggle dataset"
	@echo "  make test             Run full end-to-end system test"
	@echo "  make detect           Test detection on a sample image"
	@echo "  make lint             Run ruff lint on backend code"
	@echo "  make heatmaps         Open generated heatmaps folder"
	@echo "  make kill             Kill backend & frontend processes"
	@echo "  make clean            Remove caches, temp files, outputs"
	@echo "  make docker-up        Build & start Docker containers"
	@echo "  make docker-down      Stop Docker containers"
	@echo ""

# ---- Installation ----------------------------------------------------------

install: install-backend install-frontend
	@echo "\n✅ All dependencies installed.\n"

install-backend:
	@echo "→ Setting up Python virtual environment…"
	@test -d $(VENV) || python3 -m venv $(VENV)
	@$(ACTIVATE) && pip install --upgrade pip -q && pip install -r $(BACKEND_DIR)/requirements.txt -q
	@echo "✅ Backend dependencies installed."

install-frontend:
	@echo "→ Installing frontend packages…"
	@cd $(FRONTEND_DIR) && npm install --silent
	@echo "✅ Frontend dependencies installed."

# ---- Development -----------------------------------------------------------

dev:
	@echo "→ Starting backend + frontend…"
	@make kill 2>/dev/null || true
	@$(ACTIVATE) && cd $(BACKEND_DIR) && uvicorn api:app --port 8000 --reload --timeout-keep-alive 600 &
	@cd $(FRONTEND_DIR) && npm start &
	@echo "\n🟢 Backend  → http://localhost:8000"
	@echo "🟢 Frontend → http://localhost:3000"
	@echo "   Press Ctrl+C then 'make kill' to stop.\n"
	@wait

backend:
	@make kill-backend 2>/dev/null || true
	@echo "→ Starting FastAPI server…"
	@$(ACTIVATE) && cd $(BACKEND_DIR) && uvicorn api:app --port 8000 --reload --timeout-keep-alive 600

frontend:
	@echo "→ Starting React dev server…"
	@cd $(FRONTEND_DIR) && npm start

# ---- Training & data -------------------------------------------------------

train:
	@echo "→ Starting YOLOv8 training…"
	@$(ACTIVATE) && cd $(BACKEND_DIR) && python train.py

download:
	@echo "→ Downloading dataset from Kaggle…"
	@$(ACTIVATE) && cd $(BACKEND_DIR) && python download_dataset.py

# ---- Testing & linting -----------------------------------------------------

test:
	@echo "→ Running end-to-end system test…"
	@$(ACTIVATE) && cd $(BACKEND_DIR) && python system_test.py

detect:
	@echo "→ Running detection test…"
	@$(ACTIVATE) && cd $(BACKEND_DIR) && python detect.py

lint:
	@echo "→ Running ruff lint…"
	@$(ACTIVATE) && cd $(BACKEND_DIR) && ruff check .
	@echo "✅ Lint passed."

# ---- Heatmaps -------------------------------------------------------------

heatmaps:
	@open $(BACKEND_DIR)/heatmaps 2>/dev/null || echo "No heatmaps folder found. Run a video detection first."

# ---- Cleanup ---------------------------------------------------------------

kill-backend:
	@lsof -ti :8000 | xargs kill -9 2>/dev/null || true

kill:
	@echo "→ Killing processes on ports 8000 & 3000…"
	@lsof -ti :8000 | xargs kill -9 2>/dev/null || true
	@lsof -ti :3000 | xargs kill -9 2>/dev/null || true
	@echo "✅ Ports freed."

clean:
	@echo "→ Cleaning up…"
	@find $(BACKEND_DIR) -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find $(BACKEND_DIR) -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@rm -rf $(BACKEND_DIR)/.ruff_cache 2>/dev/null || true
	@rm -rf $(BACKEND_DIR)/heatmaps/*.png 2>/dev/null || true
	@rm -rf $(BACKEND_DIR)/outputs/*.mp4 $(BACKEND_DIR)/outputs/*.jpg 2>/dev/null || true
	@rm -rf $(FRONTEND_DIR)/build 2>/dev/null || true
	@echo "✅ Cleaned."

# ---- Docker ---------------------------------------------------------------

docker-build:
	@echo "→ Building Docker images…"
	@docker compose build

docker-up: docker-build
	@echo "→ Starting containers…"
	@docker compose up -d
	@echo "\n🟢 Backend  → http://localhost:8000"
	@echo "🟢 Frontend → http://localhost:3000\n"

docker-down:
	@echo "→ Stopping containers…"
	@docker compose down
	@echo "✅ Stopped."

docker: docker-up
