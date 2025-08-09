# Makefile for EdgeMind Platform
.PHONY: help build up down restart logs shell clean install test

# Variables
DOCKER_COMPOSE = docker-compose
PYTHON = python
PIP = pip

# Default target
help:
	@echo "EdgeMind Platform - Docker & Development Commands"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make build      - Build Docker images"
	@echo "  make up         - Start all services"
	@echo "  make down       - Stop all services"
	@echo "  make restart    - Restart all services"
	@echo "  make logs       - View logs"
	@echo "  make shell      - Enter web container shell"
	@echo "  make clean      - Remove containers and volumes"
	@echo ""
	@echo "Development Commands:"
	@echo "  make install    - Install Python dependencies"
	@echo "  make test       - Run tests"
	@echo "  make run        - Run Streamlit locally"
	@echo "  make api        - Run FastAPI locally"
	@echo ""
	@echo "Quick Start:"
	@echo "  make quick      - Install deps + start Docker"

# Docker commands
build:
	$(DOCKER_COMPOSE) build

up:
	$(DOCKER_COMPOSE) up -d
	@echo "✅ EdgeMind is running!"
	@echo "📺 Streamlit UI: http://localhost:8501"
	@echo "🚀 FastAPI: http://localhost:8000"
	@echo "📊 API Docs: http://localhost:8000/docs"

down:
	$(DOCKER_COMPOSE) down

restart:
	$(DOCKER_COMPOSE) restart

logs:
	$(DOCKER_COMPOSE) logs -f

shell:
	$(DOCKER_COMPOSE) exec edgemind-web /bin/bash

clean:
	$(DOCKER_COMPOSE) down -v
	docker system prune -f

# Development commands
install:
	$(PIP) install -r requirements.txt
	playwright install chromium

test:
	pytest tests/ -v --cov=src

run:
	streamlit run web/streamlit_app.py

api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Quick start
quick: install build up
	@echo "🎉 EdgeMind is ready!"
	@echo "Visit http://localhost:8501"

# Development workflow
dev:
	@echo "Starting development environment..."
	$(MAKE) install
	$(MAKE) up
	$(MAKE) logs

# Production build
prod:
	docker build -t edgemind:latest -f Dockerfile .
	@echo "Production image built: edgemind:latest"