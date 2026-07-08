MAMBA ?= /opt/homebrew/Caskroom/miniconda/base/condabin/mamba
ENV ?= ai_infra
HOST ?= 127.0.0.1
BACKEND_PORT ?= 8010

.PHONY: help backend backend-reload frontend dev install-backend install-frontend test build health

help:
	@echo "PulseGraph commands:"
	@echo "  make backend        Start backend on $(HOST):$(BACKEND_PORT)"
	@echo "  make backend-reload Start backend with reload"
	@echo "  make frontend       Start Vite frontend"
	@echo "  make dev            Start backend + frontend together"
	@echo "  make test           Run backend and frontend tests"
	@echo "  make build          Build frontend"
	@echo "  make health         Check backend health"

backend:
	$(MAMBA) run -n $(ENV) python -m uvicorn app.main:app --app-dir backend --host $(HOST) --port $(BACKEND_PORT)

backend-reload:
	$(MAMBA) run -n $(ENV) python -m uvicorn app.main:app --app-dir backend --reload --host $(HOST) --port $(BACKEND_PORT)

frontend:
	cd frontend && npm run dev

dev:
	$(MAKE) backend-reload & $(MAKE) frontend

install-backend:
	$(MAMBA) run -n $(ENV) python -m pip install -r backend/requirements.txt

install-frontend:
	cd frontend && npm install

test:
	$(MAMBA) run -n $(ENV) python -m pytest backend/tests
	cd frontend && npm test -- --run

build:
	cd frontend && npm run build

health:
	curl -s http://$(HOST):$(BACKEND_PORT)/health
	@printf "\n"
