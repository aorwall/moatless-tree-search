# Docker image settings
DOCKER_REPO := aorwall/moatless-tools-app

# Docker Compose settings
COMPOSE_FILE := docker-compose.yml

# Environment file
ENV_FILE ?= .env

.PHONY: help run dev stop logs status 

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

run: ## Start all services
	@echo "Starting services..."
	docker-compose --env-file $(ENV_FILE) up -d

dev: ## Start services in development mode (with logs)
	@echo "Starting services in development mode..."
	docker-compose --env-file $(ENV_FILE) up

stop: ## Stop all services
	@echo "Stopping services..."
	docker-compose down

logs: ## Show logs from all services
	docker-compose logs -f

status: ## Show status of all services
	docker-compose ps

# Default target
all: run 