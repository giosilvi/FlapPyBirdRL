#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./deploy_flappy.sh [DOMAIN]
# Default domain is vibegames.it. Run from the repo root on the server.

DOMAIN="${1:-vibegames.it}"
APP_DIR="$(pwd)"

echo "Domain: $DOMAIN"
echo "Dir:    $APP_DIR"

# 0) Basics
if ! command -v git >/dev/null 2>&1; then
  apt-get update && apt-get install -y git
fi

# 1) Docker
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
fi
if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose plugin not found; ensure recent Docker installed." >&2
  exit 1
fi

cd "$APP_DIR"

# 2) Ensure deploy files (only create if missing)
if [ ! -f requirements.deploy.txt ]; then
  cat > requirements.deploy.txt <<'R'
fastapi==0.115.2
uvicorn[standard]==0.30.6
numpy>=1.24
pygame==2.4.0
torch==2.4.0
R
fi

if [ ! -f Dockerfile ]; then
  cat > Dockerfile <<'D'
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libasound2 ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.deploy.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.deploy.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu
COPY . .
EXPOSE 8765
CMD ["uvicorn","server.ai_server:app","--host","0.0.0.0","--port","8765"]
D
fi

if [ ! -f docker-compose.yml ]; then
  cat > docker-compose.yml <<'C'
services:
  app:
    build: .
    container_name: flappy-ai
    restart: unless-stopped
    expose:
      - "8765"
  caddy:
    image: caddy:2
    container_name: caddy
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
    depends_on:
      - app
C
fi

if [ ! -f Caddyfile ]; then
  cat > Caddyfile <<C2
${DOMAIN}, www.${DOMAIN} {
    encode zstd gzip
    reverse_proxy app:8765
}
C2
else
  # Update domain in existing file header
  sed -i "1s|.*|${DOMAIN}, www.${DOMAIN} {|" Caddyfile || true
fi

# 3) Sanity: checkpoint presence
if [ ! -f checkpoints/best.pt ]; then
  echo "WARNING: checkpoints/best.pt not found in repo; the AI loop will not start." >&2
fi

# 4) Build + up
docker compose build app --no-cache
docker compose up -d

echo "-----"
echo "Deployment started. Check logs:"
echo "  docker logs -f flappy-ai"
echo "  docker logs -f caddy"
echo "Open: https://${DOMAIN}"
