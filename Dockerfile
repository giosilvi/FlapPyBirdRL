FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SDL_AUDIODRIVER=dummy

# System deps that help pygame run headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libasound2 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/FlapPyBird

# Install Python deps (Torch CPU via extra index)
COPY requirements.deploy.txt /app/FlapPyBird/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.deploy.txt \
       --extra-index-url https://download.pytorch.org/whl/cpu

# Copy project
COPY . /app/FlapPyBird

EXPOSE 8765

CMD ["uvicorn", "server.ai_server:app", "--host", "0.0.0.0", "--port", "8765"]
