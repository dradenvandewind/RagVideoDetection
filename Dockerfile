# ──────────────────────────────────────────────
# Stage 1: builder
# ──────────────────────────────────────────────
FROM python:3.12-slim AS builder
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

COPY requirements_yolo.txt .
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python --no-cache -r requirements_yolo.txt

RUN apt-get update && apt-get install -y unzip && \
    curl -fsSL https://deno.land/install.sh | sh && \
    ln -s /root/.deno/bin/deno /usr/local/bin/deno

# ──────────────────────────────────────────────
# Stage 2: runtime
# ──────────────────────────────────────────────
FROM python:3.12-slim AS runtime
LABEL maintainer="erwanleblond@gmail.com"
LABEL description="RAG LlamaIndex async — YouTube + YOLO real-time detection"

# Dépendances système :
#   - ffmpeg       : décodage vidéo HLS pour OpenCV
#   - libgl1       : requis par OpenCV headless
#   - libglib2.0-0 : requis par OpenCV
#   - firefox-esr  : optionnel pour yt-dlp (cookies)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    firefox-esr \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    nodejs \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get install -y \
    nodejs npm \
    chromium \
    libglib2.0-0 libnss3 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxkbcommon0 libxcomposite1 \
    libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2

RUN apt-get update && apt-get install -y nodejs npm && \
    npm install -g youtube-po-token-generator

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN ln -sf $(which node) /usr/local/bin/nodejs

COPY app/ ./app/

RUN mkdir -p /app/chroma_db /app/data

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CHROMA_PATH=/app/chroma_db \
    OLLAMA_MODEL=llama3.2:1b \
    EMBED_MODEL=BAAI/bge-small-en-v1.5 \
    LLM_TEMPERATURE=0.1 \
    CHUNK_SIZE=512 \
    CHUNK_OVERLAP=64 \
    PORT=8000 \
    WORKERS=4 \
    # YOLO : désactive la télémétrie Ultralytics
    YOLO_TELEMETRY=False

#COPY cookies.txt /app/cookies.txt
#RUN chmod 644 /app/cookies.txt

RUN useradd -m -u 1001 raguser && chown -R raguser:raguser /app
USER raguser

EXPOSE $PORT

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD uvicorn app.main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers ${WORKERS} \
    --log-level info
