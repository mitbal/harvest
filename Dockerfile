# Stage 1: Build stage
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies + uv via pip (more reliable than ghcr.io in Railway)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Install python dependencies into a virtual env
COPY requirements.txt .
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python -r requirements.txt

# Stage 2: Final stage
FROM python:3.12-slim

# Install make so the ENTRYPOINT can run the Makefile target, and jemalloc (see LD_PRELOAD below)
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    libjemalloc2 \
    && rm -rf /var/lib/apt/lists/*

# glibc malloc gives each thread its own arena and rarely hands freed pages back, so every Streamlit rerun
# (a fresh thread) ratchets RSS up while Python's own numbers stay flat. jemalloc returns idle pages instead.
# jemalloc sizes its arena count from the host's CPUs, not the container's share, and with the defaults RSS
# settled near 1.5 GB (glibc: ~0.8 GB). Two arenas, a 1 s dirty decay and no muzzy stage hand pages back fast.
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
ENV MALLOC_CONF=background_thread:true,narenas:2,dirty_decay_ms:1000,muzzy_decay_ms:0

WORKDIR /app

# Copy the virtual env from builder
COPY --from=builder /opt/venv /opt/venv

# Make sure the venv is on PATH
ENV PATH="/opt/venv/bin:$PATH"

# Copy application files
COPY apps ./apps
COPY asset ./asset
COPY static ./static
COPY harvest ./harvest
COPY articles ./articles
COPY main.py ./main.py
COPY server.py ./server.py
COPY templates ./templates
COPY home.py ./home.py
COPY prep_index_html.py ./prep_index_html.py
COPY README.md ./README.md
COPY Makefile ./Makefile
COPY style.html ./style.html
COPY feature.html ./feature.html
COPY readme_style.html ./readme_style.html
COPY footer.html ./footer.html
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY data ./data

EXPOSE 8501

ENTRYPOINT [ "make", "serve-app" ]
