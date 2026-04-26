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

# Install make so the ENTRYPOINT can run the Makefile target
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the virtual env from builder
COPY --from=builder /opt/venv /opt/venv

# Make sure the venv is on PATH
ENV PATH="/opt/venv/bin:$PATH"

# Copy application files
COPY apps ./apps
COPY static ./static
COPY harvest ./harvest
COPY articles ./articles
COPY main.py ./main.py
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
