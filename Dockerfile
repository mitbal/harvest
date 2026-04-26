# Stage 1: Build stage
FROM python:3.12-slim AS builder

# Install uv for extremely fast dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/uv

WORKDIR /app

# Install build dependencies (needed for some python wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies into a temporary directory
COPY requirements.txt .
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    /uv/bin/uv pip install --system --prefix=/install -r requirements.txt

# Stage 2: Final stage
FROM python:3.12-slim

# Install make so the ENTRYPOINT can run the Makefile target
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the installed python packages from the builder stage
COPY --from=builder /install /usr/local

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

# Now 'make' will be available at runtime
ENTRYPOINT [ "make", "serve-app" ]

