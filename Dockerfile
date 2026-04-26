# Stage 1: Build stage
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Final stage
FROM python:3.12-slim

WORKDIR /app

# Copy only the installed python packages from the builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

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

# Copy data - Consider if all data is needed in the image
# If some data files are very large, they might be better mounted as volumes
COPY data ./data

EXPOSE 8501
ENTRYPOINT [ "make", "serve-app" ]
