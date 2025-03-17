FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install build-essential -y --no-install-recommends

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY data ./data
COPY apps ./apps
COPY harvest ./harvest
COPY articles ./articles

COPY main.py ./main.py
COPY home.py ./home.py
COPY prep_index_html.py ./prep_index_html.py
COPY README.md ./README.md
COPY Makefile ./Makefile
COPY style.html ./style.html
COPY feature.html ./feature.html
COPY static/sitemap.txt ./static/sitemap.txt
COPY .streamlit/config.toml ./.streamlit/config.toml

EXPOSE 8501
ENTRYPOINT [ "make", "serve-app" ]
