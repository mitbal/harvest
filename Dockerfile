FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install build-essential -y --no-install-recommends

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY data ./data
COPY apps ./apps
COPY harvest ./harvest
COPY main.py ./main.py
COPY home.py ./home.py
COPY add_ga.py ./add_ga.py
COPY README.md ./README.md
COPY Makefile ./Makefile
COPY .streamlit/config.toml ./.streamlit/config.toml

EXPOSE 8501
ENTRYPOINT [ "make", "serve-app" ]
