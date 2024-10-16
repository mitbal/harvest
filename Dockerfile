FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY data ./data
COPY apps ./apps
COPY harvest ./harvest
COPY main.py ./main.py
COPY home.py ./home.py
COPY README.md ./README.md
COPY .streamlit/config.toml ./.streamlit/config.toml

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
