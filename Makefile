run:
	streamlit run main.py

build:
	docker build -t harvest:v1 .

run-docker:
	docker run -p 8501:8501 harvest:v1
