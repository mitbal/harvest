run:
	streamlit run main.py

build:
	docker build -t harvest:v1 .

run-docker:
	docker run -p 8501:8501 harvest:v1

serve-app:
	python add_ga.py && streamlit run main.py --server.port=8501 --server.address=0.0.0.0

profile-memory-usage:
	fil-profile run -m streamlit run main.py
