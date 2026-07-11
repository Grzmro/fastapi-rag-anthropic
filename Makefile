.PHONY: install serve test docker-build docker-up docker-down

install:
	pip install -r requirements.txt

serve:
	uvicorn app.main:app --reload --port 8000

test:
	pytest -v

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down
