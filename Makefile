.PHONY: ingest transform validate pipeline test clean docker-up

ingest:
	python -m src.ingestion

transform:
	python -m src.transform

validate:
	python -m src.validate

pipeline:
	python src/pipeline.py

pipeline-dry:
	python src/pipeline.py --dry-run

test:
	pytest tests/ -v

clean:
	rm -rf data/raw data/processed data/quality data/enriched data/anomalies data/agent

docker-up:
	docker-compose run --rm app pytest
