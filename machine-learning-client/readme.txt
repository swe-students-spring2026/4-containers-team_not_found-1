Machine Learning Client (Doodle Recognition)

This subsystem recognizes doodles with a pretrained model and stores prediction
events in MongoDB for downstream visualization in the web app.

Main features:
- pretrained Quick Draw image recognition
- top-k prediction output for a doodle image
- MongoDB persistence for prediction metadata
- HTTP API endpoint for web-app integration

Default pretrained model settings:
- HF_MODEL_ID=ilyesdjerfaf/vit-base-patch16-224-in21k-quickdraw
- HF_TASK=image-classification
- HF_DEVICE=-1 (CPU)

Quick start (no training)

1. Install dependencies
	 pipenv install --dev --python $(which python) --skip-lock

2. Configure environment
	 cp .env.example .env

3. Predict from image with pretrained model
	 pipenv run python -m ml_client.cli predict \
		 --env-file .env \
		 --image-path path/to/doodle.png \
		 --top-k 3

4. Log one inference event to MongoDB
	 pipenv run python -m ml_client.cli run-once \
		 --env-file .env \
		 --image-path path/to/doodle.png \
		 --source local-file-sensor

5. Start HTTP API server (for web app integration)
	 pipenv run python -m ml_client.api

API endpoints:
- GET /health
- POST /predict (raw image bytes, content-type: application/octet-stream)

Optional: override pretrained model
pipenv run python -m ml_client.cli predict \
	--env-file .env \
	--hf-model-id ilyesdjerfaf/vit-base-patch16-224-in21k-quickdraw \
	--hf-task image-classification \
	--image-path path/to/doodle.png

Run tests

pipenv run pytest

Docker

Build image:
docker build -t doodle-ml-client .

Run help:
docker run --rm doodle-ml-client --help
