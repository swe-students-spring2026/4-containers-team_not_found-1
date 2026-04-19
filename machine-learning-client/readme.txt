Machine Learning Client (Doodle Recognition)

Main features:
- pretrained Quick Draw image recognition
- top-k prediction output for a doodle image
- MongoDB persistence for prediction metadata
- HTTP API endpoint for web-app integration

Default pretrained model settings:
- HF_MODEL_ID=ilyesdjerfaf/vit-base-patch16-224-in21k-quickdraw
- HF_TASK=image-classification
- HF_DEVICE=-1 (CPU)

1. Install dependencies
	 pipenv install --dev --python $(which python) --skip-lock

2. Configure environment
	 cp .env.example .env

3. Start API server (for web app integration)
	 pipenv run python -m ml_client.api

API endpoints:
- GET /health
- POST /predict

Run tests

pipenv run pytest

Docker

Build image:
docker build -t doodle-ml-client .

Run help:
docker run --rm doodle-ml-client --help
