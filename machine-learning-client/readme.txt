Machine Learning API Client
===========================

Provides a Doodle Recognition HTTP API backed by a Hugging Face pre-trained model and MongoDB.

Features:
- Quick Draw image classification (Top-K)
- MongoDB prediction history (Save, Fetch, Delete)
- Flask HTTP API

Setup & Run:
1. `pipenv install --dev`
2. `cp .env.example .env`
3. `pipenv run python -m ml_client.api`

Testing & Quality:
- Tests: `pipenv run pytest`
- Lint: `pipenv run pylint **/*.py`
- Format: `pipenv run black .`
