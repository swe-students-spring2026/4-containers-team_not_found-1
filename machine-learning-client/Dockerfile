FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir pipenv

COPY Pipfile Pipfile.lock ./
RUN pipenv install --system --skip-lock

COPY ml_client ./ml_client
COPY .env.example ./.env.example

ENTRYPOINT ["python", "-m", "ml_client.cli"]
CMD ["--help"]
