FROM python:3.10

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install pipenv
RUN pip install --no-cache-dir pipenv

# Copy Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock ./

# Install dependencies system-wide
RUN pipenv install --system --deploy

# Install CPU-specific PyTorch explicitly to override any GPU defaults
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Set environment variable defaults
ENV ML_CLIENT_HOST=0.0.0.0 \
    ML_CLIENT_PORT=8000

# Command to run the application
CMD ["python", "-m", "ml_client.api"]
