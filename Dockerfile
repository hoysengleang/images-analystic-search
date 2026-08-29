FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Model weights land on a named volume instead of the image layer.
    HF_HOME=/opt/model-cache

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# ALLOWED_IMAGE_ROOT and COLLECTION_METADATA_PATH default to this volume.
RUN mkdir -p /data/images /opt/model-cache

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host ${APP_HOST:-0.0.0.0} --port ${APP_PORT:-8000}"]
