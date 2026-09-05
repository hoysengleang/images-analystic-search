FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Model weights land on a named volume instead of the image layer.
    HF_HOME=/opt/model-cache \
    TORCH_HOME=/opt/model-cache/torch \
    XDG_CACHE_HOME=/opt/model-cache/xdg \
    HOME=/home/openvisionsearch

WORKDIR /app

COPY requirements.txt .

# PyPI's Linux ARM64 torch wheel may pull the CUDA toolkit even though the
# default image runs inference on CPU. Install the matching official CPU wheels
# first; the general dependency pass below then reuses them on both amd64 and
# arm64 instead of adding several gigabytes of unused GPU libraries.
ARG TORCH_VERSION=2.13.0
ARG TORCHVISION_VERSION=0.28.0
RUN pip install --no-cache-dir \
        "torch==${TORCH_VERSION}+cpu" \
        "torchvision==${TORCHVISION_VERSION}+cpu" \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

RUN groupadd --system --gid 10001 openvisionsearch \
    && useradd --system --uid 10001 --gid openvisionsearch \
        --create-home --home-dir /home/openvisionsearch openvisionsearch \
    && mkdir -p /data/images /data/state /opt/model-cache \
    && chown -R openvisionsearch:openvisionsearch \
        /data /opt/model-cache /home/openvisionsearch

COPY --chown=openvisionsearch:openvisionsearch app ./app

# The service does not need root privileges. Runtime state and model weights
# are mounted at paths owned by this dedicated, non-login user.
USER openvisionsearch

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:8000' + os.getenv('API_PREFIX', '') + '/health', timeout=3).close()"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
