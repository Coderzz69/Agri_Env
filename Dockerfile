FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ENABLE_WEB_INTERFACE=true \
    PORT=8000

WORKDIR /app

COPY pyproject.toml README.md openenv.yaml inference.py /app/
COPY agri_env /app/agri_env
COPY server /app/server
COPY tests /app/tests

RUN pip install --upgrade pip && \
    pip install .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

CMD ["python", "-m", "server.app", "--port", "8000"]
