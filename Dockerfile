FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TZ=Asia/Seoul \
    NEWSLETTER_TIMEZONE=Asia/Seoul \
    PORT=8001 \
    FORWARDED_ALLOW_IPS=127.0.0.1,::1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn==23.0.0

COPY ai_newsletter ./ai_newsletter

RUN pip install --no-cache-dir -e .

RUN mkdir -p /app/data \
    && useradd -u 10001 -r -s /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app/data /app

USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; port = os.getenv('PORT', '8001'); urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health')"

CMD ["sh", "-c", "exec gunicorn -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8001} --workers 1 --timeout 120 --forwarded-allow-ips \"${FORWARDED_ALLOW_IPS:-127.0.0.1,::1}\" 'ai_newsletter.web:create_app()'"]

