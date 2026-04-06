# ============================================================
# Resume Scorer OpenEnv – Dockerfile
# ============================================================
# Build : docker build -t resume-scorer-env .
# Run   : docker run -p 7860:7860 resume-scorer-env
# Validate: docker run resume-scorer-env python validate.py
# Baseline: docker run -e OPENAI_API_KEY=$OPENAI_API_KEY \
#               resume-scorer-env python -m baseline.inference
# ============================================================

FROM python:3.11-slim

LABEL maintainer="resume-scorer-env"
LABEL openenv="true"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["python", "app.py"]
