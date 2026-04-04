FROM python:3.11-slim

WORKDIR /app

# git is required to pip-install openenv-core from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as non-root for security (OWASP container hardening)
RUN adduser --disabled-password --no-create-home appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

# Lightweight health check so orchestrators (Docker Compose, HF Spaces)
# can detect if the server is ready
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
