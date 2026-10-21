FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/.models

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Download the embedding/re-ranking models and build the index at image build time,
# so the container starts instantly.
RUN python -m policypilot ingest

EXPOSE 8000
CMD ["python", "-m", "policypilot", "serve", "--host", "0.0.0.0", "--port", "8000"]
