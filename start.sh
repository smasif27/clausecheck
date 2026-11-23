#!/bin/sh
# One container runs both the ingestion worker and the web server (fits a single free instance).
set -e
REDIS="${REDIS_URL:-redis://${REDIS_HOST:-localhost}:${REDIS_PORT:-6379}/${REDIS_DB:-0}}"
rq worker --worker-class rq.worker.SimpleWorker --url "$REDIS" ingestion &
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
