import os

from redis import Redis
from rq import Queue

from . import db  # noqa: F401  (loads .env)

# REDIS_URL (Render Key Value, Upstash, ...) wins; otherwise REDIS_HOST/PORT/DB for local use.
_url = os.getenv("REDIS_URL")
redis_conn = Redis.from_url(_url) if _url else Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    db=int(os.getenv("REDIS_DB", "0")),
)
# The worker must listen on the same name: rq worker ingestion
ingestion_queue = Queue("ingestion", connection=redis_conn)
