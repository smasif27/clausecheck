import os
import time
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)


def get_db_connection(retries=3, delay=2.0):
    """Connect with DATABASE_URL (Neon / Render) or the DB_* variables.
    Retries briefly, because a serverless database may still be waking up."""
    for attempt in range(1, retries + 1):
        try:
            url = os.getenv("DATABASE_URL")
            if url:
                return psycopg2.connect(url, connect_timeout=15)
            return psycopg2.connect(
                dbname=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                host=os.getenv("DB_HOST", "localhost"),
                port=os.getenv("DB_PORT", "5432"),
                connect_timeout=15,
            )
        except psycopg2.OperationalError:
            if attempt == retries:
                raise
            time.sleep(delay)
