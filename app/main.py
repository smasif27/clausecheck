import base64
import logging
import os
import secrets
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rq import Retry

from . import rag
from .db import get_db_connection
from .jobs import ingestion_queue

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("clausecheck")
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_BYTES = 20 * 1024 * 1024

app = FastAPI(title="ClauseCheck", description="Conversational hybrid-RAG contract analysis", version="1.0.0")


# ---- public-demo protection (all off by default, so local use is unchanged) ----
DEMO_USER = os.getenv("DEMO_USER", "demo")
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD")                    # set it to require a login
MAX_DOCUMENTS = int(os.getenv("MAX_DOCUMENTS", "0"))          # 0 = unlimited
ASK_LIMIT = int(os.getenv("ASK_LIMIT", "0"))                  # questions per IP per window, 0 = off
ASK_WINDOW = int(os.getenv("ASK_WINDOW_SECONDS", "600"))
_asks = defaultdict(deque)


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if DEMO_PASSWORD and request.url.path != "/healthz":
        header, ok = request.headers.get("authorization", ""), False
        if header.startswith("Basic "):
            try:
                user, _, pw = base64.b64decode(header[6:]).decode().partition(":")
                ok = secrets.compare_digest(user, DEMO_USER) and secrets.compare_digest(pw, DEMO_PASSWORD)
            except Exception:
                ok = False
        if not ok:
            return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="ClauseCheck demo"'})
    return await call_next(request)


def check_rate_limit(request: Request):
    if not ASK_LIMIT:
        return
    ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (request.client.host if request.client else "?")
    now, q = time.time(), _asks[ip]
    while q and now - q[0] > ASK_WINDOW:
        q.popleft()
    if len(q) >= ASK_LIMIT:
        raise HTTPException(429, "Too many questions. Please wait a few minutes and try again.")
    q.append(now)


def run_sql(sql, params=(), fetch=False):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall() if fetch else None
        conn.commit()
        return rows
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


class Turn(BaseModel):
    question: str
    answer: str


class AskRequest(BaseModel):
    question: str
    document_id: str
    history: List[Turn] = []
    lambda_param: float = 0.7


@app.on_event("startup")
def init_schema():
    """schema.sql is idempotent, so a fresh database (e.g. Neon) is ready after the first start."""
    try:
        run_sql((BASE_DIR / "schema.sql").read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Could not apply schema.sql - run it manually with psql")


@app.get("/healthz")
def healthz():
    """For the hosting platform's health checks. Deliberately does not touch the database,
    so a serverless Postgres can scale to zero."""
    return {"status": "ok"}


@app.get("/health")
def health():
    try:
        run_sql("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception:
        logger.exception("health check failed")
        return {"status": "unhealthy", "database": "connection failed"}


@app.post("/api/documents", status_code=202)
async def upload_document(file: UploadFile = File(...)):
    if MAX_DOCUMENTS and run_sql("SELECT COUNT(*) FROM documents", fetch=True)[0][0] >= MAX_DOCUMENTS:
        raise HTTPException(429, "Demo limit reached: the owner has to delete a document first.")
    document_id = str(uuid.uuid4())
    filename = Path(file.filename or "upload.pdf").name
    path = UPLOAD_DIR / f"{document_id}.pdf"
    size = 0
    try:
        with open(path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                if size == 0 and not chunk.startswith(b"%PDF"):
                    raise HTTPException(400, "Only PDF files are allowed.")
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, "File is larger than 20 MB.")
                out.write(chunk)
        if size == 0:
            raise HTTPException(400, "The file is empty.")
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    except Exception:
        logger.exception("saving upload failed")
        path.unlink(missing_ok=True)
        raise HTTPException(500, "Failed to save the file.")

    try:
        run_sql(
            "INSERT INTO documents (id, filename, file_path, status) VALUES (%s,%s,%s,'pending')",
            (document_id, filename, str(path)),
        )
    except Exception:
        logger.exception("creating document record failed")
        path.unlink(missing_ok=True)
        raise HTTPException(500, "Failed to create the document record.")

    try:
        ingestion_queue.enqueue("app.worker.process_document", document_id,
                                retry=Retry(max=2), job_timeout=600)
    except Exception:
        logger.exception("enqueue failed")
        run_sql("UPDATE documents SET status='failed', error_message=%s WHERE id=%s",
                ("Could not enqueue the processing job", document_id))
        raise HTTPException(503, "Processing queue is unavailable.")

    return {"document_id": document_id, "filename": filename, "status": "pending"}


@app.get("/api/documents")
def list_documents():
    rows = run_sql(
        """SELECT d.id, d.filename, d.status, COUNT(c.id)
           FROM documents d LEFT JOIN document_chunks c ON c.document_id = d.id
           GROUP BY d.id ORDER BY d.uploaded_at DESC""",
        fetch=True,
    )
    return [{"document_id": r[0], "filename": r[1], "status": r[2], "chunks": r[3]} for r in rows]


@app.get("/api/documents/{document_id}")
def get_document(document_id: str):
    rows = run_sql(
        "SELECT id, filename, status, page_count, chunk_count, error_message FROM documents WHERE id=%s",
        (document_id,), fetch=True,
    )
    if not rows:
        raise HTTPException(404, "Document not found.")
    r = rows[0]
    return {"document_id": r[0], "filename": r[1], "status": r[2],
            "page_count": r[3], "chunk_count": r[4], "error_message": r[5]}


@app.delete("/api/documents/{document_id}", status_code=204)
def delete_document(document_id: str):
    rows = run_sql("DELETE FROM documents WHERE id=%s RETURNING file_path", (document_id,), fetch=True)
    if not rows:
        raise HTTPException(404, "Document not found.")
    Path(rows[0][0]).unlink(missing_ok=True)  # its chunks go too (ON DELETE CASCADE)


# Plain `def`: FastAPI runs it in a worker thread, so the blocking
# psycopg2 / Groq / Cohere calls don't freeze the server.
@app.post("/api/ask")
def ask(req: AskRequest, request: Request):
    check_rate_limit(request)
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Question is empty.")
    if not 0 <= req.lambda_param <= 1:
        raise HTTPException(400, "lambda_param must be between 0 and 1.")

    rows = run_sql("SELECT status FROM documents WHERE id=%s", (req.document_id,), fetch=True)
    if not rows:
        raise HTTPException(404, "Document not found.")
    if rows[0][0] != "completed":
        raise HTTPException(409, f"Document is not ready (status: {rows[0][0]}).")

    history = [t.model_dump() for t in req.history][-3:]
    try:
        standalone = rag.rewrite_question(question, history)
        query_vector = rag.create_query_embedding(standalone)
        vector_results = rag.vector_search(query_vector, req.document_id, 20)
        keyword_results = rag.keyword_search(standalone, req.document_id, 20)
        fused = rag.rrf_fusion(vector_results, keyword_results, 60)
        final_chunks = rag.mmr_rerank(query_vector, fused, k=5, lambda_param=req.lambda_param)
        answer = rag.generate_answer(question, history, final_chunks)
        pages = dict(run_sql(
            "SELECT chunk_index, page_number FROM document_chunks WHERE document_id=%s AND chunk_index = ANY(%s)",
            (req.document_id, [c["chunk_index"] for c in final_chunks]), fetch=True))
    except Exception:
        logger.exception("ask failed")
        raise HTTPException(500, "Could not answer the question. Please try again.")

    return {
        "answer": answer,
        "standalone_question": standalone,
        "sources": [
            {
                "chunk_index": c["chunk_index"],
                "page": pages.get(c["chunk_index"]),
                "text": c["chunk_text"],
                "vector_rank": c.get("vector_rank"),
                "keyword_rank": c.get("keyword_rank"),
                "rrf_score": round(c["rrf_score"], 4),
                "mmr_score": round(c["mmr_score"], 4),
            }
            for c in final_chunks
        ],
    }


app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")
