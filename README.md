# ClauseCheck

Upload a contract PDF, then chat with it. Answers are grounded in retrieved clauses and shown with their evidence.

    Browser UI ──► FastAPI ──► Postgres (documents, document_chunks + pgvector)
                      │  enqueue                    ▲
                      ▼                             │ chunks + embeddings
                    Redis (RQ) ──► worker: extract PDF → split clauses → Cohere embed → store
    /api/ask: rewrite follow-up (Groq) → embed → vector search + keyword search → RRF → MMR → grounded answer (Groq)

## Setup
1. `python -m venv .venv`, activate it, then `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in the keys and DB settings
3. `psql -U postgres -d <DB_NAME> -f schema.sql` (needs the pgvector extension)
4. Start Redis (local install, WSL, or `docker run -p 6379:6379 redis`)

## Run (3 terminals, from the project root)
    uvicorn app.main:app --reload
    rq worker --worker-class rq.worker.SimpleWorker ingestion
    # open http://localhost:8000 and upload a PDF

## API
`POST /api/documents` (PDF upload, 202) · `GET /api/documents` · `GET /api/documents/{id}` (status) · `POST /api/ask` · `DELETE /api/documents/{id}` · `GET /health`

## Deploy (free tier: Neon + Render)
Web + worker run in one Docker container (`start.sh`); Postgres/pgvector on Neon; the job queue on Render Key Value.

1. **Neon** - create a free project (region Singapore), copy the **pooled** connection string (`postgresql://...?sslmode=require`).
2. **Push to GitHub** - `.env` must NOT be committed (it is in `.gitignore`).
3. **Render** - New > Blueprint > pick the repo. It reads `render.yaml` and creates the web service and the Key Value instance. When asked, fill in `DATABASE_URL`, `COHERE_API_KEY`, `GROQ_API_KEY` and `DEMO_PASSWORD`.
4. Open the service URL, log in as `demo` with your password, upload `samples/messy_contract.pdf` once (it is stored in Neon and survives restarts).

## Live demo

https://clausecheck-r6kv.onrender.com

Login:

username `demo` password:`smasif`

Note: this runs on free tiers, so the first load can take about a minute
while the server wakes up.


`schema.sql` is applied automatically at startup. `/healthz` is the health check (it does not touch the database, so Neon can sleep).
Demo protection (env vars, off locally): `DEMO_PASSWORD` (login), `MAX_DOCUMENTS`, `ASK_LIMIT` (questions per IP per 10 min).
Free-tier behaviour: the service sleeps after 15 min idle and takes about a minute to wake; Neon sleeps after 5 min idle; the queue is in memory only.
