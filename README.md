# ClauseCheck

Upload a contract PDF, then chat with it. Answers are grounded in the retrieved clauses and shown together with their evidence.

**Live demo:** https://clausecheck-r6kv.onrender.com
Login: user `demo`, password shared in my application / on request.
The demo runs on free tiers, so the first load can take about a minute while the server wakes up.

<!-- Add a screenshot or GIF here once you have one:  ![ClauseCheck](docs/screenshot.png) -->
![ClauseCheck answering a question, with the evidence panel on the right] (image.png)

## Features
- **Async ingestion.** Uploading a PDF returns `202` immediately. A background worker extracts the text, splits it into clauses, embeds them and stores them; the UI polls a status endpoint.
- **Hybrid retrieval.** pgvector cosine search (HNSW index) plus Postgres full-text search, fused with Reciprocal Rank Fusion and diversified with MMR.
- **Conversational.** Follow-up questions are rewritten into standalone queries using the last 3 turns before retrieval.
- **Grounded answers.** The model may only use the retrieved clauses, and says so when the contract does not contain the answer.
- **Evidence panel.** Every answer shows the clauses used, with page numbers and whether they matched semantically, by keyword, or both.
- **Demo protection.** Optional login, upload cap and per-IP question limit for a public deployment.

## Tech stack
Python, FastAPI, PostgreSQL + pgvector, Redis + RQ, Cohere embeddings (embed-v4.0, 1024 dimensions), Groq-hosted LLM, pdfplumber, vanilla JS front end, Docker, Neon + Render.

## How it works

    Browser UI ──► FastAPI ──► Postgres (documents, document_chunks + pgvector)
                      │  enqueue                    ▲
                      ▼                             │ chunks + embeddings
                    Redis (RQ) ──► worker: extract PDF → split clauses → Cohere embed → store

    /api/ask: rewrite follow-up (Groq) → embed → vector search + keyword search → RRF → MMR → grounded answer (Groq)

## Run locally
1. `python -m venv .venv`, activate it, then `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in the keys and database settings
3. Make sure PostgreSQL has the pgvector extension. `schema.sql` is applied automatically when the app starts (or run it yourself: `psql -U postgres -d <DB_NAME> -f schema.sql`)
4. Start Redis (local install, WSL, or `docker run -p 6379:6379 redis`)
5. In two terminals, from the project root:

        uvicorn app.main:app --reload
        rq worker --worker-class rq.worker.SimpleWorker ingestion

6. Open http://localhost:8000 and upload a PDF (`samples/messy_contract.pdf` works well for a first try).

## Configuration

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY`, `GROQ_MODEL` | LLM used for query rewriting and answers |
| `COHERE_API_KEY` | Embeddings |
| `DATABASE_URL` or `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT` | PostgreSQL connection |
| `REDIS_URL` or `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB` | Job queue |
| `DEMO_PASSWORD` | If set, the site asks for a login (user `demo`) |
| `MAX_DOCUMENTS` | Maximum number of uploaded documents (0 = unlimited) |
| `ASK_LIMIT`, `ASK_WINDOW_SECONDS` | Questions per IP per window (0 = off) |

## API
`POST /api/documents` (PDF upload, 202) · `GET /api/documents` · `GET /api/documents/{id}` (status) · `POST /api/ask` · `DELETE /api/documents/{id}` · `GET /health` (checks the database) · `GET /healthz` (no database access)

## Deploy (free tier: Neon + Render)
The web server and the worker run in one Docker container (`start.sh`). Postgres with pgvector is on Neon, and the job queue is on Render Key Value.

1. **Neon:** create a free project (region Singapore) and copy the **pooled** connection string (`postgresql://...?sslmode=require`).
2. **GitHub:** push the repo. `.env` must not be committed (it is in `.gitignore`).
3. **Render:** New > Blueprint > pick the repo. It reads `render.yaml` and creates the web service and the Key Value instance. When asked, fill in `DATABASE_URL`, `COHERE_API_KEY`, `GROQ_API_KEY` and `DEMO_PASSWORD`.
4. Open the service URL, log in as `demo`, and upload `samples/messy_contract.pdf` once. It is stored in Neon and survives restarts.

Free-tier behaviour: the web service sleeps after 15 minutes without traffic and takes about a minute to wake; Neon sleeps after 5 idle minutes; the queue is in memory only. `/healthz` does not touch the database, so Neon can sleep.

## Project structure

    app/            main.py (API), rag.py (retrieval + answers), worker.py (ingestion job),
                    ingestion.py (PDF → clauses), jobs.py (queue), db.py (connections)
    static/         single-page UI
    samples/        example contract
    schema.sql      tables and indexes (applied at startup)
    Dockerfile, start.sh, render.yaml

## Limitations and roadmap
- Text-based PDFs only (no OCR for scanned documents), and questions are answered one document at a time.
- Answers are not legal advice.
- Planned: a tool-calling agent on top of the retrieval functions, an evaluation harness with a question set, and a Docker Compose file for local development.
