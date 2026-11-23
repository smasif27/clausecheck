-- Run once:  psql -U postgres -d <DB_NAME> -f schema.sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id           text PRIMARY KEY,
    filename     text NOT NULL,
    file_path    text NOT NULL,
    status       text NOT NULL DEFAULT 'pending',
    uploaded_at  timestamp DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS error_message text,
    ADD COLUMN IF NOT EXISTS page_count    integer,
    ADD COLUMN IF NOT EXISTS chunk_count   integer,
    ADD COLUMN IF NOT EXISTS processed_at  timestamp;

-- If you already have an older document_chunks table, drop it first
-- (all rows can be re-created by uploading the PDFs again):
-- DROP TABLE IF EXISTS document_chunks;
CREATE TABLE IF NOT EXISTS document_chunks (
    id            bigserial PRIMARY KEY,
    document_id   text    NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   integer NOT NULL,
    page_number   integer,
    chunk_text    text    NOT NULL,
    embedding     vector(1024) NOT NULL,
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED,
    UNIQUE (document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_search    ON document_chunks USING gin (search_vector);
CREATE INDEX IF NOT EXISTS idx_chunks_document  ON document_chunks (document_id);
