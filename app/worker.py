import logging

from psycopg2.extras import execute_values

from . import rag
from .db import get_db_connection
from .ingestion import ingest_pdf

logger = logging.getLogger("clausecheck.worker")
EMBED_BATCH = 96  # Cohere accepts at most 96 texts per call


def clean(text):
    """Join PDF line wraps ('Confidential / Information') but keep the clause heading on its own line."""
    head, _, body = text.partition("\n")
    return head.strip() + "\n" + " ".join(body.split()) if body.strip() else " ".join(text.split())


def embed_documents(texts):
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        response = rag.co.embed(
            model="embed-v4.0",
            texts=texts[i:i + EMBED_BATCH],
            input_type="search_document",
            output_dimension=1024,
            embedding_types=["float"],
        )
        vectors.extend(response.embeddings.float)
    return vectors


def process_document(document_id):
    print(f"Processing the document {document_id}...\n")
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("UPDATE documents SET status=%s WHERE id=%s", ("processing", document_id))
        conn.commit()

        cursor.execute("SELECT file_path FROM documents WHERE id=%s", (document_id,))
        file_path = cursor.fetchone()[0]

        result = ingest_pdf(file_path)                       # extract + split into clauses
        vectors = embed_documents([clean(c.raw_text) for c in result.clauses])

        rows = [
            (document_id, c.clause_index, c.page_number, clean(c.raw_text),
             "[" + ",".join(map(str, v)) + "]")
            for c, v in zip(result.clauses, vectors)
        ]
        cursor.execute("DELETE FROM document_chunks WHERE document_id=%s", (document_id,))  # re-run safe
        execute_values(
            cursor,
            "INSERT INTO document_chunks (document_id, chunk_index, page_number, chunk_text, embedding) VALUES %s",
            rows,
            template="(%s,%s,%s,%s,%s::vector)",
        )
        cursor.execute(
            "UPDATE documents SET status=%s, page_count=%s, chunk_count=%s, "
            "error_message=NULL, processed_at=now() WHERE id=%s",
            ("completed", result.page_count, len(rows), document_id),
        )
        conn.commit()
        print("processing completed..\n")
        return {"status": "completed", "document_id": document_id}

    except Exception as exc:
        logger.exception("Ingestion failed for %s", document_id)
        if conn:
            conn.rollback()
            c2 = conn.cursor()
            c2.execute("UPDATE documents SET status=%s, error_message=%s WHERE id=%s",
                       ("failed", str(exc)[:500], document_id))
            conn.commit()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
