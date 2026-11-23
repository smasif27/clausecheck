import os
import re
import cohere
import psycopg2
from groq import Groq
from dotenv import load_dotenv

# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

cohere_api_key = os.getenv("COHERE_API_KEY")

db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST", "localhost")
db_port = os.getenv("DB_PORT", "5432")

groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL")

if not cohere_api_key:
    raise ValueError(
        "COHERE_API_KEY is missing in .env"
    )

if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY is missing in .env"
    )

if not groq_model:
    raise ValueError(
        "GROQ_MODEL is missing in .env"
    )

# ============================================================
# 2. INITIALIZE CLIENTS
# ============================================================

co = cohere.ClientV2(
    api_key=cohere_api_key
)

groq_client = Groq(
    api_key=groq_api_key
)

# ============================================================
# 3. DATABASE CONNECTION
# ============================================================

from .db import get_db_connection  # noqa: E402  (supports DATABASE_URL and DB_* settings)


# ============================================================
# 4. COSINE SIMILARITY
# ============================================================


def cosine_similarity(
    vector_a,
    vector_b
):

    dot_product = sum(
        a * b
        for a, b in zip(
            vector_a,
            vector_b
        )
    )

    magnitude_a = sum(
        a * a
        for a in vector_a
    ) ** 0.5

    magnitude_b = sum(
        b * b
        for b in vector_b
    ) ** 0.5

    if (
        magnitude_a == 0
        or magnitude_b == 0
    ):
        return 0.0

    return (
        dot_product
        / (magnitude_a * magnitude_b)
    )


# ============================================================
# 5. HISTORY FORMATTING
# ============================================================


def format_history(
    conversation_history,
    max_turns=3
):
    """
    Keep only the most recent conversation turns.

    Each turn contains:
        question
        answer
    """

    recent_history = conversation_history[
        -max_turns:
    ]

    if not recent_history:

        return "No previous conversation."

    parts = []

    for turn in recent_history:

        parts.append(
            f"""
User:
{turn['question']}

Assistant:
{turn['answer']}
"""
        )

    return "\n".join(parts)


# ============================================================
# 6. REWRITE FOLLOW-UP QUESTION
# ============================================================

def rewrite_question(
    question,
    conversation_history
):
    """
    Rewrite a follow-up question into a standalone
    question before retrieval.

    This solves the problem where retrieval only sees
    the raw follow-up text.
    """

    if not conversation_history:
        return question

    history_text = format_history(
        conversation_history,
        max_turns=3
    )

    prompt = f"""
You are a query rewriting component for a legal
document retrieval system.

Rewrite the user's latest question into a standalone
question that can be understood without the previous
conversation.

Use the conversation history to resolve references
such as:

- it
- they
- them
- this
- that
- the other party
- the breach
- the clause
- after that

Preserve the user's original meaning.

Do not answer the question.

Do not add information that is not supported by
the conversation.

If the user's question is already standalone,
return it unchanged.

CONVERSATION HISTORY:

{history_text}

LATEST USER QUESTION:

{question}

RETURN ONLY THE STANDALONE QUESTION.
"""

    response = (
        groq_client
        .chat.completions.create(
            model=groq_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rewrite legal retrieval "
                        "questions. Return only the "
                        "standalone question."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )
    )

    rewritten = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    return rewritten


# ============================================================
# 7. CREATE QUERY EMBEDDING
# ============================================================

def create_query_embedding(
    query
):

    response = co.embed(
        model="embed-v4.0",
        texts=[query],
        input_type="search_query",
        output_dimension=1024,
        embedding_types=["float"]
    )

    query_vector = (
        response.embeddings.float[0]
    )

    print(
        f"Query embedding dimension: "
        f"{len(query_vector)}"
    )

    return query_vector


# ============================================================
# 8. VECTOR SEARCH
# ============================================================

def vector_search(
    query_vector,
    document_id,
    top_k=20
):

    vector_string = (
        "["
        + ",".join(
            map(str, query_vector)
        )
        + "]"
    )

    conn = get_db_connection()
    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            SELECT
                id,
                document_id,
                chunk_index,
                chunk_text,
                embedding,
                embedding <=> %s::vector
                    AS distance
            FROM document_chunks
            WHERE document_id = %s
            ORDER BY distance
            LIMIT %s
            """,
            (
                vector_string,
                document_id,
                top_k
            )
        )

        rows = cursor.fetchall()

        results = []

        for rank, row in enumerate(
            rows,
            start=1
        ):

            embedding = row[4]

            if isinstance(
                embedding,
                str
            ):

                embedding = (
                    embedding
                    .strip("[]")
                )

                embedding = [
                    float(value)
                    for value in
                    embedding.split(",")
                ]

            results.append(
                {
                    "id": row[0],
                    "document_id": row[1],
                    "chunk_index": row[2],
                    "chunk_text": row[3],
                    "embedding": embedding,
                    "distance": float(row[5]),
                    "rank": rank
                }
            )

        return results

    finally:

        cursor.close()
        conn.close()


# ============================================================
# 9. KEYWORD SEARCH
# ============================================================
# FIX APPLIED: now selects and parses the embedding column
# too, so keyword-only matches carry a real embedding into
# rrf_fusion() and are not dropped later by mmr_rerank().
# ============================================================

def keyword_search(
    query_text,
    document_id,
    top_k=20
):

    or_query = ' | '.join(re.findall(r'\w+', query_text))
    if not or_query:
        return []

    conn = get_db_connection()
    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            SELECT
                id,
                document_id,
                chunk_index,
                chunk_text,
                embedding,
                ts_rank(
                    search_vector,
                    to_tsquery(
                        'english',
                        %s
                    )
                ) AS rank
            FROM document_chunks
            WHERE
                document_id = %s
                AND search_vector @@
                    to_tsquery(
                        'english',
                        %s
                    )
            ORDER BY rank DESC
            LIMIT %s
            """,
            (
                or_query,
                document_id,
                or_query,
                top_k
            )
        )

        rows = cursor.fetchall()

        results = []

        for rank, row in enumerate(
            rows,
            start=1
        ):

            embedding = row[4]

            if isinstance(
                embedding,
                str
            ):

                embedding = (
                    embedding
                    .strip("[]")
                )

                embedding = [
                    float(value)
                    for value in
                    embedding.split(",")
                ]

            results.append(
                {
                    "id": row[0],
                    "document_id": row[1],
                    "chunk_index": row[2],
                    "chunk_text": row[3],
                    "embedding": embedding,
                    "rank": rank,
                    "keyword_score": float(
                        row[5]
                    )
                }
            )

        return results

    finally:

        cursor.close()
        conn.close()


# ============================================================
# 10. RRF FUSION
# ============================================================
# FIX APPLIED: keyword-only fused entries now store the real
# embedding (result["embedding"]) instead of None.
# ============================================================

def rrf_fusion(
    vector_results,
    keyword_results,
    k=60
):

    fused = {}

    # --------------------------------------------------------
    # VECTOR RESULTS
    # --------------------------------------------------------

    for result in vector_results:

        chunk_id = result["id"]

        if chunk_id not in fused:

            fused[chunk_id] = {
                "id": chunk_id,
                "document_id": result[
                    "document_id"
                ],
                "chunk_index": result[
                    "chunk_index"
                ],
                "chunk_text": result[
                    "chunk_text"
                ],
                "embedding": result[
                    "embedding"
                ],
                "vector_rank": None,
                "keyword_rank": None,
                "vector_score": None,
                "keyword_score": None,
                "rrf_score": 0.0
            }

        fused[chunk_id][
            "vector_rank"
        ] = result["rank"]

        fused[chunk_id][
            "vector_score"
        ] = result["distance"]

        fused[chunk_id][
            "rrf_score"
        ] += (
            1 / (
                k + result["rank"]
            )
        )

    # --------------------------------------------------------
    # KEYWORD RESULTS
    # --------------------------------------------------------

    for result in keyword_results:

        chunk_id = result["id"]

        if chunk_id not in fused:

            fused[chunk_id] = {
                "id": chunk_id,
                "document_id": result[
                    "document_id"
                ],
                "chunk_index": result[
                    "chunk_index"
                ],
                "chunk_text": result[
                    "chunk_text"
                ],
                "embedding": result[
                    "embedding"
                ],
                "vector_rank": None,
                "keyword_rank": None,
                "vector_score": None,
                "keyword_score": None,
                "rrf_score": 0.0
            }

        fused[chunk_id][
            "keyword_rank"
        ] = result["rank"]

        fused[chunk_id][
            "keyword_score"
        ] = result[
            "keyword_score"
        ]

        fused[chunk_id][
            "rrf_score"
        ] += (
            1 / (
                k + result["rank"]
            )
        )

    return sorted(
        fused.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )


# ============================================================
# 11. MMR
# ============================================================

def mmr_rerank(
    query_vector,
    candidates,
    k=5,
    lambda_param=0.7
):

    remaining = [
        candidate.copy()
        for candidate in candidates
        if candidate.get("embedding")
        is not None
    ]

    selected = []

    while (
        remaining
        and len(selected) < k
    ):

        best_candidate = None
        best_score = float("-inf")

        for candidate in remaining:

            relevance = cosine_similarity(
                query_vector,
                candidate["embedding"]
            )

            if not selected:

                redundancy = 0.0

            else:

                similarities = [
                    cosine_similarity(
                        candidate["embedding"],
                        selected_candidate[
                            "embedding"
                        ]
                    )
                    for selected_candidate
                    in selected
                ]

                redundancy = max(
                    similarities
                )

            mmr_score = (
                lambda_param * relevance
                -
                (1 - lambda_param)
                * redundancy
            )

            if mmr_score > best_score:

                best_score = mmr_score

                best_candidate = (
                    candidate.copy()
                )

                best_candidate[
                    "mmr_score"
                ] = mmr_score

        selected.append(
            best_candidate
        )

        remaining = [
            candidate
            for candidate in remaining
            if candidate["id"]
            != best_candidate["id"]
        ]

    return selected


# ============================================================
# 12. BUILD ANSWER PROMPT
# ============================================================

def build_prompt(
    question,
    conversation_history,
    retrieved_chunks
):

    context_parts = []

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        context_parts.append(
            f"""
--- Context {index} ---

{chunk['chunk_text']}
"""
        )

    context = "\n".join(
        context_parts
    )

    history_text = format_history(
        conversation_history,
        max_turns=3
    )

    prompt = f"""
You are a precise legal document analysis assistant.

Answer the user's question using ONLY the provided
contract context.

You may use the conversation history to understand
references in the user's question, but all factual
contract information must come from the retrieved
contract context.

Do not use outside knowledge.

Do not invent information, and do not add details (such as who paid, who performed an act, or
amounts) that the context does not state.

Preserve the parties, conditions, time periods,
rights, and obligations exactly as stated in the
contract.

If the answer cannot be determined from the provided
contract context, say:

"I cannot determine the answer from the provided
contract context."

Keep the answer concise and factual.

CONVERSATION HISTORY:

{history_text}

RETRIEVED CONTRACT CONTEXT:

{context}

CURRENT USER QUESTION:

{question}
"""

    return prompt


# ============================================================
# 13. GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    conversation_history,
    retrieved_chunks
):

    prompt = build_prompt(
        question=question,
        conversation_history=conversation_history,
        retrieved_chunks=retrieved_chunks
    )

    response = (
        groq_client
        .chat.completions.create(
            model=groq_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise legal "
                        "document analysis assistant. "
                        "Use only the provided "
                        "contract context for "
                        "contract facts."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )
    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# ============================================================
# 14. COMPLETE CONVERSATIONAL RAG
# ============================================================

def rag_pipeline(
    question,
    document_id,
    conversation_history,
    retrieval_k=20,
    final_k=5,
    lambda_param=0.7,
    rrf_k=60
):

    print("\n")
    print("=" * 70)
    print("CURRENT QUESTION")
    print("=" * 70)

    print(question)

    # --------------------------------------------------------
    # STEP 1: REWRITE QUESTION
    # --------------------------------------------------------

    standalone_question = rewrite_question(
        question=question,
        conversation_history=conversation_history
    )

    print("\n")
    print("=" * 70)
    print("STANDALONE RETRIEVAL QUESTION")
    print("=" * 70)

    print(standalone_question)

    # --------------------------------------------------------
    # STEP 2: CREATE EMBEDDING
    # --------------------------------------------------------

    query_vector = (
        create_query_embedding(
            standalone_question
        )
    )

    # --------------------------------------------------------
    # STEP 3: VECTOR SEARCH
    # --------------------------------------------------------

    vector_results = vector_search(
        query_vector=query_vector,
        document_id=document_id,
        top_k=retrieval_k
    )

    # --------------------------------------------------------
    # STEP 4: KEYWORD SEARCH
    # --------------------------------------------------------

    keyword_results = keyword_search(
        query_text=standalone_question,
        document_id=document_id,
        top_k=retrieval_k
    )

    # --------------------------------------------------------
    # STEP 5: RRF
    # --------------------------------------------------------

    fused_results = rrf_fusion(
        vector_results=vector_results,
        keyword_results=keyword_results,
        k=rrf_k
    )

    # --------------------------------------------------------
    # STEP 6: MMR
    # --------------------------------------------------------

    final_chunks = mmr_rerank(
        query_vector=query_vector,
        candidates=fused_results,
        k=final_k,
        lambda_param=lambda_param
    )

    # --------------------------------------------------------
    # STEP 7: GENERATE ANSWER
    # --------------------------------------------------------

    answer = generate_answer(
        question=question,
        conversation_history=conversation_history,
        retrieved_chunks=final_chunks
    )

    print("\n")
    print("=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(answer)

    return answer


# ============================================================
# 15. MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("CLAUSECHECK - CONVERSATIONAL HYBRID RAG")
    print("=" * 70)

    print(
        "\nEnter 'exit' or 'quit' to stop."
    )

    document_id = input(
        "\nDocument ID "
        "(default: sample_contract_001): "
    ).strip()

    if not document_id:

        document_id = (
            "sample_contract_001"
        )

    lambda_input = input(
        "MMR lambda "
        "(default: 0.70): "
    ).strip()

    if lambda_input:

        lambda_param = float(
            lambda_input
        )

    else:

        lambda_param = 0.70

    if not 0 <= lambda_param <= 1:

        raise ValueError(
            "MMR lambda must be between 0 and 1."
        )

    # --------------------------------------------------------
    # Conversation memory
    # --------------------------------------------------------

    conversation_history = []

    print(
        f"\nUsing document: "
        f"{document_id}"
    )

    print(
        f"Using MMR lambda: "
        f"{lambda_param}"
    )

    # --------------------------------------------------------
    # Conversation loop
    # --------------------------------------------------------

    while True:

        question = input(
            "\nAsk a question : "
        ).strip()

        if question.lower() in [
            "exit",
            "quit"
        ]:

            print("\nGoodbye!")

            break

        if not question:

            print(
                "Please enter a question."
            )

            continue

        try:

            answer = rag_pipeline(
                question=question,
                document_id=document_id,
                conversation_history=conversation_history,
                retrieval_k=20,
                final_k=5,
                lambda_param=lambda_param,
                rrf_k=60
            )

            # ------------------------------------------------
            # Save conversation turn
            # ------------------------------------------------

            conversation_history.append(
                {
                    "question": question,
                    "answer": answer
                }
            )

            # ------------------------------------------------
            # Keep memory bounded
            # ------------------------------------------------

            conversation_history = (
                conversation_history[-3:]
            )

        except Exception as e:

            print("\nERROR:")
            print(str(e))


# ============================================================
# 16. RUN
# ============================================================

if __name__ == "__main__":
    main()