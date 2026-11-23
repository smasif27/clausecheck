"""
Phase 2: PDF ingestion service.

Given a PDF on disk, produce a page count and a list of clause candidates
(raw text + page number + order), ready for Phase 5 to persist as Clause
rows and Phase 3 to analyze.
"""

from dataclasses import dataclass, field
import re
from collections import Counter
import sys

import pdfplumber


class PDFIngestionError(Exception):
    """Raised when a PDF cannot be opened or contains no extractable text."""
    pass


@dataclass
class ExtractedClause:
    clause_index: int
    page_number: int
    raw_text: str


@dataclass
class IngestionResult:
    page_count: int
    clauses: list[ExtractedClause] = field(default_factory=list)


# Matches common legal clause markers at the start of a line:
#   "1. "        "1.1 "        "1.1.2) "     "(a) "     "Section 3: "
_CLAUSE_MARKER = re.compile(
    r"^\s*(?:"
    r"\d+(?:\.\d+)*[\.\)]\s+"          # 1.  1.1)  2.3.4.
    r"|\([a-zA-Z]\)\s+"                # (a)  (iv)
    r"|[ivxIVX]+[\.\)]\s+"             # roman numerals: iv.  IV)
    r"|Section\s+\d+[:.]?\s*"          # Section 3:
    r"|Article\s+\d+[:.]?\s*"          # Article 2
    r")",
)

_MIN_CLAUSE_CHARS = 20  # drop near-empty fragments (stray headers, page numbers)


def extract_pages(file_path: str) -> list[str]:
    """
    Extract text from every page of the PDF.

    Returns:
        A list containing the text of each page.
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            pages = [
                page.extract_text() or ""
                for page in pdf.pages
            ]
    except Exception as exc:
        raise PDFIngestionError(
            f"Could not open PDF: {exc}"
        ) from exc

    if not any(page.strip() for page in pages):
        raise PDFIngestionError(
            "No extractable text found. "
            "This may be a scanned/image-only PDF. "
            "OCR is not implemented in v1."
        )

    return pages


def _strip_repeated_lines(pages: list[str]) -> list[str]:
    """Remove running headers/footers (e.g. 'Page 2 of 6') that repeat on most pages."""
    if len(pages) < 3:
        return pages
    key = lambda line: re.sub(r"\d+", "#", line.strip().lower())
    counts = Counter(k for p in pages for k in {key(l) for l in p.split("\n") if l.strip()})
    limit = 0.6 * len(pages)
    return ["\n".join(l for l in p.split("\n") if not l.strip() or counts[key(l)] < limit) for p in pages]


def split_into_clauses(pages: list[str]) -> list[ExtractedClause]:
    """
    Walks each page line by line. A line matching _CLAUSE_MARKER starts a
    new clause; everything until the next marker (or end of page) belongs
    to it. Text before the first marker (title, preamble) is kept as clause 0.
    If a whole document has zero usable clauses, falls back to splitting
    on blank lines (paragraph breaks) so we still get *something* usable
    instead of an empty result.
    """
    pages = _strip_repeated_lines(pages)
    clauses: list[ExtractedClause] = []
    clause_index = 0
    current_lines: list[str] = []
    current_page = 1
    seen_first_marker = False

    def flush():
        nonlocal clause_index
        text = "\n".join(current_lines).strip()
        if len(text) >= _MIN_CLAUSE_CHARS:
            clauses.append(ExtractedClause(clause_index, current_page, text))
            clause_index += 1

    for page_num, page_text in enumerate(pages, start=1):
        for line in page_text.split("\n"):
            if _CLAUSE_MARKER.match(line):
                flush()  # also keeps the preamble (parties, purpose) as its own clause
                seen_first_marker = True
                current_lines = [line]
                current_page = page_num
            else:
                if not current_lines:
                    current_page = page_num
                current_lines.append(line)
    flush()

    if clauses:
        return clauses

    # Fallback: no usable numbered clauses — split on blank lines instead.
    clause_index = 0
    for page_number, page_text in enumerate(pages, start=1):
        for paragraph in re.split(r"\n\s*\n", page_text):
            paragraph = paragraph.strip()
            if len(paragraph) < _MIN_CLAUSE_CHARS:
                continue
            clauses.append(ExtractedClause(clause_index, page_number, paragraph))
            clause_index += 1

    return clauses


def ingest_pdf(file_path: str) -> IngestionResult:
    """
    Main entry point for the PDF ingestion service.

    PDF
      ↓
    Extract pages
      ↓
    Split into clauses
      ↓
    Return structured result
    """
    pages = extract_pages(file_path)
    clauses = split_into_clauses(pages)

    return IngestionResult(
        page_count=len(pages),
        clauses=clauses
    )


# ============================================================
# Manual testing
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "Usage: "
            "python -m app.services.pdf_ingestion "
            "<path_to_pdf>"
        )
        sys.exit(1)

    pdf_path = sys.argv[1]

    try:
        result = ingest_pdf(pdf_path)

        print(f"Pages: {result.page_count}")
        print(f"Clauses found: {len(result.clauses)}")
        print()

        for clause in result.clauses:
            preview = (
                clause.raw_text
                .replace("\n", " ")
                [:100]
            )

            print(
                f"[{clause.clause_index}] "
                f"Page {clause.page_number}: "
                f"{preview}..."
            )

    except PDFIngestionError as exc:
        print(f"PDF ingestion failed: {exc}")
        sys.exit(1)