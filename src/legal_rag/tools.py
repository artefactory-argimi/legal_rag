"""Minimal search and lookup tools that operate on provided dependencies."""

from __future__ import annotations

from pylate import models, retrieve

DEFAULT_DOC_ID_COLUMN = "id"

TEMPLATE_DOCUMENT = """Titre : {title}
Date : {decision_date}
Juridiction : {juridiction}
Formation : {formation}
Solution : {solution}
Droit appliqué : {applied_law}
Texte de la décision : {content}
"""


def parse_chunk_id(chunk_id: str) -> tuple[str, int]:
    """Parse a chunk ID into (doc_id, chunk_index).

    Args:
        chunk_id: ID in format "docid-chunkidx" (e.g., "12345-2").

    Returns:
        Tuple of (parent_doc_id, chunk_index).
    """
    parts = chunk_id.rsplit("-", 1)
    return parts[0], int(parts[1])


def get_row_by_id(
    dataset, doc_id: str, doc_id_column: str = DEFAULT_DOC_ID_COLUMN
) -> dict:
    """Retrieve a row from a dataset by its document ID.

    Args:
        dataset: A HuggingFace dataset with a doc_id column.
        doc_id: The document ID to search for.
        doc_id_column: Name of the document ID column in the dataset.

    Returns:
        The matching row as a dictionary.

    Raises:
        KeyError: If no row with the given ID is found.
    """
    for row in dataset:
        if str(row[doc_id_column]) == str(doc_id):
            return dict(row)
    raise KeyError(f"Document with {doc_id_column}='{doc_id}' not found in dataset")


def lookup_legal_doc(
    chunk_id: str,
    *,
    dataset,
    doc_id_column: str = DEFAULT_DOC_ID_COLUMN,
    score: str | float | None = None,
) -> str:
    """Resolve and render full document text from a chunk_id.

    Args:
        chunk_id: The chunk ID returned by PLAID search (format: "docid-chunkidx").
        dataset: The HuggingFace dataset containing the documents.
        doc_id_column: Name of the column containing document IDs.
        score: Optional relevance score to display.

    Returns:
        Formatted document text with metadata.
    """
    doc_id, _ = parse_chunk_id(chunk_id)
    try:
        row = get_row_by_id(dataset, doc_id, doc_id_column)
    except KeyError:
        return f"[Document with {doc_id_column}='{doc_id}' not found]"

    base = TEMPLATE_DOCUMENT.format(
        title=row["title"] or "",
        decision_date=row["decision_date"] or "",
        juridiction=row["juridiction"] or "",
        formation=row["formation"] or "",
        applied_law=row["applied_law"] or "",
        content=row["content"] or "",
        solution=row["solution"] or "",
    )
    score_str = f" (score: {score})" if score is not None else ""
    return f"Document id: {doc_id}{score_str}\n{base}"


def search_legal_docs(
    query: str,
    encoder: models.ColBERT,
    retriever: retrieve.ColBERT,
    k: int = 5,
) -> str:
    """
    Search for legal documents and return ids with scores (formatted for tools).
    """
    query_embedding = encoder.encode(
        query,
        is_query=True,
        show_progress_bar=False,
    )
    results = retriever.retrieve(
        queries_embeddings=query_embedding,
        k=k,
    )
    search_results = results[0] if results else []

    if not search_results:
        return "No results."

    return "\n\n".join(
        f"[{idx}] id={res['id']} score={res['score']:.4f}"
        for idx, res in enumerate(search_results)
    )
