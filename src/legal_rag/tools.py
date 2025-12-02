"""Minimal search and lookup tools that operate on provided dependencies."""

from __future__ import annotations

from typing import Mapping

from pylate import models, retrieve

from legal_rag.indexer import TEMPLATE_DOCUMENT


def lookup_legal_doc(
    doc_id: str,
    *,
    mapping_entries: Mapping[str, int],
    dataset,
    score: str | float | None = None,
) -> str:
    """
    Resolve and render full document text from a doc_id using mapping and datasets.

    The doc_id is assumed to be the PLAID id returned by search. mapping_entries
    is a flat map: {plaid_id: dataset_idx}. Dataset is provided up front.
    """
    doc_key = str(doc_id)
    dataset_idx = mapping_entries.get(doc_key)
    if dataset_idx is None:
        return "[Document mapping entry missing]"

    try:
        row = dataset[int(dataset_idx)]
    except Exception as exc:  # pragma: no cover
        return f"[Failed to load document content: {exc}]"

    base = TEMPLATE_DOCUMENT.format(
        title=row.get("title") or "",
        decision_date=row.get("decision_date") or "",
        juridiction=row.get("juridiction") or "",
        formation=row.get("formation") or "",
        applied_law=row.get("applied_law") or "",
        content=row.get("content") or "",
        solution=row.get("solution") or "",
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
