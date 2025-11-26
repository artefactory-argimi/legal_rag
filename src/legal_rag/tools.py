"""
Retrieval tools for the Legal RAG Agent.

This module provides lightweight search (ids + previews), full lookup,
and combined search+full text helpers. The model and retriever are passed
as parameters to avoid reinstantiation.
"""

import json
from functools import lru_cache

from etils import epath
from pylate import models, retrieve


@lru_cache(maxsize=1)
def load_doc_mapping(index_folder: str) -> dict[str, str]:
    """
    Load the document ID to text mapping from disk.

    This function is cached to ensure the mapping is loaded only once.

    Args:
        index_folder: Path to the folder containing doc_mapping.json

    Returns:
        Dictionary mapping document IDs to their full text content
    """
    mapping_file = epath.Path(index_folder) / "doc_mapping.json"
    with mapping_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def lookup_legal_doc(doc_id: str, index_folder: epath.PathLike = "./index") -> str:
    """
    Fetch the full text for a document id from disk.
    """
    doc_mapping = load_doc_mapping(str(index_folder))
    return doc_mapping.get(doc_id, "[Document not found]")


def _preview(text: str, limit: int = 160) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _extract_metadata(text: str) -> dict[str, str]:
    """Best-effort extraction of metadata from the templated document text."""
    meta = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    # Common keys produced by indexer.TEMPLATE_DOCUMENT
    title = meta.get("Title", "")
    return {
        "title": title,
        "date": meta.get("Date", ""),
        "jurisdiction": meta.get("Jurisdiction", meta.get("Juridiction", "")),
        "formation": meta.get("Formation", ""),
        "solution": meta.get("Solution", ""),
        "decision_text": meta.get("Decision Text", ""),
    }


def search_legal_docs_metadata(
    query: str,
    encoder: models.ColBERT,
    retriever: retrieve.ColBERT,
    index_folder: epath.PathLike = "./index",
    k: int = 5,
    preview_chars: int = 160,
) -> list[dict[str, str | float]]:
    """
    Search for legal documents and return ids with score, title, metadata, and a short preview.
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
    doc_mapping = load_doc_mapping(str(index_folder))

    enriched_results = []
    for result in search_results:
        doc_id = result["id"]
        text = doc_mapping.get(doc_id, "[Document not found]")
        meta = _extract_metadata(text)
        enriched_results.append(
            {
                "id": doc_id,
                "score": result["score"],
                "title": meta.get("title", ""),
                "metadata": meta,
                "preview": _preview(text, limit=preview_chars),
            }
        )
    return enriched_results


def search_legal_docs(
    query: str,
    encoder: models.ColBERT,
    retriever: retrieve.ColBERT,
    index_folder: epath.PathLike = "./index",
    k: int = 5,
) -> list[dict[str, str | float]]:
    """
    Search for legal documents and return their full text.

    This function encodes the query using the provided model, retrieves the top-k
    most relevant documents using the retriever, and returns their full text content.

    Args:
        query: The legal question or search query
        model: The ColBERT model instance for encoding queries
        retriever: The PLAID index instance for retrieving documents
        index_folder: Path to the index folder containing doc_mapping.json
        k: Number of documents to retrieve (default: 5)

    Returns:
        List of dictionaries containing:
        - "id": document ID
        - "score": relevance score
        - "text": full text content of the document
    """
    # First get metadata to avoid duplicating logic.
    meta = search_legal_docs_metadata(
        query=query,
        encoder=encoder,
        retriever=retriever,
        index_folder=index_folder,
        k=k,
        preview_chars=10_000,  # large enough to avoid truncation for full text
    )
    # Now attach full text instead of previews.
    doc_mapping = load_doc_mapping(str(index_folder))
    results = []
    for item in meta:
        doc_id = item["id"]
        results.append(
            {
                "id": doc_id,
                "score": item["score"],
                "text": doc_mapping.get(doc_id, "[Document not found]"),
            }
        )
    return results
