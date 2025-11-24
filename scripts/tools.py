"""
Retrieval tools for the Legal RAG Agent.

This module provides the search_legal_docs function that retrieves full text
of legal documents based on a query. The model and retriever are passed as
parameters to avoid reinstantiation.
"""

import json
from functools import lru_cache

from etils import epath
from pylate import indexes, models, retrieve


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
    # Encode the query
    query_embedding = encoder.encode(
        query,
        is_query=True,
        show_progress_bar=False,
    )

    # Retrieve top-k documents
    results = retriever.retrieve(
        queries_embeddings=query_embedding,
        k=k,
    )

    # Get the results for the first (and only) query
    search_results = results[0] if results else []

    # Load the document mapping
    doc_mapping = load_doc_mapping(str(index_folder))

    # Enrich results with full text
    enriched_results = []
    for result in search_results:
        doc_id = result["id"]
        enriched_results.append(
            {
                "id": doc_id,
                "score": result["score"],
                "text": doc_mapping.get(doc_id, "[Document not found]"),
            }
        )

    return enriched_results
