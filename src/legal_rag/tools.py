"""Minimal search and lookup tools that operate on provided dependencies."""

from __future__ import annotations

import random

from pylate import models, retrieve

from legal_rag.chunking import DocumentChunkCache

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


def lookup_chunk(
    chunk_id: str,
    *,
    chunk_cache: DocumentChunkCache,
    include_context: bool = True,
    context_chunks: int = 1,
) -> str:
    """Retrieve a specific chunk with optional surrounding context.

    Args:
        chunk_id: The chunk ID in format "docid-chunkidx".
        chunk_cache: DocumentChunkCache for retrieving chunk text.
        include_context: Whether to include surrounding chunks.
        context_chunks: Number of chunks before/after to include.

    Returns:
        The chunk text, optionally with surrounding context.
    """
    doc_id, chunk_idx = parse_chunk_id(chunk_id)
    chunks = chunk_cache.get_chunks(doc_id)

    if chunks is None:
        return f"[Document '{doc_id}' not found]"

    if chunk_idx >= len(chunks):
        return f"[Chunk {chunk_idx} not found in document '{doc_id}']"

    metadata = chunk_cache.get_document_metadata(doc_id)
    header = (
        f"Document: {doc_id} | Titre: {metadata['title']} | "
        f"Date: {metadata['decision_date']}\n"
        f"{'=' * 60}\n"
    )

    if include_context:
        start_idx = max(0, chunk_idx - context_chunks)
        end_idx = min(len(chunks), chunk_idx + context_chunks + 1)

        context_texts = []
        for i in range(start_idx, end_idx):
            marker = ">>> " if i == chunk_idx else "    "
            context_texts.append(f"{marker}[Chunk {i}] {chunks[i].text}")

        return header + "\n".join(context_texts)

    return header + f"[Chunk {chunk_idx}] {chunks[chunk_idx].text}"


def lookup_legal_doc(
    chunk_id: str,
    *,
    dataset=None,
    chunk_cache: DocumentChunkCache | None = None,
    doc_id_column: str = DEFAULT_DOC_ID_COLUMN,
    score: str | float | None = None,
) -> str:
    """Resolve and render full document text from a chunk_id.

    Args:
        chunk_id: The chunk ID returned by PLAID search (format: "docid-chunkidx").
        dataset: The HuggingFace dataset containing the documents (legacy).
        chunk_cache: DocumentChunkCache for retrieving document (preferred).
        doc_id_column: Name of the column containing document IDs.
        score: Optional relevance score to display.

    Returns:
        Formatted document text with metadata.
    """
    doc_id, _ = parse_chunk_id(chunk_id)

    # Prefer chunk_cache if available
    if chunk_cache is not None:
        row = chunk_cache._get_row(doc_id)
        if row is None:
            return f"[Document with id='{doc_id}' not found]"
    elif dataset is not None:
        try:
            row = get_row_by_id(dataset, doc_id, doc_id_column)
        except KeyError:
            return f"[Document with {doc_id_column}='{doc_id}' not found]"
    else:
        return "[Error: No dataset or chunk_cache provided]"

    base = TEMPLATE_DOCUMENT.format(
        title=row.get("title", ""),
        decision_date=row.get("decision_date", ""),
        juridiction=row.get("juridiction", ""),
        formation=row.get("formation", ""),
        applied_law=row.get("applied_law", ""),
        content=row.get("content", ""),
        solution=row.get("solution", ""),
    )
    score_str = f" (score: {score})" if score is not None else ""
    return f"Document id: {doc_id}{score_str}\n{base}"


def search_legal_docs(
    query: str,
    encoder: models.ColBERT,
    retriever: retrieve.ColBERT,
    k: int = 100,
    chunk_cache: DocumentChunkCache | None = None,
    max_preview_chars: int = 300,
) -> str:
    """Search for legal documents and return chunk IDs with scores and previews.

    Args:
        query: The search query string.
        encoder: ColBERT encoder for query embedding.
        retriever: ColBERT retriever for searching the index.
        k: Number of chunks to return (default 100 for reranking).
        chunk_cache: Optional DocumentChunkCache for generating chunk previews.
            When provided, each result includes the actual chunk text.
        max_preview_chars: Maximum characters for chunk previews.

    Returns:
        Formatted search results with chunk IDs, scores, and content previews.
        The agent should use these previews to select which chunks to analyze.
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
        return "No results found for your query."

    # Shuffle results to treat them as an unordered set (no PLAID score bias)
    random.shuffle(search_results)

    formatted = []
    for res in search_results:
        chunk_id = res["id"]
        doc_id, _ = parse_chunk_id(chunk_id)

        if chunk_cache is not None:
            chunk = chunk_cache.get_chunk(chunk_id)
            metadata = chunk_cache.get_document_metadata(doc_id)

            if chunk and metadata:
                chunk_text = chunk.text[:max_preview_chars]
                if len(chunk.text) > max_preview_chars:
                    chunk_text += "..."
                formatted.append(
                    f"chunk_id={chunk_id}\n"
                    f"    Titre: {metadata['title'] or 'N/A'} | "
                    f"Date: {metadata['decision_date'] or 'N/A'}\n"
                    f"    Extrait: {chunk_text}"
                )
            else:
                formatted.append(f"chunk_id={chunk_id} [chunk not found]")
        else:
            formatted.append(f"chunk_id={chunk_id}")

    return "\n\n".join(formatted)
