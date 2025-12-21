"""Shared chunking logic for indexing and retrieval.

This module provides consistent document chunking used by both the indexer
(at index time) and the agent (at retrieval time for runtime chunk extraction).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from chonkie import SemanticChunker, SentenceChunker


@dataclass(frozen=True)
class ChunkConfig:
    """Configuration for document chunking.

    Attributes:
        chunk_size: Maximum tokens per chunk (ColBERT limit is 511).
        chunk_overlap: Overlap between chunks (SentenceChunker only).
        chunker_type: Type of chunker ("semantic" or "sentence").
        chunk_tokenizer: For SentenceChunker: tokenizer name.
            For SemanticChunker: embedding model.
    """

    chunk_size: int = 511
    chunk_overlap: int = 0
    chunker_type: str = "semantic"
    chunk_tokenizer: str = "minishlab/potion-base-32M"


FIRST_CHUNK_HEADER = """Titre: {title} | Date: {decision_date}
"""

DEFAULT_CHUNK_CONFIG = ChunkConfig()


class ChunkerProtocol(Protocol):
    """Protocol for chunker objects."""

    def chunk(self, text: str) -> list: ...


def build_chunker(config: ChunkConfig = DEFAULT_CHUNK_CONFIG) -> ChunkerProtocol:
    """Build a chunker based on configuration.

    Args:
        config: Chunking configuration.

    Returns:
        A configured chunker instance (SemanticChunker or SentenceChunker).
    """
    if config.chunker_type == "semantic":
        return SemanticChunker(
            embedding_model=config.chunk_tokenizer,
            chunk_size=config.chunk_size,
            min_sentences_per_chunk=1,
            min_characters_per_sentence=12,
            delim=[". ", "! ", "? ", "\n\n"],
        )
    return SentenceChunker(
        tokenizer=config.chunk_tokenizer,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        min_sentences_per_chunk=1,
        min_characters_per_sentence=12,
        delim=[". ", "! ", "? ", "\n\n"],
    )


@lru_cache(maxsize=1)
def get_default_chunker() -> ChunkerProtocol:
    """Get or create the default chunker (cached singleton).

    Returns:
        A cached SemanticChunker with default configuration.
    """
    return build_chunker(DEFAULT_CHUNK_CONFIG)


@dataclass(frozen=True)
class Chunk:
    """A chunk of a document with its metadata.

    Attributes:
        chunk_id: Unique ID in format "{doc_id}-{chunk_idx}".
        doc_id: Parent document ID.
        chunk_idx: Zero-based index of this chunk within the document.
        text: The chunk text content.
    """

    chunk_id: str
    doc_id: str
    chunk_idx: int
    text: str


def chunk_document(
    content: str,
    doc_id: str,
    title: str | None = None,
    decision_date: str | None = None,
    chunker: ChunkerProtocol | None = None,
) -> list[Chunk]:
    """Chunk a document into multiple indexed segments.

    Args:
        content: The document text content.
        doc_id: The parent document ID.
        title: Optional title for first chunk header.
        decision_date: Optional date for first chunk header.
        chunker: Pre-configured chunker instance. Uses default if None.

    Returns:
        List of Chunk objects with chunk_id = "{doc_id}-{idx}".
    """
    if chunker is None:
        chunker = get_default_chunker()

    raw_chunks = chunker.chunk(content or "")
    results: list[Chunk] = []

    for idx, chunk in enumerate(raw_chunks):
        chunk_id = f"{doc_id}-{idx}"
        if idx == 0 and (title or decision_date):
            header = FIRST_CHUNK_HEADER.format(
                title=title or "",
                decision_date=decision_date or "",
            )
            text = header + chunk.text
        else:
            text = chunk.text
        results.append(
            Chunk(chunk_id=chunk_id, doc_id=doc_id, chunk_idx=idx, text=text)
        )

    if not results:
        results.append(
            Chunk(chunk_id=f"{doc_id}-0", doc_id=doc_id, chunk_idx=0, text="")
        )

    return results


class DocumentChunkCache:
    """Cache for chunked documents to avoid re-chunking the same document.

    This cache stores the chunks for each document ID, allowing efficient
    access to chunk texts when multiple chunks from the same document are
    retrieved in a search.
    """

    def __init__(
        self,
        dataset,
        doc_id_column: str = "id",
        chunker: ChunkerProtocol | None = None,
    ) -> None:
        """Initialize the cache.

        Args:
            dataset: HuggingFace dataset containing documents.
            doc_id_column: Column name for document IDs.
            chunker: Chunker to use. Defaults to the shared default chunker.
        """
        self._cache: dict[str, list[Chunk]] = {}
        self._dataset = dataset
        self._doc_id_column = doc_id_column
        self._chunker = chunker or get_default_chunker()
        self._dataset_index: dict[str, int] | None = None

    def _build_index(self) -> None:
        """Build an index mapping doc_id → dataset row index for O(1) lookup."""
        if self._dataset_index is None:
            self._dataset_index = {}
            for idx, row in enumerate(self._dataset):
                doc_id = str(row[self._doc_id_column])
                self._dataset_index[doc_id] = idx

    def _get_row(self, doc_id: str) -> dict | None:
        """Get a dataset row by document ID.

        Args:
            doc_id: The document ID to look up.

        Returns:
            The row as a dict, or None if not found.
        """
        self._build_index()
        idx = self._dataset_index.get(doc_id)
        if idx is None:
            return None
        return dict(self._dataset[idx])

    def get_chunks(self, doc_id: str) -> list[Chunk] | None:
        """Get all chunks for a document, chunking it if not cached.

        Args:
            doc_id: The document ID.

        Returns:
            List of Chunk objects, or None if document not found.
        """
        if doc_id in self._cache:
            return self._cache[doc_id]

        row = self._get_row(doc_id)
        if row is None:
            return None

        chunks = chunk_document(
            content=row.get("content", ""),
            doc_id=doc_id,
            title=row.get("title", None),
            decision_date=row.get("decision_date", None),
            chunker=self._chunker,
        )
        self._cache[doc_id] = chunks
        return chunks

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Get a specific chunk by its chunk_id.

        Args:
            chunk_id: The chunk ID in format "{doc_id}-{chunk_idx}".

        Returns:
            The Chunk object, or None if not found.
        """
        parts = chunk_id.rsplit("-", 1)
        if len(parts) != 2:
            return None

        doc_id, chunk_idx_str = parts
        try:
            chunk_idx = int(chunk_idx_str)
        except ValueError:
            return None

        chunks = self.get_chunks(doc_id)
        if chunks is None or chunk_idx >= len(chunks):
            return None

        return chunks[chunk_idx]

    def get_chunk_text(self, chunk_id: str) -> str | None:
        """Get the text content of a specific chunk.

        Args:
            chunk_id: The chunk ID in format "{doc_id}-{chunk_idx}".

        Returns:
            The chunk text, or None if not found.
        """
        chunk = self.get_chunk(chunk_id)
        return chunk.text if chunk else None

    def get_document_metadata(self, doc_id: str) -> dict | None:
        """Get document metadata (title, date, etc.) without chunking.

        Args:
            doc_id: The document ID.

        Returns:
            Dict with document metadata, or None if not found.
        """
        row = self._get_row(doc_id)
        if row is None:
            return None
        return {
            "title": row.get("title", ""),
            "decision_date": row.get("decision_date", ""),
            "juridiction": row.get("juridiction", ""),
            "formation": row.get("formation", ""),
            "solution": row.get("solution", ""),
            "applied_law": row.get("applied_law", ""),
        }

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    def stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with cache stats (num_docs, num_chunks).
        """
        return {
            "num_docs": len(self._cache),
            "num_chunks": sum(len(chunks) for chunks in self._cache.values()),
        }
