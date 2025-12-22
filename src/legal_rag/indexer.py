import os
from dataclasses import dataclass
from functools import partial

import grain
import toolz as tlz
from absl import logging
from datasets import load_dataset
from etils import epath
from pylate import indexes, models

from legal_rag.chunking import (
    FIRST_CHUNK_HEADER,
    ChunkConfig,
    build_chunker,
    chunk_document,
)
from legal_rag.colbert_utils import fix_colbert_embeddings
from legal_rag.tools import DEFAULT_DOC_ID_COLUMN, TEMPLATE_DOCUMENT

__all__ = [
    "ScriptConfig",
    "TEMPLATE_DOCUMENT",
    "FIRST_CHUNK_HEADER",
    "chunk_document",
    "preprocess",
    "build_index",
    "fix_colbert_embeddings",
]


@dataclass(frozen=True)
class ScriptConfig:
    model: str = "maastrichtlawtech/colbert-legal-french"
    dataset: str = "artefactory/Argimi-Legal-French-Jurisprudence"
    subset: str | None = None  # optional config override (cetat, juri, constit)
    split: str | None = None  # Hugging Face split name (defaults to train if None)
    slice_size: int | None = None  # limit number of rows for debugging
    seed: int = 42
    batch_size: int = 1024  # batch size for encoding (affects GPU memory usage)
    accumulation_size: int | None = (
        None  # docs to accumulate before encoding (defaults to batch_size)
    )
    index_folder: epath.Path = epath.Path("./index")
    index_name: str = "legal_french_index"
    doc_id_column: str = DEFAULT_DOC_ID_COLUMN
    device: str | None = None  # device for model ("cuda", "cpu", None for auto)
    # PLAID index configuration
    nbits: int = 4  # bits for quantization (1-4, lower = smaller index)
    pool_factor: int = 8  # reduce token embeddings by this factor (1 = no pooling)
    # Chunking configuration
    chunk_size: int = 511  # max tokens per chunk (ColBERT limit)
    chunk_overlap: int = 0  # overlap between chunks (SentenceChunker only)
    chunker_type: str = "semantic"  # "semantic" or "sentence"
    # For SentenceChunker: tokenizer name (e.g., "gpt2", "character")
    # For SemanticChunker: embedding model (e.g., "minishlab/potion-base-32M")
    chunk_tokenizer: str = "minishlab/potion-base-32M"
    # Force re-indexing even if output already exists (default: skip if exists)
    force: bool = False
    # ColBERT model configuration
    query_prefix: str | None = None  # e.g., "[QueryMarker]" for Jina ColBERT v2
    document_prefix: str | None = None  # e.g., "[DocumentMarker]" for Jina ColBERT v2
    attend_to_expansion_tokens: bool | None = None  # True for Jina ColBERT v2
    trust_remote_code: bool = False  # required for models with custom code (e.g., Jina)


def preprocess(sample, doc_id_column: str = DEFAULT_DOC_ID_COLUMN):
    """Preprocess a sample from the dataset for chunking.

    Args:
        sample: A row from the HuggingFace dataset.
        doc_id_column: Name of the column containing the document ID.

    Returns:
        A dict with 'doc_id', 'content', 'title', and 'decision_date' keys.
    """
    content, decision_date, title = tlz.get(
        ["content", "decision_date", "title"],
        sample,
        default=None,
    )
    return {
        "doc_id": sample[doc_id_column],
        "content": content or "",
        "title": title,
        "decision_date": decision_date,
    }


def build_index(cfg: ScriptConfig):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    logging.info("Indexer configuration:")
    logging.info("  model: %s", cfg.model)
    logging.info("  dataset: %s", cfg.dataset)
    logging.info("  subset: %s", cfg.subset)
    logging.info("  split: %s", cfg.split)
    logging.info("  index_folder: %s", cfg.index_folder)
    logging.info("  index_name: %s", cfg.index_name)
    logging.info("  batch_size: %d", cfg.batch_size)
    logging.info("  accumulation_size: %s", cfg.accumulation_size)
    logging.info("  doc_id_column: %s", cfg.doc_id_column)
    logging.info("  nbits: %d", cfg.nbits)
    logging.info("  pool_factor: %d (1 = no pooling)", cfg.pool_factor)
    logging.info("  chunk_size: %d", cfg.chunk_size)
    logging.info("  chunker_type: %s", cfg.chunker_type)
    logging.info("  force: %s", cfg.force)

    index_path = epath.Path(cfg.index_folder) / cfg.index_name
    fast_plaid_path = index_path / "fast_plaid_index"
    if index_path.exists() and not cfg.force:
        # Check if the fast_plaid_index subdirectory exists (indicates a complete index)
        if fast_plaid_path.exists():
            logging.info(
                "Valid index already exists at %s, skipping indexation", index_path
            )
            print(index_path)
            raise SystemExit(0)
        else:
            raise FileExistsError(
                f"Directory exists at {index_path} but is not a valid index "
                f"(missing {fast_plaid_path}). Use --force to rebuild."
            )

    logging.info(
        "Building PLAID index: encoder=%s dataset=%s slice=%s output=%s",
        cfg.model,
        cfg.dataset,
        cfg.slice_size,
        cfg.index_folder,
    )

    # Single-config flow: index one config (default juri unless overridden).
    config_name = cfg.subset or "juri"
    split_name = cfg.split or "train"

    model = models.ColBERT(
        model_name_or_path=cfg.model,
        document_length=496,  # TODO(hicham): the document length should be configurable
        device=cfg.device,
        query_prefix=cfg.query_prefix,
        document_prefix=cfg.document_prefix,
        attend_to_expansion_tokens=cfg.attend_to_expansion_tokens,
        trust_remote_code=cfg.trust_remote_code,
    )
    logging.info("Model loaded on device: %s", model.device)
    model = fix_colbert_embeddings(model)
    logging.info("Model loaded and embeddings fixed")

    index = indexes.PLAID(
        str(cfg.index_folder),
        index_name=cfg.index_name,
        override=True,
        nbits=cfg.nbits,
    )

    chunk_config = ChunkConfig(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        chunker_type=cfg.chunker_type,
        chunk_tokenizer=cfg.chunk_tokenizer,
    )
    chunker = build_chunker(chunk_config)
    logging.info(
        "Initialized %s chunker with chunk_size=%d, tokenizer=%s",
        cfg.chunker_type,
        cfg.chunk_size,
        cfg.chunk_tokenizer,
    )

    logging.info("Loading dataset config=%s split=%s", config_name, split_name)
    hf_ds = load_dataset(cfg.dataset, config_name, split=split_name)
    base_ds = grain.MapDataset.source(hf_ds)
    if cfg.slice_size:
        base_ds = base_ds.slice(slice(0, cfg.slice_size))
    ds = base_ds.map(
        partial(preprocess, doc_id_column=cfg.doc_id_column)
    ).to_iter_dataset()

    def generate_chunks():
        """Yield (chunk_id, chunk_text) pairs from all documents."""
        for item in ds:
            chunks = chunk_document(
                content=item["content"],
                doc_id=str(item["doc_id"]),
                title=item["title"],
                decision_date=item["decision_date"],
                chunker=chunker,
            )
            for chunk in chunks:
                yield chunk.chunk_id, chunk.text

    def encode_and_index_batch(doc_ids: list[str], docs: list[str]) -> int:
        """Encode documents and add them to the index.

        Args:
            doc_ids: List of document IDs.
            docs: List of document texts.

        Returns:
            Number of documents indexed.
        """
        if not doc_ids:
            return 0

        logging.info(
            "Encoding %d documents with batch_size=%d", len(docs), cfg.batch_size
        )
        embeddings = model.encode(
            docs,
            is_query=False,
            show_progress_bar=True,
            batch_size=cfg.batch_size,
            pool_factor=cfg.pool_factor,
        )

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                logging.info("Cleared CUDA cache before PLAID add_documents.")
        except Exception:
            pass

        index.add_documents(documents_ids=doc_ids, documents_embeddings=embeddings)
        logging.info("Indexed %d documents in this batch", len(doc_ids))
        return len(doc_ids)

    accumulation_size = cfg.accumulation_size or cfg.batch_size
    batch_ids: list[str] = []
    batch_texts: list[str] = []
    total = 0

    for chunk_id, chunk_text in generate_chunks():
        batch_ids.append(chunk_id)
        batch_texts.append(chunk_text)
        if len(batch_ids) >= accumulation_size:
            total += encode_and_index_batch(batch_ids, batch_texts)
            batch_ids = []
            batch_texts = []

    # Process remaining documents.
    total += encode_and_index_batch(batch_ids, batch_texts)

    logging.info("Indexing complete (%d chunks), stored at %s", total, index_path)
    return index_path


# Keep the historical entry point name for compatibility.
main = build_index
