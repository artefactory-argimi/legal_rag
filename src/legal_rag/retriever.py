"""Helpers to load the ColBERT encoder and PLAID retriever without agent wiring."""

from etils import epath
from pylate import indexes, models, retrieve

from legal_rag.colbert_utils import fix_colbert_embeddings

DEFAULT_ENCODER_MODEL = "maastrichtlawtech/colbert-legal-french"
DEFAULT_INDEX_FOLDER = epath.Path("./index")
DEFAULT_INDEX_NAME = "legal_french_index"


def build_encoder(
    encoder_model: str = DEFAULT_ENCODER_MODEL,
    query_prefix: str | None = None,
    document_prefix: str | None = None,
    attend_to_expansion_tokens: bool | None = None,
    trust_remote_code: bool = False,
) -> models.ColBERT:
    """Load ColBERT using PyLate's native loading (same as indexer).

    Args:
        encoder_model: HuggingFace model name or path.
        query_prefix: Prefix for queries (e.g., "[QueryMarker]" for Jina ColBERT v2).
        document_prefix: Prefix for documents (e.g., "[DocumentMarker]" for Jina ColBERT v2).
        attend_to_expansion_tokens: Whether to attend to expansion tokens (True for Jina).
        trust_remote_code: Whether to trust remote code (required for Jina ColBERT v2).

    Returns:
        Configured ColBERT encoder.
    """
    print(f"[legal_rag] Loading encoder from {encoder_model}")
    encoder = models.ColBERT(
        model_name_or_path=encoder_model,
        document_length=496,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
        attend_to_expansion_tokens=attend_to_expansion_tokens,
        trust_remote_code=trust_remote_code,
    )
    return fix_colbert_embeddings(encoder)


def build_retriever(
    index_folder: epath.Path = DEFAULT_INDEX_FOLDER,
    index_name: str = DEFAULT_INDEX_NAME,
) -> retrieve.ColBERT:
    """Load and return the PLAID retriever."""
    print(f"[legal_rag] Loading index from {index_folder} (name={index_name})")

    index = indexes.PLAID(
        index_folder,
        index_name=index_name,
        override=False,
        show_progress=False,
    )
    # Avoid spawning multiple CPU workers in constrained environments (e.g., mac sandbox).
    try:
        fast_plaid = index._index.fast_plaid  # type: ignore[attr-defined]
        devices = getattr(fast_plaid, "devices", None)
        if devices and len(devices) > 1:
            fast_plaid.devices = ["cpu"]
            print("[legal_rag] FastPLAID constrained to a single CPU worker.")
    except Exception:
        pass

    try:
        is_indexed = getattr(index._index, "is_indexed", None)
        print(f"[legal_rag] PLAID index status is_indexed={is_indexed}")
    except Exception:
        print("[legal_rag] PLAID index status unavailable")

    return retrieve.ColBERT(index=index)
