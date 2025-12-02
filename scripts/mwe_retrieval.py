"""Minimal example: load retriever, run a query, print results."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from datasets import load_dataset

from legal_rag.retriever import build_encoder, build_retriever
from legal_rag.tools import lookup_legal_doc, search_legal_docs

DEFAULT_DATASET = "artefactory/Argimi-Legal-French-Jurisprudence"
DEFAULT_CONFIG = "juri"
DEFAULT_SPLIT = "train"
DEFAULT_ENCODER = "maastrichtlawtech/colbert-legal-french"


def _resolve_index_dir(base: Path) -> Path:
    """Return the index root (parent of the folder that contains fast_plaid_index)."""
    fast_paths = sorted(
        (p for p in base.glob("**/fast_plaid_index") if p.is_dir()),
        key=lambda p: len(p.relative_to(base).parts),
    )
    if not fast_paths:
        raise FileNotFoundError(f"No fast_plaid_index found under {base}")
    fast_dir = fast_paths[0]
    # fast_dir = <index_root>/<index_name>/fast_plaid_index
    return fast_dir.parent.parent


def _load_mapping(
    mapping_path: Path,
    *,
    dataset_override: str | None,
    config_override: str | None,
    split_override: str | None,
) -> tuple[Mapping[str, Any], str, str, str]:
    """Load doc_mapping.json handling both legacy flat maps and new metadata format."""
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "entries" in payload:
        entries = payload.get("entries") or {}
        dataset_name = payload.get("dataset")
        config = payload.get("config")
        split = payload.get("split")
    elif isinstance(payload, dict):
        # Legacy mapping where the file itself is plaid_id -> dataset_idx.
        entries = payload
        dataset_name = None
        config = None
        split = None
    else:
        raise ValueError(f"Unrecognized doc_mapping.json structure in {mapping_path}")

    dataset_name = dataset_override or dataset_name or DEFAULT_DATASET
    config = config_override or config or DEFAULT_CONFIG
    split = split_override or split or DEFAULT_SPLIT
    return entries, dataset_name, config, split


def run_demo(
    *,
    encoder_dir: str | Path,
    index_dir: Path,
    query: str,
    k: int,
    dataset_name: str | None = None,
    dataset_config: str | None = None,
    dataset_split: str | None = None,
) -> None:
    """Fetch and print the top-k documents for the given query."""
    encoder_path = Path(encoder_dir)
    encoder_model = (
        str(encoder_path.expanduser().resolve())
        if encoder_path.exists()
        else str(encoder_dir)
    )
    encoder = build_encoder(encoder_model=encoder_model)
    resolved_index_dir = _resolve_index_dir(index_dir)
    retriever = build_retriever(
        index_folder=resolved_index_dir, index_name="legal_french_index"
    )

    # search_legal_docs returns a formatted string; parse ids and scores.
    results_str = search_legal_docs(
        query=query,
        encoder=encoder,
        retriever=retriever,
        k=k,
    )
    if not results_str or results_str == "No results.":
        print("No results.")
        return

    # For full lookup we need mapping and dataset (single-config) from doc_mapping.
    mapping_path = resolved_index_dir / "doc_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"doc_mapping.json not found under {resolved_index_dir}")
    entries, ds_name, config, split = _load_mapping(
        mapping_path,
        dataset_override=dataset_name,
        config_override=dataset_config,
        split_override=dataset_split,
    )

    dataset = load_dataset(ds_name, config, split=split)

    parsed = []
    for line in results_str.splitlines():
        if not line.strip():
            continue
        # Lines look like: "[0] id=123 score=0.1234"
        parts = line.replace("[", "").replace("]", "").split()
        doc_part = next((p for p in parts if p.startswith("id=")), "")
        score_part = next((p for p in parts if p.startswith("score=")), "")
        doc_id = doc_part.split("=", 1)[1] if doc_part else None
        score = score_part.split("=", 1)[1] if score_part else ""
        if doc_id:
            parsed.append({"id": doc_id, "score": score})

    if not parsed:
        print(results_str)
        return

    for idx, res in enumerate(parsed, start=1):
        doc_id = res["id"]
        text = lookup_legal_doc(doc_id, mapping_entries=entries, dataset=dataset)
        print(f"\n--- Result {idx} ---")
        print(f"id: {doc_id}")
        print(f"score: {res['score']}")
        print(f"text:\n{text}")


if __name__ == "__main__":
    os.environ.setdefault("JOBLIB_START_METHOD", "threading")
    os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    parser = argparse.ArgumentParser(description="MWE for querying a legal index.")
    parser.add_argument(
        "--encoder",
        default=DEFAULT_ENCODER,
        help=f"Encoder model id on Hugging Face (default: {DEFAULT_ENCODER}).",
    )
    parser.add_argument(
        "--index",
        default=".context/pylate",
        help="Path to index directory (anything above fast_plaid_index).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=f"Hugging Face dataset id for lookup (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--config",
        dest="config",
        default=None,
        help=f"Hugging Face dataset config (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--split",
        dest="split",
        default=None,
        help=f"Hugging Face dataset split (default: {DEFAULT_SPLIT}).",
    )
    parser.add_argument(
        "--query",
        default="Quel article définit la constitution française ?",
        help="Query text to search.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="Number of results to return.",
    )
    args = parser.parse_args()
    run_demo(
        encoder_dir=args.encoder,
        index_dir=Path(args.index).expanduser().resolve(),
        query=args.query,
        k=args.k,
        dataset_name=args.dataset,
        dataset_config=args.config,
        dataset_split=args.split,
    )
