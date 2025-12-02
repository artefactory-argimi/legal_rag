import argparse
import os
from pathlib import Path

import json

from datasets import load_dataset

from legal_rag.retriever import build_encoder, build_retriever
from legal_rag.tools import lookup_legal_doc, search_legal_docs

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
    index_root = fast_dir.parent.parent
    return index_root


def main() -> None:
    os.environ.setdefault("JOBLIB_START_METHOD", "threading")
    os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    parser = argparse.ArgumentParser(
        description="Query a legal index and print top result id + content."
    )
    parser.add_argument(
        "--encoder",
        default="maastrichtlawtech/colbert-legal-french",
        help="Encoder model id on Hugging Face (default: maastrichtlawtech/colbert-legal-french).",
    )
    parser.add_argument(
        "--index",
        required=True,
        help="Path to index directory containing legal_french_index and doc_mapping.json.",
    )
    parser.add_argument("--query", required=True, help="Query text to search.")
    parser.add_argument(
        "--k", type=int, default=1, help="Number of results to return (default: 1)."
    )
    args = parser.parse_args()

    encoder_path = args.encoder
    index_path = _resolve_index_dir(Path(args.index).expanduser().resolve())
    print(f"[legal_rag] Resolved index root: {index_path}")

    print(f"[legal_rag] Encoder: {encoder_path}")
    print(f"[legal_rag] Index:   {index_path}")
    encoder = build_encoder(encoder_model=str(encoder_path))
    retriever = build_retriever(index_folder=index_path, index_name="legal_french_index")

    # Load mapping and dataset once for lookup.
    mapping_path = index_path / "doc_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"doc_mapping.json not found under {index_path}")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    entries = mapping.get("entries") or {}
    dataset_name = mapping.get("dataset")
    config = mapping.get("config") or "juri"
    split = mapping.get("split") or "train"
    dataset = load_dataset(dataset_name, config, split=split)

    results_str = search_legal_docs(
        query=args.query,
        encoder=encoder,
        retriever=retriever,
        k=args.k,
    )
    if not results_str or results_str == "No results.":
        print("No results.")
        return

    # Parse back minimal info from the formatted string lines.
    parsed = []
    for line in results_str.splitlines():
        if not line.strip():
            continue
        parts = line.replace("[", "").replace("]", "").split()
        doc_id = parts[1].split("=")[1] if len(parts) > 1 and "id=" in parts[1] else None
        score_part = parts[2] if len(parts) > 2 else ""
        score = score_part.split("=")[1] if "score=" in score_part else ""
        if doc_id:
            parsed.append({"id": doc_id, "score": score})

    if not parsed:
        print(results_str)
        return

    for idx, res in enumerate(parsed, start=1):
        doc_id = res["id"]
        score = res.get("score")
        text = lookup_legal_doc(doc_id, mapping_entries=entries, dataset=dataset)
        print("\n--- Result", idx, "---")
        print(f"id: {doc_id}")
        print(f"score: {score}")
        print(f"text:\n{text}")


if __name__ == "__main__":
    main()
