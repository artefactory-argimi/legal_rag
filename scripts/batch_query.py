#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "absl-py>=2.1.0",
#     "datasets>=3.2.0",
#     "etils[eapp,epath]>=1.9.0",
#     "numpy>=1.24.0",
#     "pylate>=0.0.4",
#     "sentence-transformers>=3.4.1",
# ]
# ///
"""Query the PLAID index in batch and save results in MSMARCO format."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from absl import app, logging
from datasets import load_dataset, load_from_disk
from etils import eapp, epath

from legal_rag.retriever import build_encoder, build_retriever
from legal_rag.tools import parse_chunk_id


def chunk_matches_doc(chunk_id: str, target_doc_id: str) -> bool:
    """Check if a chunk ID belongs to the target document.

    Args:
        chunk_id: The retrieved chunk ID (format: "docid-chunkidx").
        target_doc_id: The target parent document ID.

    Returns:
        True if the chunk belongs to the target document.
    """
    parent_id, _ = parse_chunk_id(chunk_id)
    return parent_id == target_doc_id


@dataclass(frozen=True)
class IndexConfig:
    """PLAID index configuration."""

    path: epath.Path = epath.Path("./index")
    name: str = "legal_french_index"
    encoder: str = "maastrichtlawtech/colbert-legal-french"


@dataclass(frozen=True)
class SourceDatasetConfig:
    """Source dataset configuration for title lookup."""

    name: str = "artefactory/Argimi-Legal-French-Jurisprudence"
    subset: str = "juri"
    split: str = "train"
    doc_id_column: str = "id"
    title_column: str = "title"


@dataclass(frozen=True)
class QueryConfig:
    """Batch query configuration."""

    questions: epath.Path
    output: epath.Path
    index: IndexConfig = IndexConfig()
    source: SourceDatasetConfig = SourceDatasetConfig()
    k: int = 100
    batch_size: int = 32
    force: bool = False
    sample_interval: int = 100


def run_batch_query(cfg: QueryConfig) -> None:
    # Fail-fast checks before expensive operations.
    output = cfg.output.expanduser()
    run_file = output / "run.txt"
    questions_path = cfg.questions.expanduser()

    if not questions_path.exists():
        raise FileNotFoundError(f"Questions dataset not found at {questions_path}")
    if run_file.exists() and not cfg.force:
        raise FileExistsError(
            f"Run file already exists at {run_file}. Use --force to overwrite."
        )

    logging.info("Loading questions from %s", cfg.questions)
    questions_ds = load_from_disk(str(questions_path))
    if len(questions_ds) == 0:
        raise ValueError("No questions found in the dataset.")
    num_queries = len(questions_ds)
    logging.info("Loaded %d questions", num_queries)

    logging.info(
        "Loading source dataset %s/%s for title lookup",
        cfg.source.name,
        cfg.source.subset,
    )
    source_ds = load_dataset(cfg.source.name, cfg.source.subset, split=cfg.source.split)
    doc_id_to_title: dict[str, str] = {}
    doc_id_to_content: dict[str, str] = {}
    for row in source_ds:
        doc_id = row[cfg.source.doc_id_column]
        doc_id_to_title[doc_id] = row.get(cfg.source.title_column, "")
        doc_id_to_content[doc_id] = row.get("content", "")
    logging.info("Built title/content mapping for %d documents", len(doc_id_to_title))

    logging.info("Building encoder from %s", cfg.index.encoder)
    encoder = build_encoder(cfg.index.encoder)

    logging.info("Building retriever from %s", cfg.index.path)
    retriever = build_retriever(cfg.index.path, cfg.index.name)

    queries = [q["question"] for q in questions_ds]
    qids = [q["id"] for q in questions_ds]
    target_doc_ids = [q["doc_id"] for q in questions_ds]

    logging.info("Encoding %d queries in batches of %d", num_queries, cfg.batch_size)
    query_embeddings = encoder.encode(
        queries,
        is_query=True,
        show_progress_bar=True,
        batch_size=cfg.batch_size,
    )

    logging.info("Retrieving with k=%d", cfg.k)
    all_results = retriever.retrieve(
        queries_embeddings=query_embeddings,
        k=cfg.k,
    )

    logging.info("Building result matrices")
    retrieved_doc_ids: list[list[str]] = []
    scores = np.zeros((num_queries, cfg.k), dtype=np.float32)

    for q_idx, results in enumerate(all_results):
        doc_ids: list[str] = []
        for rank, res in enumerate(results):
            doc_id = str(res["id"])
            doc_ids.append(doc_id)
            scores[q_idx, rank] = res["score"]
        retrieved_doc_ids.append(doc_ids)

    output.mkdir(parents=True, exist_ok=True)
    qrels_file = output / "qrels.txt"

    logging.info("Writing MSMARCO format files")
    with run_file.open("w", encoding="utf-8") as f:
        for q_idx in range(num_queries):
            qid = qids[q_idx]
            seen_docs: set[str] = set()
            new_rank = 1
            first_chunk_id: str | None = None
            first_doc_id: str | None = None
            first_chunk_idx: int | None = None
            for rank in range(cfg.k):
                chunk_id = retrieved_doc_ids[q_idx][rank]
                doc_id, chunk_idx = parse_chunk_id(chunk_id)
                if doc_id in seen_docs:
                    continue
                seen_docs.add(doc_id)
                if first_doc_id is None:
                    first_chunk_id = chunk_id
                    first_doc_id = doc_id
                    first_chunk_idx = chunk_idx
                score = scores[q_idx, rank]
                f.write(f"{qid} Q0 {doc_id} {new_rank} {score:.6f} colbert\n")
                new_rank += 1

            if q_idx % cfg.sample_interval == 0:
                target_doc_id = target_doc_ids[q_idx]
                retrieved_title = doc_id_to_title.get(first_doc_id or "", "<unknown>")
                target_title = doc_id_to_title.get(target_doc_id, "<unknown>")
                retrieved_content = doc_id_to_content.get(first_doc_id or "", "")
                target_content = doc_id_to_content.get(target_doc_id, "")
                hit = "HIT" if first_doc_id == target_doc_id else "MISS"
                logging.info(
                    "Sample [%d/%d] %s\n"
                    "  Query: %s\n"
                    "  Retrieved: %s (chunk %s)\n"
                    "    Title: %s\n"
                    "    Content: %s\n"
                    "  Target: %s\n"
                    "    Title: %s\n"
                    "    Content: %s",
                    q_idx,
                    num_queries,
                    hit,
                    queries[q_idx],
                    first_chunk_id,
                    first_chunk_idx,
                    retrieved_title or "<no title>",
                    retrieved_content.replace("\n", " ")
                    if retrieved_content
                    else "<no content>",
                    target_doc_id,
                    target_title or "<no title>",
                    target_content.replace("\n", " ")
                    if target_content
                    else "<no content>",
                )

    if qrels_file.exists():
        logging.info("Qrels file already exists at %s, skipping", qrels_file)
    else:
        with qrels_file.open("w", encoding="utf-8") as f:
            for q_idx in range(num_queries):
                qid = qids[q_idx]
                doc_id = target_doc_ids[q_idx]
                f.write(f"{qid} 0 {doc_id} 1\n")
        logging.info("Wrote qrels file to %s (%d entries)", qrels_file, num_queries)

    logging.info(
        "Wrote run file to %s (%d queries x %d results)", run_file, num_queries, cfg.k
    )


if __name__ == "__main__":
    eapp.better_logging()
    flags_parser = eapp.make_flags_parser(QueryConfig)
    app.run(run_batch_query, flags_parser=flags_parser)
