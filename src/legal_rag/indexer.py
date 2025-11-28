from dataclasses import dataclass
from uuid import uuid4

import grain
import toolz as tlz
from absl import logging
from datasets import load_dataset
from etils import epath
from pylate import indexes, models

from legal_rag.colbert_utils import fix_colbert_embeddings

__all__ = [
    "ScriptConfig",
    "TEMPLATE_DOCUMENT",
    "preprocess",
    "build_index",
    "fix_colbert_embeddings",
]


@dataclass(frozen=True)
class ScriptConfig:
    model: str = "maastrichtlawtech/colbert-legal-french"
    dataset: str = "artefactory/Argimi-Legal-French-Jurisprudence"
    subset: str = "cetat"
    split: str = "train"
    slice_size: int | None = None  # limit number of rows for debugging
    seed: int = 42
    batch_size: int = 1024
    index_folder: epath.Path = epath.Path("./index")


TEMPLATE_DOCUMENT = """Title: {title}
Date: {decision_date}
Jurisdiction: {juridiction}
Formation: {formation}
Solution: {solution}
Decision Text: {content}
"""


def preprocess(sample):
    # TODO: Clean html from content
    # Process the document to add the metadata inside the templated structure
    content, decision_date, title, juridiction, formation, solution = tlz.get(
        ["content", "decision_date", "title", "juridiction", "formation", "solution"],
        sample,
        default=None,
    )
    doc_id = str(uuid4())
    return {
        "document_id": doc_id,
        "document": TEMPLATE_DOCUMENT.format(
            title=title or "",
            decision_date=decision_date or "",
            juridiction=juridiction or "",
            formation=formation or "",
            content=content or "",
            solution=solution or "",
        ),
    }


def build_index(cfg: ScriptConfig):
    logging.info(
        "Starting index build: model=%s dataset=%s subset=%s split=%s slice_size=%s index_folder=%s",
        cfg.model,
        cfg.dataset,
        cfg.subset,
        cfg.split,
        cfg.slice_size,
        cfg.index_folder,
    )
    hf_ds = load_dataset(cfg.dataset, cfg.subset, split=cfg.split)
    base_ds = grain.MapDataset.source(hf_ds)
    if cfg.slice_size:
        # Use Grain's built-in slicing for fast subset debugging.
        base_ds = base_ds.slice(slice(0, cfg.slice_size))
    ds = base_ds.shuffle(seed=cfg.seed).map(preprocess).to_iter_dataset()
    doc_ids, documents = zip(*tlz.pluck(["document_id", "document"], iter(ds)))
    logging.info("Prepared %d documents for indexing", len(documents))

    model = models.ColBERT(
        model_name_or_path=cfg.model,
        document_length=496,
    )
    model = fix_colbert_embeddings(model)
    logging.info("Model loaded and embeddings fixed")

    documents_embeddings = model.encode(
        documents, is_query=False, show_progress_bar=True, batch_size=cfg.batch_size
    )
    index = indexes.PLAID(
        cfg.index_folder, index_name="legal_french_index", override=True
    )
    index.add_documents(
        documents_ids=doc_ids, documents_embeddings=documents_embeddings
    )
    logging.info("Indexing complete, stored at %s", cfg.index_folder)
    print(f"\n✓ Index created successfully at {cfg.index_folder}")


# Keep the historical entry point name for compatibility.
main = build_index
