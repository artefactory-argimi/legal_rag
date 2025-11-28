import os
from dataclasses import dataclass
from uuid import uuid4

import grain
import toolz as tlz
from absl import logging
from datasets import load_dataset
from etils import epath
from pylate import indexes, models

from legal_rag.colbert_utils import fix_colbert_embeddings
from legal_rag.assets import extract_zip, fetch_zip, resolve_model_dir

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
    encoder_zip_uri: str = "https://github.com/artefactory-argimi/legal_rag/releases/download/data-juri-v1/colbert-encoder.zip"
    dataset: str = "artefactory/Argimi-Legal-French-Jurisprudence"
    subset: str | None = None  # optional config override (cetat, juri, constit)
    split: str | None = None  # Hugging Face split name (defaults to train if None)
    slice_size: int | None = None  # limit number of rows for debugging
    seed: int = 42
    batch_size: int = 1024
    index_folder: epath.Path = epath.Path("./index")


TEMPLATE_DOCUMENT = """Titre : {title}
Date : {decision_date}
Juridiction : {juridiction}
Formation : {formation}
Solution : {solution}
Droit appliqué : {applied_law}
Texte de la décision : {content}
"""


def preprocess(sample):
    # TODO: Clean html from content
    # TODO: Split long decisions into chunks, apply the template per chunk, and index all chunks to avoid losing text beyond the 496-token limit.
    # Process the document to add the metadata inside the templated structure
    content, decision_date, title, juridiction, formation, solution, applied_law = tlz.get(
        ["content", "decision_date", "title", "juridiction", "formation", "solution", "applied_law"],
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
            applied_law=applied_law or "",
            content=content or "",
            solution=solution or "",
        ),
    }


def build_index(cfg: ScriptConfig):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logging.info(
        "Building PLAID index: encoder=%s dataset=%s slice=%s output=%s",
        cfg.encoder_zip_uri or cfg.model,
        cfg.dataset,
        cfg.slice_size,
        cfg.index_folder,
    )
    tmp_model_dir = None
    model_path = cfg.model
    if cfg.encoder_zip_uri:
        import shutil
        import tempfile

        tmp_model_dir = epath.Path(tempfile.mkdtemp(prefix="colbert_model_"))
        enc_zip_path = tmp_model_dir / "encoder.zip"
        logging.info("Fetching encoder zip from %s", cfg.encoder_zip_uri)
        fetch_zip(cfg.encoder_zip_uri, enc_zip_path)
        try:
            encoder_dir = extract_zip(enc_zip_path, tmp_model_dir / "encoder")
            encoder_dir = resolve_model_dir(encoder_dir)
            model_path = str(encoder_dir)
            logging.info("Downloaded encoder from %s to %s", cfg.encoder_zip_uri, model_path)
        finally:
            enc_zip_path.unlink(missing_ok=True)
    elif str(cfg.model).endswith(".zip"):
        import shutil
        import tempfile
        import zipfile

        model_zip = epath.Path(cfg.model)
        if not model_zip.exists():
            raise FileNotFoundError(f"Model zip not found: {model_zip}")
        tmp_model_dir = epath.Path(tempfile.mkdtemp(prefix="colbert_model_"))
        with zipfile.ZipFile(model_zip, "r") as zf:
            zf.extractall(tmp_model_dir)
        model_path = str(resolve_model_dir(tmp_model_dir))
        logging.info("Extracted model zip to %s", model_path)

    # Always index the three known configs unless explicitly overridden.
    config_list = [cfg.subset] if cfg.subset else ["cetat", "constit", "juri"]
    split_name = cfg.split or "train"
    try:
        model = models.ColBERT(
            model_name_or_path=model_path,
            document_length=496,
        )
        model = fix_colbert_embeddings(model)
        logging.info("Model loaded and embeddings fixed")

        index = indexes.PLAID(
            cfg.index_folder, index_name="legal_french_index", override=True
        )
        CHUNK_SIZE = 32768
        batch_ids: list[str] = []
        batch_docs: list[str] = []

        def flush_batch() -> int:
            if not batch_ids:
                return 0
            embeddings = model.encode(
                batch_docs,
                is_query=False,
                show_progress_bar=True,
                batch_size=cfg.batch_size,
            )
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    logging.info("Cleared CUDA cache before PLAID add_documents.")
            except Exception:
                pass
            index.add_documents(documents_ids=batch_ids, documents_embeddings=embeddings)
            processed = len(batch_ids)
            logging.info("Indexed %d documents in this batch", processed)
            batch_ids.clear()
            batch_docs.clear()
            return processed

        total = 0
        for config in config_list:
            logging.info("Loading dataset config=%s split=%s", config, split_name)
            hf_ds = load_dataset(cfg.dataset, config, split=split_name)
            base_ds = grain.MapDataset.source(hf_ds)
            if cfg.slice_size:
                # Use Grain's built-in slicing for fast subset debugging.
                base_ds = base_ds.slice(slice(0, cfg.slice_size))
            ds = base_ds.shuffle(seed=cfg.seed).map(preprocess).to_iter_dataset()
            for doc_id, document in tlz.pluck(["document_id", "document"], iter(ds)):
                batch_ids.append(doc_id)
                batch_docs.append(document)
                if len(batch_ids) >= CHUNK_SIZE:
                    total += flush_batch()
        # Flush remainder
        total += flush_batch()

        logging.info("Indexing complete (%d documents), stored at %s", total, cfg.index_folder)
        print(f"\n✓ Index created successfully at {cfg.index_folder} ({total} documents)")
    finally:
        if tmp_model_dir:
            import shutil

            shutil.rmtree(tmp_model_dir, ignore_errors=True)
            logging.info("Cleaned up temporary model directory %s", tmp_model_dir)


# Keep the historical entry point name for compatibility.
main = build_index
