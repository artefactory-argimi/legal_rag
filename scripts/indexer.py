import json
from dataclasses import dataclass
from uuid import uuid4

import grain
import toolz as tlz
from absl import app
from datasets import load_dataset
from etils import eapp, epath
from pylate import indexes, models


@dataclass(frozen=True)
class ScriptConfig:
    model: str = "maastrichtlawtech/colbert-legal-french"
    dataset: str = "artefactory/Argimi-Legal-French-Jurisprudence"
    subset: str = "cetat"
    split: str = "train"
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


def fix_colbert_embeddings(model):
    """
    Fix the token embedding size issue in ColBERT models.

    The bug: PyLate adds special tokens ([Q], [D]) but doesn't always
    properly resize the embedding layer, causing token IDs to be out of bounds.
    """
    # Get current sizes
    vocab_size = len(model.tokenizer)
    embedding_layer = model[0].auto_model.get_input_embeddings()
    embedding_size = embedding_layer.num_embeddings

    # Get special token IDs
    query_id = model.query_prefix_id
    doc_id = model.document_prefix_id

    print(f"Tokenizer vocab size: {vocab_size}")
    print(f"Embedding layer size: {embedding_size}")
    print(f"Query prefix '{model.query_prefix}': ID {query_id}")
    print(f"Document prefix '{model.document_prefix}': ID {doc_id}")

    # Calculate required size
    max_token_id = max(vocab_size - 1, query_id, doc_id)
    required_size = max_token_id + 1

    # Resize if needed
    if required_size > embedding_size:
        print("\n⚠️  Token IDs exceed embedding size!")
        print(f"Resizing from {embedding_size} to {required_size}")
        model[0].auto_model.resize_token_embeddings(required_size)
        new_size = model[0].auto_model.get_input_embeddings().num_embeddings
        print(f"✓ Resized to {new_size}")

        # Verify fix
        assert query_id < new_size, f"Query ID {query_id} still out of bounds!"
        assert doc_id < new_size, f"Document ID {doc_id} still out of bounds!"
        print("✓ All token IDs are now valid")
    else:
        print("✓ Embedding size is already sufficient")

    return model


def preprocess(sample):
    # TODO: Clean html from content
    # Process the document to add the metadata inside the temlated structure
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


def main(cfg: ScriptConfig):
    hf_ds = load_dataset(cfg.dataset, cfg.subset, split=cfg.split)
    ds = (
        grain.MapDataset.source(hf_ds)
        .shuffle(seed=cfg.seed)
        .map(preprocess)
        .to_iter_dataset()
    )
    doc_ids, documents = zip(*tlz.pluck(["document_id", "document"], iter(ds)))
    model = models.ColBERT(
        model_name_or_path=cfg.model,
        document_length=496,
    )
    model = fix_colbert_embeddings(model)

    documents_embeddings = model.encode(
        documents, is_query=False, show_progress_bar=True, batch_size=cfg.batch_size
    )
    index = indexes.PLAID(
        cfg.index_folder, index_name="legal_french_index", override=True
    )
    index.add_documents(
        documents_ids=doc_ids, documents_embeddings=documents_embeddings
    )
    print(f"\n✓ Index created successfully at {cfg.index_folder}")

    # Save doc_id -> document text mapping
    mapping_file = cfg.index_folder / "doc_mapping.json"
    doc_mapping = {doc_id: doc for doc_id, doc in zip(doc_ids, documents)}
    with mapping_file.open("w", encoding="utf-8") as f:
        json.dump(doc_mapping, f, ensure_ascii=False, indent=2)
    print(f"✓ Document mapping saved to {mapping_file}")


if __name__ == "__main__":
    eapp.better_logging()
    app.run(main, flags_parser=eapp.make_flags_parser(ScriptConfig))
