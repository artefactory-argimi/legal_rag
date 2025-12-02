"""Tests for tools.py using a tiny ColBERT/PLAID pipeline."""

import json
from pathlib import Path

from absl.testing import absltest
from datasets import Dataset
from pylate import indexes, models, retrieve
from sqlitedict import SqliteDict

from legal_rag import indexer
from legal_rag.tools import lookup_legal_doc, search_legal_docs


def _fake_corpus() -> tuple[list[str], list[str]]:
    doc_ids = [f"doc-{i}" for i in range(5)]
    documents = [
        "French contract law requires consent, capacity, and a lawful cause.",
        "GDPR compliance requires a lawful basis for personal data processing.",
        "Maritime salvage law rewards rescuers who save ships in peril at sea.",
        "Tenant rights include habitability, repairs, and quiet enjoyment duties.",
        "Intellectual property spans patents, trademarks, and copyrights protection.",
    ]
    return doc_ids, documents


class ToolsTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.tmp_dir = Path(self.create_tempdir().full_path)
        self.index_folder = self.tmp_dir / "index"
        self.index_folder.mkdir(parents=True, exist_ok=True)

        doc_ids, documents = _fake_corpus()

        # Build the ColBERT encoder and fix embeddings like scripts/indexer.py.
        self.encoder = models.ColBERT(
            model_name_or_path="maastrichtlawtech/colbert-legal-french",
            document_length=128,
        )
        self.encoder = indexer.fix_colbert_embeddings(self.encoder)

        # Encode documents and create a PLAID index.
        documents_embeddings = self.encoder.encode(
            documents,
            is_query=False,
            show_progress_bar=False,
            batch_size=2,
        )

        index = indexes.PLAID(
            self.index_folder,
            index_name="test_index",
            override=True,
            show_progress=False,
        )
        index.add_documents(documents_ids=doc_ids, documents_embeddings=documents_embeddings)
        self.retriever = retrieve.ColBERT(index=index)

        # Persist doc mapping for search_legal_docs to enrich results.
        mapping_file = self.index_folder / "doc_mapping.json"
        sqlite_map = self.index_folder / "test_index" / "documents_ids_to_plaid_ids.sqlite"
        entries: dict[str, int] = {}
        with SqliteDict(sqlite_map, outer_stack=False) as db:
            for doc_id in doc_ids:
                plaid_id = db.get(doc_id)
                if plaid_id is not None:
                    entries[str(plaid_id)] = int(doc_id.split("-")[-1])
        mapping_file.write_text(
            json.dumps(
                {"dataset": "dummy_dataset", "split": "train", "config": "juri", "entries": entries},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # Patch dataset loading to a small in-memory dataset for lookup.
        rows = [
            {
                "title": f"title-{i}",
                "decision_date": "",
                "juridiction": "",
                "formation": "",
                "applied_law": "",
                "content": doc,
                "solution": "",
            }
            for i, doc in enumerate(documents)
        ]
        self.dataset = Dataset.from_dict(rows)
        self.entries = entries

    def test_search_returns_relevant_full_text(self):
        results = search_legal_docs(
            query="maritime salvage rewards rescuers",
            encoder=self.encoder,
            retriever=self.retriever,
            k=3,
        )

        self.assertTrue(results, "Expected at least one search result.")
        self.assertIn("id=", results)
        plaid_key = next(iter(self.entries))
        full_text = lookup_legal_doc(
            doc_id=plaid_key,
            mapping_entries=self.entries,
            dataset=self.dataset,
        )
        self.assertIn("salvage", full_text)

    def test_lookup_returns_full_text(self):
        plaid_key = next(iter(self.entries))
        text = lookup_legal_doc(
            doc_id=plaid_key,
            mapping_entries=self.entries,
            dataset=self.dataset,
        )
        self.assertIn("GDPR compliance", text)

    def test_lookup_uses_mapping_entries(self):
        plaid_key = next(iter(self.entries))
        text = lookup_legal_doc(
            doc_id=plaid_key,
            mapping_entries=self.entries,
            dataset=self.dataset,
        )
        self.assertIn("French contract law", text)


if __name__ == "__main__":
    absltest.main()
