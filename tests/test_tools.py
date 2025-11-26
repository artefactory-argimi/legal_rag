"""Tests for tools.py using a tiny ColBERT/PLAID pipeline."""

import json
from pathlib import Path

from absl.testing import absltest
from pylate import indexes, models, retrieve

from legal_rag import indexer
from legal_rag.tools import (
    load_doc_mapping,
    lookup_legal_doc,
    search_legal_docs,
    search_legal_docs_metadata,
)


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
        with mapping_file.open("w", encoding="utf-8") as f:
            json.dump({doc_id: doc for doc_id, doc in zip(doc_ids, documents)}, f)

        load_doc_mapping.cache_clear()

    def test_search_returns_relevant_full_text(self):
        results = search_legal_docs(
            query="maritime salvage rewards rescuers",
            encoder=self.encoder,
            retriever=self.retriever,
            index_folder=self.index_folder,
            k=3,
        )

        self.assertTrue(results, "Expected at least one search result.")
        top = results[0]
        self.assertEqual(top["id"], "doc-2")
        self.assertIn("salvage", top["text"])
        self.assertLessEqual(len(results), 3)

    def test_search_metadata_returns_ids_and_previews(self):
        results = search_legal_docs_metadata(
            query="personal data compliance",
            encoder=self.encoder,
            retriever=self.retriever,
            index_folder=self.index_folder,
            k=3,
            preview_chars=50,
        )
        self.assertTrue(results, "Expected metadata results.")
        top = results[0]
        self.assertIn("doc-", top["id"])
        self.assertTrue(top["preview"])
        self.assertLessEqual(len(top["preview"]), 50)
        self.assertIn("title", top["metadata"])
        self.assertLessEqual(len(results), 3)

    def test_lookup_returns_full_text(self):
        text = lookup_legal_doc("doc-1", index_folder=self.index_folder)
        self.assertIn("GDPR compliance", text)

    def test_load_doc_mapping_is_cached(self):
        mapping1 = load_doc_mapping(str(self.index_folder))
        mapping2 = load_doc_mapping(str(self.index_folder))

        self.assertIs(mapping1, mapping2)
        self.assertIn("doc-0", mapping1)
        self.assertIn("French contract law", mapping1["doc-0"])


if __name__ == "__main__":
    absltest.main()
