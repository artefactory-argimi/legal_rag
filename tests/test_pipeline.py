"""Tests for individual pipeline steps: chunking, indexing, retrieval, lookup."""

from pathlib import Path

import numpy as np
from absl.testing import absltest
from chonkie import SentenceChunker
from pylate import indexes, models, retrieve

from legal_rag.indexer import chunk_document, preprocess, fix_colbert_embeddings
from legal_rag.tools import lookup_legal_doc, parse_chunk_id, search_legal_docs


class TestChunkingStep(absltest.TestCase):
    """Tests for the chunking step of the pipeline."""

    def setUp(self):
        super().setUp()
        self.chunker = SentenceChunker(
            tokenizer="character",
            chunk_size=511,
            chunk_overlap=0,
        )

    def test_document_is_chunked_with_correct_ids(self):
        content = "First sentence. Second sentence. Third sentence."
        chunks = chunk_document(
            content=content,
            doc_id="DOC1",
            title="Title",
            decision_date="2024",
            chunker=self.chunker,
        )
        self.assertGreaterEqual(len(chunks), 1)
        for idx, (chunk_id, _) in enumerate(chunks):
            self.assertEqual(chunk_id, f"DOC1-{idx}")

    def test_chunk_id_can_be_parsed_back_to_parent(self):
        chunks = chunk_document(
            content="Some content.",
            doc_id="PARENT123",
            title=None,
            decision_date=None,
            chunker=self.chunker,
        )
        chunk_id, _ = chunks[0]
        parent_id, idx = parse_chunk_id(chunk_id)
        self.assertEqual(parent_id, "PARENT123")
        self.assertEqual(idx, 0)


class TestIndexingStep(absltest.TestCase):
    """Tests for the indexing step - chunk IDs are stored correctly."""

    def setUp(self):
        super().setUp()
        self.tmpdir = Path(self.create_tempdir().full_path)
        self.chunker = SentenceChunker(
            tokenizer="character",
            chunk_size=511,
            chunk_overlap=0,
        )

    def test_chunk_ids_are_indexed(self):
        # Simulate chunk IDs that would come from chunk_document
        chunk_ids = ["DOC1-0", "DOC1-1", "DOC1-2", "DOC2-0"]
        embed_dim = 128
        tokens = 4
        embeddings = np.random.rand(len(chunk_ids), tokens, embed_dim).astype(
            np.float32
        )

        index = indexes.PLAID(
            index_folder=self.tmpdir,
            index_name="test_index",
            override=True,
            show_progress=False,
        )
        index.add_documents(documents_ids=chunk_ids, documents_embeddings=embeddings)

        # Verify chunk IDs are in the sqlite mapping
        from sqlitedict import SqliteDict

        docid_sqlite = self.tmpdir / "test_index" / "documents_ids_to_plaid_ids.sqlite"
        with SqliteDict(docid_sqlite, outer_stack=False) as mapping:
            for chunk_id in chunk_ids:
                self.assertIn(chunk_id, mapping)

    def test_chunked_document_ids_are_indexed_correctly(self):
        """Verify that chunk_document produces IDs that are correctly stored in index."""
        # Chunk a document using the real chunk_document function
        # Use longer content to produce multiple chunks
        content = "This is a legal sentence. " * 100
        doc_id = "JURITEXT000012345"
        chunks = chunk_document(
            content=content,
            doc_id=doc_id,
            title="Test Title",
            decision_date="2024-01-01",
            chunker=self.chunker,
        )

        # Ensure we have multiple chunks
        self.assertGreater(len(chunks), 1)

        # Extract chunk IDs and create fake embeddings
        chunk_ids = [cid for cid, _ in chunks]
        embed_dim = 128
        tokens = 4
        embeddings = np.random.rand(len(chunk_ids), tokens, embed_dim).astype(
            np.float32
        )

        # Index the chunks
        index = indexes.PLAID(
            index_folder=self.tmpdir,
            index_name="test_index",
            override=True,
            show_progress=False,
        )
        index.add_documents(documents_ids=chunk_ids, documents_embeddings=embeddings)

        # Verify all chunk IDs are in the index
        from sqlitedict import SqliteDict

        docid_sqlite = self.tmpdir / "test_index" / "documents_ids_to_plaid_ids.sqlite"
        with SqliteDict(docid_sqlite, outer_stack=False) as mapping:
            for chunk_id in chunk_ids:
                self.assertIn(chunk_id, mapping)
                # Verify each chunk ID can be parsed back to parent
                parent_id, chunk_idx = parse_chunk_id(chunk_id)
                self.assertEqual(parent_id, doc_id)

    def test_multiple_documents_indexed_with_distinct_chunk_ids(self):
        """Verify multiple documents produce distinct chunk IDs in the index."""
        documents = [
            {
                "doc_id": "DOC_A",
                "content": "Content for doc A. More content.",
                "title": "Doc A",
            },
            {
                "doc_id": "DOC_B",
                "content": "Content for doc B. Additional text.",
                "title": "Doc B",
            },
            {
                "doc_id": "DOC_C",
                "content": "Content for doc C. Extra info.",
                "title": "Doc C",
            },
        ]

        all_chunk_ids = []
        all_embeddings = []

        for doc in documents:
            chunks = chunk_document(
                content=doc["content"],
                doc_id=doc["doc_id"],
                title=doc["title"],
                decision_date="2024",
                chunker=self.chunker,
            )
            for chunk_id, _ in chunks:
                all_chunk_ids.append(chunk_id)
                # Create fake embedding
                all_embeddings.append(np.random.rand(4, 128).astype(np.float32))

        embeddings = np.array(all_embeddings)

        # Index all chunks
        index = indexes.PLAID(
            index_folder=self.tmpdir,
            index_name="test_index",
            override=True,
            show_progress=False,
        )
        index.add_documents(
            documents_ids=all_chunk_ids, documents_embeddings=embeddings
        )

        # Verify all chunk IDs are unique and in the index
        self.assertEqual(len(all_chunk_ids), len(set(all_chunk_ids)))

        from sqlitedict import SqliteDict

        docid_sqlite = self.tmpdir / "test_index" / "documents_ids_to_plaid_ids.sqlite"
        with SqliteDict(docid_sqlite, outer_stack=False) as mapping:
            self.assertEqual(len(mapping), len(all_chunk_ids))
            for chunk_id in all_chunk_ids:
                self.assertIn(chunk_id, mapping)


class TestRetrievalStep(absltest.TestCase):
    """Tests for the retrieval step - chunk IDs are returned."""

    def setUp(self):
        super().setUp()
        self.tmpdir = Path(self.create_tempdir().full_path)

        # Create index with chunk IDs
        self.chunk_ids = ["DOC1-0", "DOC1-1", "DOC2-0"]
        embed_dim = 128
        tokens = 4
        embeddings = np.random.rand(len(self.chunk_ids), tokens, embed_dim).astype(
            np.float32
        )

        index = indexes.PLAID(
            index_folder=self.tmpdir,
            index_name="test_index",
            override=True,
            show_progress=False,
        )
        index.add_documents(
            documents_ids=self.chunk_ids, documents_embeddings=embeddings
        )
        self.retriever = retrieve.ColBERT(index=index)

    def test_retrieved_ids_are_chunk_ids(self):
        # Create a random query embedding
        query_embedding = np.random.rand(1, 4, 128).astype(np.float32)
        results = self.retriever.retrieve(queries_embeddings=query_embedding, k=3)

        self.assertGreater(len(results), 0)
        for result in results[0]:
            retrieved_id = result["id"]
            # Verify the ID is parseable as a chunk ID
            parent_id, chunk_idx = parse_chunk_id(retrieved_id)
            self.assertIn(parent_id, ["DOC1", "DOC2"])
            self.assertIsInstance(chunk_idx, int)


class TestLookupStep(absltest.TestCase):
    """Tests for the lookup step - parent document is retrieved from chunk ID."""

    def test_lookup_retrieves_parent_document(self):
        dataset = [
            {
                "id": "DOC1",
                "title": "Document One Title",
                "content": "Document one content",
                "decision_date": "2024-01-01",
                "juridiction": "Court",
                "formation": "F",
                "solution": "S",
                "applied_law": "L",
            },
            {
                "id": "DOC2",
                "title": "Document Two Title",
                "content": "Document two content",
                "decision_date": "2024-02-02",
                "juridiction": "Court",
                "formation": "F",
                "solution": "S",
                "applied_law": "L",
            },
        ]

        # Lookup using chunk ID should return parent document
        result = lookup_legal_doc("DOC1-5", dataset=dataset)
        self.assertIn("Document One Title", result)
        self.assertIn("Document one content", result)

        result = lookup_legal_doc("DOC2-0", dataset=dataset)
        self.assertIn("Document Two Title", result)
        self.assertIn("Document two content", result)

    def test_lookup_different_chunks_same_parent(self):
        dataset = [
            {
                "id": "DOC1",
                "title": "Same Parent",
                "content": "Parent content",
                "decision_date": "2024",
                "juridiction": "J",
                "formation": "F",
                "solution": "S",
                "applied_law": "L",
            },
        ]

        # Different chunk IDs from same parent should return same document
        result_chunk0 = lookup_legal_doc("DOC1-0", dataset=dataset)
        result_chunk5 = lookup_legal_doc("DOC1-5", dataset=dataset)
        result_chunk99 = lookup_legal_doc("DOC1-99", dataset=dataset)

        self.assertEqual(result_chunk0, result_chunk5)
        self.assertEqual(result_chunk5, result_chunk99)


if __name__ == "__main__":
    absltest.main()
