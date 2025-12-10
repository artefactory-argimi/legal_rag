"""Unit tests for document chunking functionality."""

import os

# Force CPU-only mode for tests
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from absl.testing import absltest
from chonkie import SemanticChunker, SentenceChunker

from legal_rag.indexer import FIRST_CHUNK_HEADER, chunk_document, preprocess
from legal_rag.tools import parse_chunk_id


class TestParseChunkId(absltest.TestCase):
    """Tests for parse_chunk_id function."""

    def test_parse_simple_chunk_id(self):
        doc_id, chunk_idx = parse_chunk_id("DOC123-0")
        self.assertEqual(doc_id, "DOC123")
        self.assertEqual(chunk_idx, 0)

    def test_parse_chunk_id_with_higher_index(self):
        doc_id, chunk_idx = parse_chunk_id("DOC123-15")
        self.assertEqual(doc_id, "DOC123")
        self.assertEqual(chunk_idx, 15)

    def test_parse_chunk_id_with_hyphen_in_doc_id(self):
        doc_id, chunk_idx = parse_chunk_id("JURITEXT000007022836-2")
        self.assertEqual(doc_id, "JURITEXT000007022836")
        self.assertEqual(chunk_idx, 2)

    def test_parse_chunk_id_with_multiple_hyphens(self):
        doc_id, chunk_idx = parse_chunk_id("DOC-WITH-HYPHENS-5")
        self.assertEqual(doc_id, "DOC-WITH-HYPHENS")
        self.assertEqual(chunk_idx, 5)


class TestChunkDocumentWithSentenceChunker(absltest.TestCase):
    """Tests for chunk_document function with SentenceChunker."""

    def setUp(self):
        super().setUp()
        self.chunker = SentenceChunker(
            tokenizer="character",
            chunk_size=511,
            chunk_overlap=0,
        )

    def test_single_chunk_document(self):
        chunks = chunk_document(
            content="Short text.",
            doc_id="DOC1",
            title="Title",
            decision_date="2024-01-01",
            chunker=self.chunker,
        )
        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0][0], "DOC1-0")
        self.assertIn("Title", chunks[0][1])

    def test_multiple_chunks(self):
        long_content = "Sentence one. " * 200
        chunks = chunk_document(
            content=long_content,
            doc_id="DOC2",
            title="Title",
            decision_date="2024-01-01",
            chunker=self.chunker,
        )
        self.assertGreater(len(chunks), 1)
        for idx, (chunk_id, _) in enumerate(chunks):
            self.assertEqual(chunk_id, f"DOC2-{idx}")
        self.assertIn("Title", chunks[0][1])
        if len(chunks) > 1:
            self.assertNotIn("Title", chunks[1][1])

    def test_empty_document(self):
        chunks = chunk_document(
            content="",
            doc_id="EMPTY",
            title=None,
            decision_date=None,
            chunker=self.chunker,
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][0], "EMPTY-0")

    def test_long_content_produces_multiple_chunks(self):
        # Verify that long content with sentences produces multiple chunks
        long_content = "This is a sentence. " * 200
        chunks = chunk_document(
            content=long_content,
            doc_id="LONG",
            title=None,
            decision_date=None,
            chunker=self.chunker,
        )
        self.assertGreater(len(chunks), 1)

    def test_first_chunk_has_header_when_metadata_present(self):
        chunks = chunk_document(
            content="Some legal text content.",
            doc_id="DOC3",
            title="Test Title",
            decision_date="2024-05-15",
            chunker=self.chunker,
        )
        self.assertIn("Test Title", chunks[0][1])
        self.assertIn("2024-05-15", chunks[0][1])

    def test_no_header_when_no_metadata(self):
        chunks = chunk_document(
            content="Some legal text content.",
            doc_id="DOC4",
            title=None,
            decision_date=None,
            chunker=self.chunker,
        )
        self.assertNotIn("Titre:", chunks[0][1])

    def test_chunk_ids_are_sequential(self):
        long_content = "A sentence. " * 500
        chunks = chunk_document(
            content=long_content,
            doc_id="SEQ",
            title=None,
            decision_date=None,
            chunker=self.chunker,
        )
        expected_ids = [f"SEQ-{i}" for i in range(len(chunks))]
        actual_ids = [chunk_id for chunk_id, _ in chunks]
        self.assertEqual(actual_ids, expected_ids)


class TestChunkDocumentWithSemanticChunker(absltest.TestCase):
    """Tests for chunk_document function with SemanticChunker."""

    def setUp(self):
        super().setUp()
        self.chunker = SemanticChunker(
            embedding_model="minishlab/potion-base-8M",
            chunk_size=511,
            min_sentences_per_chunk=1,
            min_characters_per_sentence=12,
        )

    def test_single_chunk_document(self):
        chunks = chunk_document(
            content="Short text.",
            doc_id="DOC1",
            title="Title",
            decision_date="2024-01-01",
            chunker=self.chunker,
        )
        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0][0], "DOC1-0")
        self.assertIn("Title", chunks[0][1])

    def test_multiple_chunks(self):
        long_content = "Sentence one about legal matters. " * 200
        chunks = chunk_document(
            content=long_content,
            doc_id="DOC2",
            title="Title",
            decision_date="2024-01-01",
            chunker=self.chunker,
        )
        self.assertGreater(len(chunks), 1)
        for idx, (chunk_id, _) in enumerate(chunks):
            self.assertEqual(chunk_id, f"DOC2-{idx}")
        self.assertIn("Title", chunks[0][1])
        if len(chunks) > 1:
            self.assertNotIn("Title", chunks[1][1])

    def test_empty_document(self):
        chunks = chunk_document(
            content="",
            doc_id="EMPTY",
            title=None,
            decision_date=None,
            chunker=self.chunker,
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][0], "EMPTY-0")

    def test_long_content_produces_multiple_chunks(self):
        long_content = "This is a legal sentence about constitutional law. " * 200
        chunks = chunk_document(
            content=long_content,
            doc_id="LONG",
            title=None,
            decision_date=None,
            chunker=self.chunker,
        )
        self.assertGreater(len(chunks), 1)

    def test_first_chunk_has_header_when_metadata_present(self):
        chunks = chunk_document(
            content="Some legal text content.",
            doc_id="DOC3",
            title="Test Title",
            decision_date="2024-05-15",
            chunker=self.chunker,
        )
        self.assertIn("Test Title", chunks[0][1])
        self.assertIn("2024-05-15", chunks[0][1])

    def test_chunk_ids_are_sequential(self):
        long_content = "A legal sentence about law. " * 500
        chunks = chunk_document(
            content=long_content,
            doc_id="SEQ",
            title=None,
            decision_date=None,
            chunker=self.chunker,
        )
        expected_ids = [f"SEQ-{i}" for i in range(len(chunks))]
        actual_ids = [chunk_id for chunk_id, _ in chunks]
        self.assertEqual(actual_ids, expected_ids)


class TestPreprocess(absltest.TestCase):
    """Tests for preprocess function."""

    def test_preprocess_extracts_fields(self):
        sample = {
            "id": "DOC123",
            "content": "Legal content here",
            "title": "Case Title",
            "decision_date": "2024-01-15",
            "juridiction": "Court",
            "formation": "Formation",
            "solution": "Solution",
            "applied_law": "Law",
        }
        result = preprocess(sample)
        self.assertEqual(result["doc_id"], "DOC123")
        self.assertEqual(result["content"], "Legal content here")
        self.assertEqual(result["title"], "Case Title")
        self.assertEqual(result["decision_date"], "2024-01-15")

    def test_preprocess_handles_none_values(self):
        sample = {
            "id": "DOC456",
            "content": None,
            "title": None,
            "decision_date": None,
        }
        result = preprocess(sample)
        self.assertEqual(result["doc_id"], "DOC456")
        self.assertEqual(result["content"], "")
        self.assertIsNone(result["title"])
        self.assertIsNone(result["decision_date"])

    def test_preprocess_with_custom_doc_id_column(self):
        sample = {
            "custom_id": "CUSTOM123",
            "content": "Content",
            "title": "Title",
            "decision_date": "2024",
        }
        result = preprocess(sample, doc_id_column="custom_id")
        self.assertEqual(result["doc_id"], "CUSTOM123")


if __name__ == "__main__":
    absltest.main()
