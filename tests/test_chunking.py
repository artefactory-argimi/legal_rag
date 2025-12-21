"""Unit tests for document chunking functionality."""

import os

# Force CPU-only mode and enable MPS fallback for tests
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from absl.testing import absltest
from chonkie import SemanticChunker, SentenceChunker

from legal_rag.chunking import (
    Chunk,
    ChunkConfig,
    DocumentChunkCache,
    build_chunker,
    chunk_document,
)
from legal_rag.indexer import preprocess
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
        self.assertIsInstance(chunks[0], Chunk)
        self.assertEqual(chunks[0].chunk_id, "DOC1-0")
        self.assertIn("Title", chunks[0].text)

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
        for idx, chunk in enumerate(chunks):
            self.assertEqual(chunk.chunk_id, f"DOC2-{idx}")
        self.assertIn("Title", chunks[0].text)
        if len(chunks) > 1:
            self.assertNotIn("Title", chunks[1].text)

    def test_empty_document(self):
        chunks = chunk_document(
            content="",
            doc_id="EMPTY",
            title=None,
            decision_date=None,
            chunker=self.chunker,
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_id, "EMPTY-0")

    def test_long_content_produces_multiple_chunks(self):
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
        self.assertIn("Test Title", chunks[0].text)
        self.assertIn("2024-05-15", chunks[0].text)

    def test_no_header_when_no_metadata(self):
        chunks = chunk_document(
            content="Some legal text content.",
            doc_id="DOC4",
            title=None,
            decision_date=None,
            chunker=self.chunker,
        )
        self.assertNotIn("Titre:", chunks[0].text)

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
        actual_ids = [chunk.chunk_id for chunk in chunks]
        self.assertEqual(actual_ids, expected_ids)

    def test_chunk_has_correct_doc_id(self):
        chunks = chunk_document(
            content="Some text.",
            doc_id="DOC5",
            title=None,
            decision_date=None,
            chunker=self.chunker,
        )
        for chunk in chunks:
            self.assertEqual(chunk.doc_id, "DOC5")


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
        self.assertIsInstance(chunks[0], Chunk)
        self.assertEqual(chunks[0].chunk_id, "DOC1-0")
        self.assertIn("Title", chunks[0].text)

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
        for idx, chunk in enumerate(chunks):
            self.assertEqual(chunk.chunk_id, f"DOC2-{idx}")
        self.assertIn("Title", chunks[0].text)
        if len(chunks) > 1:
            self.assertNotIn("Title", chunks[1].text)

    def test_empty_document(self):
        chunks = chunk_document(
            content="",
            doc_id="EMPTY",
            title=None,
            decision_date=None,
            chunker=self.chunker,
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_id, "EMPTY-0")

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
        self.assertIn("Test Title", chunks[0].text)
        self.assertIn("2024-05-15", chunks[0].text)

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
        actual_ids = [chunk.chunk_id for chunk in chunks]
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


class TestBuildChunker(absltest.TestCase):
    """Tests for build_chunker factory function."""

    def test_build_semantic_chunker(self):
        config = ChunkConfig(
            chunker_type="semantic",
            chunk_tokenizer="minishlab/potion-base-8M",
            chunk_size=256,
        )
        chunker = build_chunker(config)
        self.assertIsInstance(chunker, SemanticChunker)

    def test_build_sentence_chunker(self):
        config = ChunkConfig(
            chunker_type="sentence",
            chunk_tokenizer="character",
            chunk_size=256,
        )
        chunker = build_chunker(config)
        self.assertIsInstance(chunker, SentenceChunker)


class TestDocumentChunkCache(absltest.TestCase):
    """Tests for DocumentChunkCache class."""

    def setUp(self):
        super().setUp()
        # Create a simple mock dataset as a list of dicts
        self.dataset = [
            {
                "id": "DOC1",
                "content": "This is the first document. It has some content.",
                "title": "Document One",
                "decision_date": "2024-01-01",
                "juridiction": "Court A",
                "formation": "F1",
                "solution": "S1",
                "applied_law": "L1",
            },
            {
                "id": "DOC2",
                "content": "This is the second document. " * 100,
                "title": "Document Two",
                "decision_date": "2024-02-02",
                "juridiction": "Court B",
                "formation": "F2",
                "solution": "S2",
                "applied_law": "L2",
            },
        ]
        self.cache = DocumentChunkCache(
            dataset=self.dataset,
            doc_id_column="id",
        )

    def test_get_chunks_returns_chunks(self):
        chunks = self.cache.get_chunks("DOC1")
        self.assertIsNotNone(chunks)
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], Chunk)

    def test_get_chunks_caches_result(self):
        self.cache.get_chunks("DOC1")
        stats1 = self.cache.stats()
        self.assertEqual(stats1["num_docs"], 1)

        self.cache.get_chunks("DOC1")
        stats2 = self.cache.stats()
        self.assertEqual(stats2["num_docs"], 1)

    def test_get_chunks_returns_none_for_missing_doc(self):
        chunks = self.cache.get_chunks("NONEXISTENT")
        self.assertIsNone(chunks)

    def test_get_chunk_by_id(self):
        chunk = self.cache.get_chunk("DOC1-0")
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk.chunk_id, "DOC1-0")
        self.assertEqual(chunk.doc_id, "DOC1")

    def test_get_chunk_returns_none_for_invalid_id(self):
        chunk = self.cache.get_chunk("INVALID")
        self.assertIsNone(chunk)

    def test_get_chunk_returns_none_for_out_of_range_index(self):
        chunk = self.cache.get_chunk("DOC1-999")
        self.assertIsNone(chunk)

    def test_get_chunk_text(self):
        text = self.cache.get_chunk_text("DOC1-0")
        self.assertIsNotNone(text)
        self.assertIsInstance(text, str)

    def test_get_document_metadata(self):
        metadata = self.cache.get_document_metadata("DOC1")
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata["title"], "Document One")
        self.assertEqual(metadata["decision_date"], "2024-01-01")

    def test_get_document_metadata_returns_none_for_missing(self):
        metadata = self.cache.get_document_metadata("NONEXISTENT")
        self.assertIsNone(metadata)

    def test_clear_cache(self):
        self.cache.get_chunks("DOC1")
        self.cache.get_chunks("DOC2")
        self.assertEqual(self.cache.stats()["num_docs"], 2)

        self.cache.clear()
        self.assertEqual(self.cache.stats()["num_docs"], 0)

    def test_stats(self):
        self.cache.get_chunks("DOC1")
        stats = self.cache.stats()
        self.assertIn("num_docs", stats)
        self.assertIn("num_chunks", stats)
        self.assertEqual(stats["num_docs"], 1)

    def test_chunks_have_correct_structure(self):
        chunks = self.cache.get_chunks("DOC1")
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.doc_id, "DOC1")
            self.assertEqual(chunk.chunk_idx, i)
            self.assertEqual(chunk.chunk_id, f"DOC1-{i}")
            self.assertIsInstance(chunk.text, str)

    def test_multiple_docs_cached_independently(self):
        self.cache.get_chunks("DOC1")
        self.cache.get_chunks("DOC2")

        stats = self.cache.stats()
        self.assertEqual(stats["num_docs"], 2)

        # Verify chunks are different
        chunks1 = self.cache.get_chunks("DOC1")
        chunks2 = self.cache.get_chunks("DOC2")
        self.assertNotEqual(chunks1[0].text, chunks2[0].text)

    def test_long_document_produces_multiple_chunks(self):
        chunks = self.cache.get_chunks("DOC2")
        self.assertGreater(len(chunks), 1)

    def test_chunk_text_matches_get_chunk(self):
        chunks = self.cache.get_chunks("DOC1")
        for chunk in chunks:
            retrieved = self.cache.get_chunk(chunk.chunk_id)
            self.assertEqual(chunk.text, retrieved.text)

    def test_first_chunk_contains_header_with_title(self):
        chunks = self.cache.get_chunks("DOC1")
        first_chunk = chunks[0]
        self.assertIn("Document One", first_chunk.text)
        self.assertIn("2024-01-01", first_chunk.text)


if __name__ == "__main__":
    absltest.main()
