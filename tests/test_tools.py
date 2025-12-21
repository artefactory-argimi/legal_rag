"""Tests for tools.py lookup functionality with chunk IDs."""

import os

# Force CPU-only mode and enable MPS fallback for tests
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from absl.testing import absltest

from legal_rag.chunking import DocumentChunkCache
from legal_rag.tools import (
    get_row_by_id,
    lookup_chunk,
    lookup_legal_doc,
    parse_chunk_id,
)


class TestParseChunkId(absltest.TestCase):
    """Tests for parse_chunk_id function."""

    def test_parse_basic_chunk_id(self):
        doc_id, chunk_idx = parse_chunk_id("DOC123-0")
        self.assertEqual(doc_id, "DOC123")
        self.assertEqual(chunk_idx, 0)

    def test_parse_multi_digit_index(self):
        doc_id, chunk_idx = parse_chunk_id("DOC123-42")
        self.assertEqual(doc_id, "DOC123")
        self.assertEqual(chunk_idx, 42)


class TestGetRowById(absltest.TestCase):
    """Tests for get_row_by_id function."""

    def test_get_existing_row(self):
        dataset = [
            {"id": "DOC1", "title": "First"},
            {"id": "DOC2", "title": "Second"},
        ]
        row = get_row_by_id(dataset, "DOC2")
        self.assertEqual(row["title"], "Second")

    def test_get_nonexistent_row_raises(self):
        dataset = [{"id": "DOC1", "title": "First"}]
        with self.assertRaises(KeyError):
            get_row_by_id(dataset, "NONEXISTENT")


class TestLookupLegalDoc(absltest.TestCase):
    """Tests for lookup_legal_doc with chunk IDs."""

    def test_lookup_extracts_parent_doc_id(self):
        dataset = [
            {
                "id": "DOC123",
                "title": "Test Title",
                "content": "Document content",
                "decision_date": "2024-01-01",
                "juridiction": "Court",
                "formation": "Formation",
                "solution": "Solution",
                "applied_law": "Law",
            }
        ]
        result = lookup_legal_doc("DOC123-5", dataset=dataset)
        self.assertIn("Test Title", result)
        self.assertIn("Document content", result)

    def test_lookup_with_first_chunk(self):
        dataset = [
            {
                "id": "ABC",
                "title": "ABC Title",
                "content": "ABC content",
                "decision_date": "2024",
                "juridiction": "J",
                "formation": "F",
                "solution": "S",
                "applied_law": "L",
            }
        ]
        result = lookup_legal_doc("ABC-0", dataset=dataset)
        self.assertIn("ABC Title", result)

    def test_lookup_not_found(self):
        dataset = [{"id": "OTHER", "title": "Other"}]
        result = lookup_legal_doc("NOTFOUND-0", dataset=dataset)
        self.assertIn("not found", result)

    def test_lookup_with_score(self):
        dataset = [
            {
                "id": "DOC1",
                "title": "Title",
                "content": "Content",
                "decision_date": "2024",
                "juridiction": "J",
                "formation": "F",
                "solution": "S",
                "applied_law": "L",
            }
        ]
        result = lookup_legal_doc("DOC1-0", dataset=dataset, score=0.95)
        self.assertIn("0.95", result)


class TestLookupChunk(absltest.TestCase):
    """Tests for lookup_chunk function with DocumentChunkCache."""

    def setUp(self):
        super().setUp()
        self.dataset = [
            {
                "id": "DOC1",
                "content": "First sentence of document. Second sentence here. Third sentence.",
                "title": "Test Document",
                "decision_date": "2024-01-15",
                "juridiction": "Court",
                "formation": "Formation",
                "solution": "Solution",
                "applied_law": "Law",
            },
            {
                "id": "DOC2",
                "content": "Another document with more content. " * 100,
                "title": "Long Document",
                "decision_date": "2024-02-20",
                "juridiction": "J",
                "formation": "F",
                "solution": "S",
                "applied_law": "L",
            },
        ]
        self.cache = DocumentChunkCache(
            dataset=self.dataset,
            doc_id_column="id",
        )

    def test_lookup_chunk_returns_content(self):
        result = lookup_chunk("DOC1-0", chunk_cache=self.cache)
        self.assertIn("Test Document", result)
        self.assertIn("Document: DOC1", result)

    def test_lookup_chunk_with_context(self):
        result = lookup_chunk(
            "DOC1-0",
            chunk_cache=self.cache,
            include_context=True,
            context_chunks=1,
        )
        self.assertIn(">>>", result)

    def test_lookup_chunk_without_context(self):
        result = lookup_chunk(
            "DOC1-0",
            chunk_cache=self.cache,
            include_context=False,
        )
        self.assertIn("[Chunk 0]", result)
        self.assertNotIn(">>>", result)

    def test_lookup_chunk_not_found_doc(self):
        result = lookup_chunk("NONEXISTENT-0", chunk_cache=self.cache)
        self.assertIn("not found", result)

    def test_lookup_chunk_invalid_index(self):
        result = lookup_chunk("DOC1-999", chunk_cache=self.cache)
        self.assertIn("not found", result)

    def test_lookup_chunk_includes_metadata_in_header(self):
        result = lookup_chunk("DOC1-0", chunk_cache=self.cache)
        self.assertIn("Titre: Test Document", result)
        self.assertIn("Date: 2024-01-15", result)

    def test_lookup_chunk_context_shows_surrounding_chunks(self):
        # DOC2 has long content that produces multiple chunks
        chunks = self.cache.get_chunks("DOC2")
        if len(chunks) >= 3:
            # Get middle chunk with context
            result = lookup_chunk(
                "DOC2-1",
                chunk_cache=self.cache,
                include_context=True,
                context_chunks=1,
            )
            # Should have chunk 0, 1, and 2
            self.assertIn("[Chunk 0]", result)
            self.assertIn("[Chunk 1]", result)
            self.assertIn("[Chunk 2]", result)
            # Chunk 1 should be marked with >>>
            self.assertIn(">>> [Chunk 1]", result)

    def test_lookup_chunk_reuses_cached_document(self):
        # First lookup
        lookup_chunk("DOC1-0", chunk_cache=self.cache)
        stats1 = self.cache.stats()

        # Second lookup of different chunk from same doc
        lookup_chunk("DOC1-0", chunk_cache=self.cache)
        stats2 = self.cache.stats()

        # Should not have added another doc to cache
        self.assertEqual(stats1["num_docs"], stats2["num_docs"])


class TestLookupLegalDocWithCache(absltest.TestCase):
    """Tests for lookup_legal_doc with DocumentChunkCache."""

    def setUp(self):
        super().setUp()
        self.dataset = [
            {
                "id": "DOC1",
                "content": "Document content here.",
                "title": "Test Title",
                "decision_date": "2024-01-01",
                "juridiction": "Court",
                "formation": "Formation",
                "solution": "Solution",
                "applied_law": "Law",
            }
        ]
        self.cache = DocumentChunkCache(
            dataset=self.dataset,
            doc_id_column="id",
        )

    def test_lookup_with_cache(self):
        result = lookup_legal_doc("DOC1-0", chunk_cache=self.cache)
        self.assertIn("Test Title", result)
        self.assertIn("Document content here", result)

    def test_lookup_with_cache_not_found(self):
        result = lookup_legal_doc("NONEXISTENT-0", chunk_cache=self.cache)
        self.assertIn("not found", result)


if __name__ == "__main__":
    absltest.main()
