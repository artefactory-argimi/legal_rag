"""Tests for tools.py lookup functionality with chunk IDs."""

from absl.testing import absltest

from legal_rag.tools import get_row_by_id, lookup_legal_doc, parse_chunk_id


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


if __name__ == "__main__":
    absltest.main()
