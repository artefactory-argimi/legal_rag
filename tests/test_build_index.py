"""Unit tests for build_index functionality using synthetic data."""

import os
import shutil
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from absl.testing import absltest
from sqlitedict import SqliteDict

from legal_rag.indexer import ScriptConfig, build_index


def create_synthetic_dataset(num_docs: int, content_length: int = 500) -> list[dict]:
    """Create a synthetic dataset for testing.

    Args:
        num_docs: Number of documents to generate.
        content_length: Approximate length of content per document.

    Returns:
        List of document dictionaries.
    """
    base_sentence = "This is a legal sentence about constitutional law. "
    repeats = max(1, content_length // len(base_sentence))

    return [
        {
            "id": f"TESTDOC{i:06d}",
            "content": base_sentence * repeats,
            "title": f"Test Document Title {i}",
            "decision_date": f"2024-01-{(i % 28) + 1:02d}",
        }
        for i in range(num_docs)
    ]


class TestBuildIndexIntegration(absltest.TestCase):
    """Integration tests for build_index using synthetic data."""

    def setUp(self):
        super().setUp()
        self.tmpdir = Path(tempfile.mkdtemp(prefix="build_index_test_"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        super().tearDown()

    def test_build_index_creates_sqlite_mappings(self):
        """Test that build_index creates SQLite mappings for all chunks."""
        from unittest import mock

        synthetic_data = create_synthetic_dataset(num_docs=50, content_length=300)

        with mock.patch("legal_rag.indexer.load_dataset") as mock_load:
            mock_load.return_value = synthetic_data

            cfg = ScriptConfig(
                index_folder=self.tmpdir,
                index_name="sqlite_test",
                chunker_type="sentence",
                chunk_tokenizer="character",
                chunk_size=200,
                accumulation_size=500,
            )

            build_index(cfg)

        index_dir = self.tmpdir / "sqlite_test"
        self.assertTrue(index_dir.exists())
        self.assertTrue((index_dir / "fast_plaid_index").exists())

        docid_sqlite = index_dir / "documents_ids_to_plaid_ids.sqlite"
        self.assertTrue(docid_sqlite.exists())

        with SqliteDict(str(docid_sqlite), outer_stack=False) as db:
            self.assertGreater(len(db), 0)
            self.assertIn("TESTDOC000000-0", db)

    def test_build_index_skips_existing(self):
        """Test that build_index skips if index exists and force=False."""
        index_dir = self.tmpdir / "existing_index"
        index_dir.mkdir(parents=True)

        cfg = ScriptConfig(
            index_folder=self.tmpdir,
            index_name="existing_index",
            force=False,
        )

        build_index(cfg)

        self.assertTrue(index_dir.exists())
        self.assertFalse((index_dir / "fast_plaid_index").exists())

    def test_build_index_force_rebuilds(self):
        """Test that build_index rebuilds when force=True."""
        from unittest import mock

        index_dir = self.tmpdir / "force_rebuild"
        index_dir.mkdir(parents=True)

        synthetic_data = create_synthetic_dataset(num_docs=30, content_length=200)

        with mock.patch("legal_rag.indexer.load_dataset") as mock_load:
            mock_load.return_value = synthetic_data

            cfg = ScriptConfig(
                index_folder=self.tmpdir,
                index_name="force_rebuild",
                force=True,
                chunker_type="sentence",
                chunk_tokenizer="character",
                chunk_size=150,
                accumulation_size=200,
            )

            build_index(cfg)

        self.assertTrue((index_dir / "fast_plaid_index").exists())

        docid_sqlite = index_dir / "documents_ids_to_plaid_ids.sqlite"
        with SqliteDict(str(docid_sqlite), outer_stack=False) as db:
            self.assertGreater(len(db), 0)

    def test_build_index_with_sentence_chunker(self):
        """Test build_index with SentenceChunker produces multiple chunks."""
        from unittest import mock

        long_content = "This is a legal sentence about law. " * 100
        synthetic_data = [
            {
                "id": f"LONGDOC{i:03d}",
                "content": long_content,
                "title": f"Long Document {i}",
                "decision_date": "2024-01-01",
            }
            for i in range(20)
        ]

        with mock.patch("legal_rag.indexer.load_dataset") as mock_load:
            mock_load.return_value = synthetic_data

            cfg = ScriptConfig(
                index_folder=self.tmpdir,
                index_name="sentence_chunker_test",
                chunker_type="sentence",
                chunk_tokenizer="character",
                chunk_size=200,
                accumulation_size=500,
            )

            build_index(cfg)

        index_dir = self.tmpdir / "sentence_chunker_test"
        docid_sqlite = index_dir / "documents_ids_to_plaid_ids.sqlite"

        with SqliteDict(str(docid_sqlite), outer_stack=False) as db:
            self.assertGreater(len(db), 20)
            self.assertIn("LONGDOC000-0", db)
            self.assertIn("LONGDOC000-1", db)

    def test_build_index_chunk_ids_are_correct_format(self):
        """Test that chunk IDs follow the expected format: docid-chunkidx."""
        from unittest import mock

        synthetic_data = create_synthetic_dataset(num_docs=25, content_length=400)

        with mock.patch("legal_rag.indexer.load_dataset") as mock_load:
            mock_load.return_value = synthetic_data

            cfg = ScriptConfig(
                index_folder=self.tmpdir,
                index_name="chunk_format_test",
                chunker_type="sentence",
                chunk_tokenizer="character",
                chunk_size=150,
                accumulation_size=300,
            )

            build_index(cfg)

        index_dir = self.tmpdir / "chunk_format_test"
        docid_sqlite = index_dir / "documents_ids_to_plaid_ids.sqlite"

        with SqliteDict(str(docid_sqlite), outer_stack=False) as db:
            for chunk_id in db.keys():
                parts = chunk_id.rsplit("-", 1)
                self.assertEqual(len(parts), 2)
                self.assertTrue(parts[1].isdigit())


class TestBuildIndexBatching(absltest.TestCase):
    """Tests for batching behavior in build_index."""

    def setUp(self):
        super().setUp()
        self.tmpdir = Path(tempfile.mkdtemp(prefix="build_index_batch_test_"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        super().tearDown()

    def test_small_accumulation_size_produces_multiple_batches(self):
        """Test that a small accumulation_size causes multiple index updates."""
        from unittest import mock

        synthetic_data = create_synthetic_dataset(num_docs=40, content_length=300)

        with mock.patch("legal_rag.indexer.load_dataset") as mock_load:
            mock_load.return_value = synthetic_data

            cfg = ScriptConfig(
                index_folder=self.tmpdir,
                index_name="batch_size_test",
                chunker_type="sentence",
                chunk_tokenizer="character",
                chunk_size=100,
                accumulation_size=50,
            )

            build_index(cfg)

        index_dir = self.tmpdir / "batch_size_test"
        docid_sqlite = index_dir / "documents_ids_to_plaid_ids.sqlite"

        with SqliteDict(str(docid_sqlite), outer_stack=False) as db:
            self.assertGreaterEqual(len(db), 40)


if __name__ == "__main__":
    absltest.main()
