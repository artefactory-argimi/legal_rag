import shutil
import tempfile
from pathlib import Path

from absl.testing import absltest
import numpy as np

from pylate import indexes


class IndexCreationTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.tmpdir = Path(tempfile.mkdtemp(prefix="plaid_index_test_"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        super().tearDown()

    def test_index_created_and_reloaded(self):
        index_name = "simple"
        # FastPlaid expects ColBERT-sized token embeddings (dim ~128); keep small batch.
        embed_dim = 128
        doc_embeddings = np.random.rand(3, 4, embed_dim).astype(np.float32)
        doc_ids = [f"doc-{i}" for i in range(len(doc_embeddings))]

        # Create and populate the index.
        index = indexes.PLAID(
            index_folder=self.tmpdir,
            index_name=index_name,
            override=True,
            show_progress=False,
            batch_size=128,
            n_full_scores=32,
        )
        index.add_documents(documents_ids=doc_ids, documents_embeddings=doc_embeddings)

        # Sanity-check the on-disk artifacts were created.
        index_dir = self.tmpdir / index_name
        self.assertTrue((index_dir / "fast_plaid_index").exists())
        docid_sqlite = index_dir / "documents_ids_to_plaid_ids.sqlite"
        self.assertTrue(docid_sqlite.exists())

        # Reload the on-disk index and check metadata/mappings without performing a search
        # (search spawns worker processes that are disallowed in this test environment).
        reloaded = indexes.PLAID(
            index_folder=self.tmpdir,
            index_name=index_name,
            override=False,
            show_progress=False,
        )
        # FastPlaid sets an internal flag when data is present; ensure it's true.
        self.assertTrue(getattr(reloaded._index, "is_indexed", False))

        # Verify the sqlite mapping contains the inserted ids.
        from sqlitedict import SqliteDict

        with SqliteDict(docid_sqlite, outer_stack=False) as mapping:
            self.assertGreater(len(mapping), 0)
            for doc_id in doc_ids:
                self.assertIn(doc_id, mapping)


if __name__ == "__main__":
    absltest.main()
