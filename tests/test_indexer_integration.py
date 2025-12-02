from pathlib import Path

from absl.testing import absltest
from sqlitedict import SqliteDict
from pylate import indexes

from legal_rag.assets import prepare_assets


class IndexerIntegrationTest(absltest.TestCase):
    def test_prepare_assets_with_local_random_index(self):
        tmp_root = Path(self.create_tempdir().full_path)
        encoder_dest = tmp_root / "encoder"
        index_root = tmp_root / "local_index"

        # Build a tiny PLAID index with random vectors on disk.
        num_docs = 3
        embed_dim = 64
        tokens = 4
        doc_embeddings = np.random.rand(num_docs, tokens, embed_dim).astype(np.float32)
        doc_ids = [f"doc-{i}" for i in range(num_docs)]

        plaid_index = indexes.PLAID(
            index_folder=index_root,
            index_name="legal_french_index",
            override=True,
            show_progress=False,
            batch_size=128,
            n_full_scores=32,
        )
        plaid_index.add_documents(documents_ids=doc_ids, documents_embeddings=doc_embeddings)

        # Use prepare_assets to load the locally created index (no fake mode).
        encoder_zip = Path("assets/colbert-encoder.zip")
        encoder_path_str, index_dir = prepare_assets(
            encoder_zip_uri=str(encoder_zip),
            index_zip_uri=index_root,
            encoder_dest=encoder_dest,
            index_dest=tmp_root / "index_copy",
        )

        encoder_path = Path(encoder_path_str)
        self.assertTrue((encoder_path / "config.json").exists() or (encoder_path / "config_sentence_transformers.json").exists())
        self.assertTrue(index_dir.exists())
        self.assertTrue((index_dir / "fast_plaid_index").exists())

        # The PLAID sqlite mappings should exist and contain entries.
        docid_sqlite = index_dir / "documents_ids_to_plaid_ids.sqlite"
        self.assertTrue(docid_sqlite.exists())
        with SqliteDict(docid_sqlite, outer_stack=False) as db:
            self.assertGreaterEqual(len(db), 3)

        # Build retrieval stack on the local assets and ensure the index reports data.
        reloaded = indexes.PLAID(
            index_folder=index_dir.parent,
            index_name="legal_french_index",
            override=False,
            show_progress=False,
        )
        self.assertTrue(getattr(reloaded._index, "is_indexed", False))


if __name__ == "__main__":
    absltest.main()
