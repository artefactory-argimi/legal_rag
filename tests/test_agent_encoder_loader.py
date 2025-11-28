from pathlib import Path
import tempfile
import zipfile
from urllib.error import URLError

from absl.testing import absltest

from legal_rag.agent import build_retrieval
from legal_rag.assets import prepare_assets


ENCODER_ZIP_URL = "https://github.com/artefactory-argimi/legal_rag/releases/download/data-juri-v1/colbert-encoder.zip"
INDEX_ZIP_URL = "https://github.com/artefactory-argimi/legal_rag/releases/download/data-juri-v1/index.zip"


class EncoderLoadTest(absltest.TestCase):
    def test_encoder_and_index_load_and_encode(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            try:
                encoder_path_str, index_dir = prepare_assets(
                    encoder_zip_uri=ENCODER_ZIP_URL,
                    index_zip_uri=INDEX_ZIP_URL,
                    encoder_dest=tmp_path / "encoder",
                    index_dest=tmp_path / "index",
                )
                encoder_dir = Path(encoder_path_str)
            except URLError:
                self.skipTest("Network unavailable to download assets")

            self.assertTrue(
                (encoder_dir / "config.json").exists() or (encoder_dir / "config_sentence_transformers.json").exists(),
                "Encoder config not found after download",
            )
            self.assertTrue(index_dir.exists(), "Index directory missing after download")
            self.assertTrue(any(index_dir.iterdir()), "Index directory is empty")

            encoder, retriever = build_retrieval(
                encoder_model=str(encoder_dir),
                index_folder=index_dir,
                index_name="legal_french_index",
            )

            tokenizer = getattr(encoder, "tokenizer", None)
            if tokenizer is not None and hasattr(tokenizer, "is_fast"):
                self.assertFalse(tokenizer.is_fast, "Expected slow tokenizer to avoid conversion errors")

            # Ensure embedding table is large enough for tokenizer vocab.
            embedding_size = encoder[0].auto_model.get_input_embeddings().num_embeddings
            if tokenizer is not None:
                self.assertGreaterEqual(
                    embedding_size, len(tokenizer), "Embedding size should cover tokenizer vocab"
                )

            # Smoke test encoding.
            query_emb = encoder.encode("test query", is_query=True, show_progress_bar=False)
            self.assertTrue(hasattr(query_emb, "shape") and query_emb.shape[0] > 0, "Expected non-empty query embedding")


if __name__ == "__main__":
    absltest.main()
