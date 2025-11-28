from pathlib import Path
import tempfile
import zipfile
from urllib.request import urlretrieve

from absl.testing import absltest

from legal_rag.agent import build_retrieval


ENCODER_ZIP_URL = "https://github.com/artefactory-argimi/legal_rag/releases/download/data-juri-v1/colbert-encoder.zip"
INDEX_ZIP_URL = "https://github.com/artefactory-argimi/legal_rag/releases/download/data-juri-v1/index.zip"


def _fetch_and_extract(url: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    local_zip = dest / Path(url).name
    urlretrieve(url, local_zip)
    with zipfile.ZipFile(local_zip, "r") as zf:
        zf.extractall(dest)
    # Prefer a directory that has a config; otherwise return dest.
    for cand in [dest] + [p for p in dest.iterdir() if p.is_dir()]:
        if (cand / "config.json").exists() or (cand / "config_sentence_transformers.json").exists():
            return cand
    return dest


class EncoderLoadTest(absltest.TestCase):
    def test_encoder_and_index_load_and_encode(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            encoder_dir = _fetch_and_extract(ENCODER_ZIP_URL, tmp_path / "encoder")
            index_dir = _fetch_and_extract(INDEX_ZIP_URL, tmp_path / "index")

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
