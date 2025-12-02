import subprocess
from pathlib import Path
import tempfile

from absl.testing import absltest

from legal_rag.assets import prepare_assets
from legal_rag.retriever import build_encoder, build_retriever
from legal_rag.retriever import build_encoder, build_retriever


class EncoderLoadTest(absltest.TestCase):
    def test_encoder_and_index_load_and_encode(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            encoder_path_str, index_dir = prepare_assets(
                encoder_zip_uri=str(Path("assets/colbert-encoder.zip")),
                index_zip_uri=Path("assets/index_legal_constit"),
                encoder_dest=tmp_path / "encoder",
                index_dest=tmp_path / "index",
            )
            encoder_dir = Path(encoder_path_str)

            self.assertTrue(
                (encoder_dir / "config.json").exists() or (encoder_dir / "config_sentence_transformers.json").exists(),
                "Encoder config not found",
            )
            self.assertTrue((index_dir / "fast_plaid_index").exists(), "fast_plaid_index missing in index directory")

            script = f"""
import sys, json
from legal_rag.retriever import build_encoder, build_retriever
from legal_rag.tools import search_legal_docs
from etils import epath
enc = r\"{encoder_dir}\"
idx = epath.Path(r\"{index_dir}\")
print("[test] encoder", enc)
print("[test] index", idx)
encoder = build_encoder(encoder_model=enc)
retriever = build_retriever(index_folder=idx, index_name="legal_french_index")
print("[test] index is_indexed", getattr(retriever.index._index, "is_indexed", None))
res = search_legal_docs(query="test", encoder=encoder, retriever=retriever, index_folder=idx, k=1)
print("[test] search results", json.dumps(res))
"""
            proc = subprocess.run(
                ["python", "-c", script],
                text=True,
                capture_output=True,
            )
            if proc.returncode != 0:
                self.fail(f"Subprocess failed with code {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")


if __name__ == "__main__":
    absltest.main()
