from pathlib import Path
import tempfile
import zipfile

from absl.testing import absltest

from legal_rag.assets import prepare_assets


class AssetsTest(absltest.TestCase):
    def test_prepare_assets_uses_existing_index_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            encoder_zip = tmp_path / "encoder.zip"
            with zipfile.ZipFile(encoder_zip, "w") as zf:
                zf.writestr("config.json", "{}")

            index_dir = tmp_path / "prebuilt_index"
            index_dir.mkdir()
            (index_dir / "manifest.txt").write_text("ready")

            encoder_dest = tmp_path / "encoder_extracted"
            index_dest = tmp_path / "index_extracted"

            encoder_path, resolved_index = prepare_assets(
                encoder_zip_uri=str(encoder_zip),
                index_zip_uri=str(index_dir),
                encoder_dest=encoder_dest,
                index_dest=index_dest,
            )

            self.assertEqual(resolved_index, index_dir)
            self.assertTrue(Path(encoder_path).is_dir())
            self.assertTrue((Path(encoder_path) / "config.json").exists())


if __name__ == "__main__":
    absltest.main()
