from pathlib import Path

from absl.testing import absltest

from legal_rag.assets import prepare_assets


class IndexPathResolutionTest(absltest.TestCase):
    def test_prepare_assets_raises_on_nested_index(self):
        tmp = Path(self.create_tempdir().full_path)
        encoder_dest = tmp / "enc"
        index_dest = tmp / "idx"

        # Build a dummy encoder zip (minimal config file).
        enc_zip = tmp / "enc.zip"
        with open(enc_zip, "wb") as f:
            import zipfile

            with zipfile.ZipFile(f, "w") as zf:
                zf.writestr("config.json", "{}")

        # Build an index with an extra nesting level (fast_plaid_index too deep).
        nested = index_dest / "wrong_nested" / "deep" / "fast_plaid_index"
        nested.mkdir(parents=True, exist_ok=True)
        (nested.parent / "doc_mapping.json").write_text("{}", encoding="utf-8")

        with self.assertRaises(ValueError):
            prepare_assets(
                encoder_zip_uri=str(enc_zip),
                index_zip_uri=index_dest,
                encoder_dest=encoder_dest,
                index_dest=index_dest,
            )

    def test_prepare_assets_accepts_flat_index(self):
        tmp = Path(self.create_tempdir().full_path)
        encoder_dest = tmp / "enc"
        index_dest = tmp / "idx"

        enc_zip = tmp / "enc.zip"
        with open(enc_zip, "wb") as f:
            import zipfile

            with zipfile.ZipFile(f, "w") as zf:
                zf.writestr("config.json", "{}")

        flat = index_dest / "my_custom_index_name" / "fast_plaid_index"
        flat.mkdir(parents=True, exist_ok=True)
        (flat.parent / "doc_mapping.json").write_text("{}", encoding="utf-8")

        encoder_path, index_dir = prepare_assets(
            encoder_zip_uri=str(enc_zip),
            index_zip_uri=index_dest,
            encoder_dest=encoder_dest,
            index_dest=index_dest,
        )

        self.assertTrue(Path(encoder_path).exists())
        self.assertEqual(index_dir, flat.parent)


if __name__ == "__main__":
    absltest.main()
