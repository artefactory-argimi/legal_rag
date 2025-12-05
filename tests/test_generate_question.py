"""Offline tests for question generation with span extraction."""

from pathlib import Path
from unittest import mock

from absl.testing import absltest
import datasets
import dspy

from scripts.generate_question import GenerationConfig, main


class GenerateQuestionTest(absltest.TestCase):
    def test_generates_dataset_and_propagates_ids(self):
        tmp_dir = Path(self.create_tempdir().full_path)

        dataset = datasets.Dataset.from_dict(
            {"id": ["plaid-1"], "content": ["ABCDEFGHIJKLMNO"]}
        )
        cfg = GenerationConfig(
            dataset="dummy_dataset",
            config="dummy_config",
            split="train",
            max_context_chars=10,
            output_dir=tmp_dir / "out_ds",
            overwrite=True,
        )

        class FakeAgent:
            def __init__(self):
                self.calls: list[str] = []

            def __call__(self, *, context: str):
                self.calls.append(context)
                return dspy.Prediction(
                    question="Doc question",
                    answer="ABCDE",
                )

        fake_agent = FakeAgent()

        with mock.patch("scripts.generate_question.load_dataset", return_value=dataset), \
            mock.patch("scripts.generate_question.QAAgent", return_value=fake_agent), \
            mock.patch("scripts.generate_question.dspy.LM", return_value=None), \
            mock.patch("scripts.generate_question.dspy.configure"):
            main(cfg)

        saved = datasets.load_from_disk(str(cfg.output_dir))
        self.assertLen(saved, 1)
        row = saved[0]
        self.assertEqual(row["doc_id"], "plaid-1")
        self.assertIsNotNone(row["answer_span_start"])
        self.assertIsNotNone(row["answer_span_end"])
        self.assertEqual(fake_agent.calls[0], row["context_excerpt"])
        self.assertGreaterEqual(len(fake_agent.calls), 1)


if __name__ == "__main__":
    absltest.main()
