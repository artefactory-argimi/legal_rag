#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "absl-py>=2.1.0",
#     "datasets>=3.2.0",
#     "dspy>=3.0.4",
#     "etils[eapp]>=1.9.0",
#     "importlib_resources>=6.4.0",
# ]
# ///

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable

from absl import app, logging
import datasets
from datasets import load_dataset
from etils import eapp, epath

import dspy

DEFAULT_DATASET = "artefactory/Argimi-Legal-French-Jurisprudence"
DEFAULT_CONFIG = "juri"
DEFAULT_SPLIT = "train"
DEFAULT_COLUMN = "content"
DEFAULT_MODEL = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
DEFAULT_LOCAL_API_BASE = "http://localhost:8000/v1"


@dataclass(frozen=True)
class QuestionGenConfig:
    dataset: str = DEFAULT_DATASET
    config: str = DEFAULT_CONFIG
    split: str = DEFAULT_SPLIT
    column: str = DEFAULT_COLUMN
    model: str = DEFAULT_MODEL
    api_key: str | None = None  # Optional; local sglang default does not require a key.
    api_base: str | None = DEFAULT_LOCAL_API_BASE  # OpenAI-compatible base URL; defaults to local sglang.
    temperature: float = 0.2
    max_tokens: int = 128
    sample_size: int = 10
    seed: int | None = 0
    output: epath.Path = epath.Path("./artifacts/generated_questions.parquet")


class GenerateQuestion(dspy.Module):
    """Small DSPy module that turns a context into a single question."""

    def __init__(self) -> None:
        super().__init__()
        signature = dspy.Signature(
            "context -> question:str",
            instructions=(
                "Given a passage from a legal decision, write one concise question in French "
                "that could be answered using only that passage. Output only the question text."
            ),
        )
        self.generator = dspy.ChainOfThought(signature)

    def forward(self, context: str) -> dspy.Prediction:
        return self.generator(context=context)


def configure_lm(cfg: QuestionGenConfig) -> dspy.LM:
    """Instantiate and register the LM client based on the provided flags."""
    api_key = cfg.api_key or os.environ.get("OPENAI_API_KEY")

    # Local sglang path (default): OpenAI-compatible server, no token required.
    if cfg.api_base:
        # litellm still expects a non-empty key; use a benign default when none is provided.
        api_key = api_key or "local"
        lm = dspy.LM(
            f"openai/{cfg.model}",
            api_base=cfg.api_base.rstrip("/"),
            api_key=api_key,
            model_type="chat",
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )
    else:
        # Fallback to HF inference only if explicitly requested.
        api_key = api_key or os.environ.get("HF_API_TOKEN")
        if not api_key:
            raise ValueError(
                "api_key (HF token) is required when no OpenAI-compatible api_base is provided. "
                "Set --api_base (default local sglang) to avoid needing a token."
            )
        lm = dspy.LM(
            f"huggingface/{cfg.model}",
            api_key=api_key,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )
    dspy.configure(lm=lm)
    return lm


def load_samples(cfg: QuestionGenConfig):
    """Load and optionally subsample the dataset slice requested by the user."""
    logging.info(
        "Loading dataset=%s config=%s split=%s column=%s",
        cfg.dataset,
        cfg.config,
        cfg.split,
        cfg.column,
    )
    dataset = load_dataset(cfg.dataset, cfg.config, split=cfg.split)
    if cfg.column not in dataset.column_names:
        raise ValueError(
            f"Column '{cfg.column}' not found in dataset columns: {dataset.column_names}"
        )
    if cfg.sample_size and cfg.sample_size > 0:
        if cfg.seed is not None:
            dataset = dataset.shuffle(seed=cfg.seed)
        dataset = dataset.select(range(min(cfg.sample_size, len(dataset))))
    return dataset


def generate_questions(cfg: QuestionGenConfig) -> Iterable[dict[str, str]]:
    """Generate questions for the configured dataset slice."""
    configure_lm(cfg)
    module = GenerateQuestion()
    dataset = load_samples(cfg)

    for idx, row in enumerate(dataset):
        context_text = (row.get(cfg.column) or "").strip()
        if not context_text:
            logging.warning("Skipping row with empty '%s' field.", cfg.column)
            continue
        prediction = module(context=context_text)
        question = (prediction.question or "").strip()
        if not question:
            logging.warning("No question generated for row; skipping.")
            continue
        yield {
            "question": question,
            "source_dataset": cfg.dataset,
            "source_config": cfg.config,
            "source_split": cfg.split,
            "source_column": cfg.column,
            "dataset_id": row.get("id"),
            "dataset_index": idx,
            "context_length": len(context_text),
        }


def run_cli(cfg: QuestionGenConfig) -> None:
    questions = list(generate_questions(cfg))
    if not questions:
        raise ValueError("No questions were generated; check dataset/column settings.")

    # Persist as Hugging Face–loadable parquet.
    output_path = cfg.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds = datasets.Dataset.from_list(questions)
    ds.to_parquet(str(output_path))

    print(f"Saved {len(questions)} questions to {output_path}")
    print(
        "Load with: load_dataset('parquet', data_files='%s')" % str(output_path)
    )


def main(argv=None):
    eapp.better_logging()
    flags_parser = eapp.make_flags_parser(QuestionGenConfig)
    app.run(run_cli, flags_parser=flags_parser, argv=argv)


if __name__ == "__main__":
    main()
