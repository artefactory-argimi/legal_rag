#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "absl-py>=2.1.0",
#     "datasets>=3.2.0",
#     "dspy>=3.0.4",
#     "etils[eapp,epath]>=1.9.0",
# ]
# ///

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable

import datasets
import dspy
from absl import app, logging
from datasets import load_dataset
from etils import eapp, epath
from tqdm.auto import tqdm


@dataclass(frozen=True)
class GenerationConfig:
    dataset: str = "artefactory/Argimi-Legal-French-Jurisprudence"
    config: str | None = None
    split: str = "train"
    text_column: str = "content"
    id_column: str = "id"
    sample_limit: int | None = None
    seed: int | None = 0
    api_key: str | None = None
    api_base: str = "http://localhost:8000/v1"
    model: str = "local"
    temperature: float = 0.2
    max_tokens: int = 32768
    output_dir: epath.Path = epath.Path("./artifacts/question_answers")
    overwrite: bool = False
    max_context_chars: int = 8000


class QAWithSpan(dspy.Signature):
    """Lis le texte, propose une question et donne une réponse exacte tirée du texte."""

    context: str = dspy.InputField()
    question: str = dspy.OutputField()
    answer: str = dspy.OutputField()


class QAAgent(dspy.Module):
    """Simple DSPy agent that asks one grounded question per context."""

    def __init__(self) -> None:
        super().__init__()
        instructions = (
            "Lis le texte d'une décision juridique. Génère UNE question concise en français "
            "qui est répondable par ce texte (même langue que le texte). "
            "Donne une réponse exacte citée du texte (sans reformulation) pour permettre l'extraction d'un span. "
            "Vérifie que la question est bien répondable par le texte et que ta réponse est exactement présente dans le texte."
        )
        self.predictor = dspy.ChainOfThought(QAWithSpan, instructions=instructions)

    def forward(self, context: str) -> dspy.Prediction:
        return self.predictor(context=context)


def find_span(context: str, answer: str) -> tuple[int | None, int | None]:
    """Locate answer span in context (case-insensitive exact match)."""
    if not context or not answer:
        return None, None
    pattern = re.escape(answer.strip())
    match = re.search(pattern, context, flags=re.IGNORECASE)
    if match:
        return match.start(), match.end()
    return None, None


def trim_context(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    head = max_chars // 2
    tail = max_chars - head
    snippet = f"{text[:head].rstrip()}\n...\n{text[-tail:].lstrip()}"
    return snippet, True


def build_qa_records(
    agent: QAAgent,
    rows: Iterable[dict],
    *,
    text_column: str,
    id_column: str,
    max_context_chars: int,
    max_records: int | None = None,
    log_every: int = 20,
) -> list[dict]:
    records: list[dict] = []
    total = None
    try:
        total = len(rows)  # type: ignore[arg-type]
    except Exception:
        total = None

    for idx, row in enumerate(tqdm(rows, total=total, desc="Generating QA")):
        if max_records is not None and len(records) >= max_records:
            break
        context = (row.get(text_column) or "").strip()
        doc_id = str(row.get(id_column) or idx)
        if not context:
            logging.warning("Skipping doc_id %s because context is empty.", doc_id)
            continue

        context_excerpt, truncated = trim_context(context, max_context_chars)
        pred = agent(context=context_excerpt)

        question = (pred.question or "").strip()
        answer = (pred.answer or "").strip()

        if not question or not answer:
            logging.warning(
                "Skipping doc_id %s because question/answer is empty.", doc_id
            )
            continue

        # Heuristic language guard: if context is non-ASCII but question is pure ASCII, skip.
        context_has_non_ascii = any(ord(c) > 127 for c in context_excerpt)
        question_has_non_ascii = any(ord(c) > 127 for c in question)
        if context_has_non_ascii and not question_has_non_ascii:
            logging.warning(
                "Skipping doc_id %s because question does not match context language (expected non-ASCII).",
                doc_id,
            )
            continue

        # Self-check: if span not found, ask once more with a stricter prompt.
        span_start, span_end = find_span(context_excerpt, answer)
        if answer and span_start is None:
            follow_up = agent(
                context=context_excerpt
                + "\n\nRéponds en citant exactement une séquence de mots du texte."
            )
            answer = (follow_up.answer or answer).strip()
            span_start, span_end = find_span(context_excerpt, answer)

        if span_start is None or span_end is None:
            logging.warning(
                "Skipping doc_id %s because answer span not found in context.", doc_id
            )
            continue

        records.append(
            {
                "id": f"{doc_id}-qa",
                "doc_id": doc_id,
                "question": question,
                "answer": answer,
                "answer_span_start": span_start,
                "answer_span_end": span_end,
                "context_excerpt": context_excerpt,
                "context_was_truncated": truncated,
                "context_length": len(context),
            }
        )
        if total:
            percent = (idx + 1) * 100 // total
            if percent and percent % max(log_every, 1) == 0:
                logging.info(
                    "Progress %d%% — doc_id=%s | question=%s | answer=%s",
                    percent,
                    doc_id,
                    question,
                    answer,
                )
    return records


def save_dataset(
    records: list[dict], output_dir: epath.Path, overwrite: bool
) -> datasets.Dataset:
    output_dir = output_dir.expanduser()
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"{output_dir} already exists. Pass --overwrite to replace the existing dataset."
        )
    if output_dir.exists() and overwrite:
        output_dir.rmtree()
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = (
        datasets.Dataset.from_list(records)
        if records
        else datasets.Dataset.from_dict({"id": []})
    )
    ds.save_to_disk(str(output_dir))
    return ds


def main(cfg: GenerationConfig) -> None:
    logging.info(
        "Loading dataset=%s config=%s split=%s", cfg.dataset, cfg.config, cfg.split
    )
    if cfg.config:
        ds = load_dataset(cfg.dataset, cfg.config, split=cfg.split)
    else:
        ds = load_dataset(cfg.dataset, split=cfg.split)
    if cfg.seed is not None:
        ds = ds.shuffle(seed=cfg.seed)
    if cfg.sample_limit is not None and cfg.sample_limit > 0:
        ds = ds.select(range(min(cfg.sample_limit, len(ds))))

    api_key = cfg.api_key or os.environ.get("OPENAI_API_KEY") or "local"
    lm = dspy.LM(
        f"openai/{cfg.model}",
        api_base=cfg.api_base.rstrip("/"),
        api_key=api_key,
        model_type="chat",
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )
    dspy.configure(lm=lm)
    agent = QAAgent()

    records = build_qa_records(
        agent,
        ds,
        text_column=cfg.text_column,
        id_column=cfg.id_column,
        max_context_chars=cfg.max_context_chars,
        max_records=cfg.sample_limit,
        log_every=20,
    )

    ds_out = save_dataset(records, cfg.output_dir, cfg.overwrite)
    logging.info("Saved %d rows to %s", len(ds_out), cfg.output_dir)
    logging.info(
        "Inspect with: from datasets import load_from_disk; ds = load_from_disk('%s'); print(ds[0])",
        cfg.output_dir,
    )
    if len(ds_out):
        sample = ds_out[0]
        logging.info(
            "Sample -> doc_id: %s | question: %s | answer: %s | span: (%s, %s)",
            sample["doc_id"],
            sample["question"],
            sample["answer"],
            sample["answer_span_start"],
            sample["answer_span_end"],
        )


if __name__ == "__main__":
    eapp.better_logging()
    flags_parser = eapp.make_flags_parser(GenerationConfig)
    app.run(main, flags_parser=flags_parser)
