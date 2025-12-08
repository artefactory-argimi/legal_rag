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

import re
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import datasets
import dspy
from absl import app, logging
from datasets import load_dataset
from etils import eapp, epath
from dspy.utils.exceptions import AdapterParseError


@dataclass(frozen=True)
class GenerationConfig:
    dataset: str = "artefactory/Argimi-Legal-French-Jurisprudence"
    config: str | None = None
    split: str = "train"
    text_column: str = "content"
    id_column: str = "id"
    sample_limit: int | None = None
    seed: int | None = 0
    api_key: str = "local"
    api_base: str = "http://localhost:8000/v1"
    model: str = "local"
    temperature: float = 0.2
    max_tokens: int = 32768
    output_dir: epath.Path = epath.Path("./artifacts/question_answers")
    overwrite: bool = False
    max_context_chars: int = 8000
    num_threads: int = 4
    log_every: int = 20


class SummarizeAndExtractTopic(dspy.Signature):
    """Résume le texte juridique et identifie le sujet principal."""

    context: str = dspy.InputField(desc="Texte d'une décision juridique en français")
    summary: str = dspy.OutputField(
        desc="Résumé concis du texte en 2-3 phrases, en français uniquement"
    )
    main_topic: str = dspy.OutputField(
        desc="Le sujet principal du texte en quelques mots, en français uniquement"
    )


class QAWithSpan(dspy.Signature):
    """Génère une question courte en français avec une réponse exacte tirée du texte."""

    summary: str = dspy.InputField(desc="Résumé du texte juridique")
    main_topic: str = dspy.InputField(desc="Sujet principal du texte")
    context: str = dspy.InputField(desc="Texte source pour extraire la réponse")
    question: str = dspy.OutputField(
        desc="Question courte et précise en français (max 15 mots), portant sur le sujet principal"
    )
    answer: str = dspy.OutputField(
        desc="Réponse courte (max 10 mots) citée EXACTEMENT du texte, en français"
    )


class ValidateFrench(dspy.Signature):
    """Vérifie que le texte est en français."""

    text: str = dspy.InputField(desc="Texte à valider")
    is_french: bool = dspy.OutputField(
        desc="True si le texte est en français, False sinon"
    )


class QAAgent(dspy.Module):
    """Multi-step DSPy agent: summarize, extract topic, then generate one grounded Q&A."""

    def __init__(self) -> None:
        super().__init__()
        summarize_instructions = (
            "Lis attentivement ce texte juridique français. "
            "Produis un résumé concis (2-3 phrases) et identifie le sujet principal. "
            "IMPORTANT: Réponds UNIQUEMENT en français. "
            "Ne mélange JAMAIS le français avec d'autres langues."
        )
        qa_instructions = (
            "À partir du résumé et du sujet principal, génère UNE question courte en français "
            "(maximum 15 mots) portant sur le sujet principal. "
            "La réponse doit être une citation EXACTE du texte source (maximum 10 mots). "
            "IMPORTANT: Question et réponse UNIQUEMENT en français. "
            "Ne mélange JAMAIS le français avec d'autres langues (pas d'anglais, pas de mots étrangers). "
            "Vérifie que ta réponse est exactement présente dans le texte."
        )
        self.summarizer = dspy.ChainOfThought(
            SummarizeAndExtractTopic, instructions=summarize_instructions
        )
        self.qa_generator = dspy.ChainOfThought(
            QAWithSpan, instructions=qa_instructions
        )
        self.language_validator = dspy.Predict(ValidateFrench)

    def forward(self, context: str) -> dspy.Prediction:
        summary_pred = self.summarizer(context=context)
        qa_pred = self.qa_generator(
            summary=summary_pred.summary,
            main_topic=summary_pred.main_topic,
            context=context,
        )
        combined_text = f"{qa_pred.question} {qa_pred.answer}"
        validation = self.language_validator(text=combined_text)
        return dspy.Prediction(
            question=qa_pred.question,
            answer=qa_pred.answer,
            main_topic=summary_pred.main_topic,
            is_french=validation.is_french,
        )


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


class RowProcessor:
    """Callable wrapper for processing a single row with dspy.Parallel."""

    def __init__(
        self,
        agent: QAAgent,
        text_column: str,
        id_column: str,
        max_context_chars: int,
    ) -> None:
        self.agent = agent
        self.text_column = text_column
        self.id_column = id_column
        self.max_context_chars = max_context_chars

    def __call__(self, row: dict, idx: int) -> dict | None:
        """Process a single row and return a record or None if failed."""
        try:
            context = (row.get(self.text_column) or "").strip()
            doc_id = str(row.get(self.id_column) or idx)
            if not context:
                logging.warning("Skipping doc_id %s because context is empty.", doc_id)
                return None

            context_excerpt, _ = trim_context(context, self.max_context_chars)
            try:
                pred = self.agent(context=context_excerpt)
            except AdapterParseError as e:
                logging.warning(
                    "Skipping doc_id %s due to adapter parse error: %s", doc_id, e
                )
                return None
            except Exception as e:  # noqa: BLE001
                logging.warning("Skipping doc_id %s due to DSPy error: %s", doc_id, e)
                return None

            question = (pred.question or "").strip()
            answer = (pred.answer or "").strip()
            main_topic = (getattr(pred, "main_topic", "") or "").strip()
            is_french = getattr(pred, "is_french", True)

            if not question or not answer:
                logging.warning(
                    "Skipping doc_id %s because question/answer is empty.", doc_id
                )
                return None

            if not is_french:
                logging.warning(
                    "Skipping doc_id %s because Q&A is not in French: "
                    "question=%s, answer=%s",
                    doc_id,
                    question,
                    answer,
                )
                return None

            span_start, span_end = find_span(context_excerpt, answer)
            if answer and span_start is None:
                try:
                    retry_context = (
                        context_excerpt
                        + "\n\nIMPORTANT: La réponse doit être une citation EXACTE "
                        "du texte ci-dessus (max 10 mots), en français."
                    )
                    follow_up = self.agent(context=retry_context)
                    answer = (follow_up.answer or answer).strip()
                    main_topic = (
                        getattr(follow_up, "main_topic", "") or main_topic
                    ).strip()
                    is_french = getattr(follow_up, "is_french", True)
                    if not is_french:
                        logging.warning(
                            "Skipping doc_id %s because retry Q&A is not in French: "
                            "answer=%s",
                            doc_id,
                            answer,
                        )
                        return None
                    span_start, span_end = find_span(context_excerpt, answer)
                except Exception as e:  # noqa: BLE001
                    logging.warning(
                        "Skipping doc_id %s due to DSPy retry failure: %s", doc_id, e
                    )
                    return None

            if span_start is None or span_end is None:
                logging.warning(
                    "Skipping doc_id %s because answer span not found in context.",
                    doc_id,
                )
                return None

            return (
                str(uuid.uuid4()),
                doc_id,
                question,
                answer,
                span_start,
                span_end,
                main_topic,
            )
        except Exception as e:  # noqa: BLE001
            logging.warning("Skipping row %s due to unexpected error: %s", idx, e)
            return None


RECORD_FIELDS = (
    "id",
    "doc_id",
    "question",
    "answer",
    "answer_span_start",
    "answer_span_end",
    "main_topic",
)


def build_qa_records(
    agent: QAAgent,
    rows: Iterable[dict],
    *,
    text_column: str,
    id_column: str,
    max_context_chars: int,
    num_threads: int = 4,
    max_records: int | None = None,
    log_every: int = 20,
) -> list[tuple]:
    """Build QA records from rows using parallel execution."""
    rows_list = list(rows)
    if max_records is not None and max_records > 0:
        rows_list = rows_list[:max_records]

    processor = RowProcessor(agent, text_column, id_column, max_context_chars)
    parallel = dspy.Parallel(num_threads=num_threads)

    exec_pairs = [(processor, (row, idx)) for idx, row in enumerate(rows_list)]
    results = parallel(exec_pairs)

    records: deque[tuple] = deque()
    for result in results:
        if result is not None:
            records.append(result)
            if log_every > 0 and len(records) % log_every == 0:
                _, doc_id, question, answer, _, _, main_topic = result
                logging.info(
                    "Progress: %d records | doc_id=%s | topic=%s | Q=%s | A=%s",
                    len(records),
                    doc_id,
                    main_topic,
                    question,
                    answer,
                )

    logging.info("Generated %d records from %d rows", len(records), len(rows_list))
    return list(records)


def save_dataset(
    records: list[tuple], output_dir: epath.Path, overwrite: bool
) -> datasets.Dataset:
    output_dir = output_dir.expanduser()
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"{output_dir} already exists. Pass --overwrite to replace the existing dataset."
        )
    if output_dir.exists() and overwrite:
        output_dir.rmtree()
    output_dir.mkdir(parents=True, exist_ok=True)

    records_as_dicts = [dict(zip(RECORD_FIELDS, record)) for record in records]
    ds = (
        datasets.Dataset.from_list(records_as_dicts)
        if records_as_dicts
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

    lm = dspy.LM(
        f"openai/{cfg.model}",
        api_base=cfg.api_base.rstrip("/"),
        api_key=cfg.api_key,
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
        num_threads=cfg.num_threads,
        max_records=cfg.sample_limit,
        log_every=cfg.log_every,
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
            "Sample -> doc_id: %s | topic: %s | question: %s | answer: %s",
            sample["doc_id"],
            sample["main_topic"],
            sample["question"],
            sample["answer"],
        )


if __name__ == "__main__":
    eapp.better_logging()
    flags_parser = eapp.make_flags_parser(GenerationConfig)
    app.run(main, flags_parser=flags_parser)
