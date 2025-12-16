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

from legal_rag.tools import DEFAULT_DOC_ID_COLUMN


@dataclass(frozen=True)
class GenerationConfig:
    dataset: str = "artefactory/Argimi-Legal-French-Jurisprudence"
    config: str | None = None
    split: str = "train"
    text_column: str = "content"
    doc_id_column: str = DEFAULT_DOC_ID_COLUMN
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


class ExtractCandidates(dspy.Signature):
    """Extrait plusieurs informations clés d'un texte juridique pour générer des questions.

    Identifie des faits variés: dates, montants, noms, décisions, lieux, durées, etc.
    """

    context: str = dspy.InputField(desc="Texte d'une décision juridique en français")
    main_topic: str = dspy.OutputField(
        desc="Le sujet principal du texte en quelques mots, en français"
    )
    candidate_1: str = dspy.OutputField(
        desc="Premier fait clé: information factuelle courte (max 15 mots) extraite du texte"
    )
    candidate_2: str = dspy.OutputField(
        desc="Deuxième fait clé: information factuelle différente (max 15 mots)"
    )
    candidate_3: str = dspy.OutputField(
        desc="Troisième fait clé: information factuelle différente (max 15 mots)"
    )


class SelectAndGenerateQA(dspy.Signature):
    """Sélectionne le meilleur candidat et génère une question dont il est la réponse.

    Choisit le candidat qui permet de formuler la question la plus claire et naturelle.
    """

    context: str = dspy.InputField(desc="Texte source")
    main_topic: str = dspy.InputField(desc="Sujet principal du texte")
    candidates: str = dspy.InputField(
        desc="Liste des faits candidats extraits du texte"
    )
    selected_candidate: int = dspy.OutputField(
        desc="Numéro du candidat sélectionné (1, 2 ou 3) - celui qui permet "
        "la question la plus claire et vérifiable"
    )
    question: str = dspy.OutputField(
        desc="Question courte et naturelle en français (max 12 mots), "
        "formulée comme une requête de recherche, dont la réponse est le candidat sélectionné"
    )
    answer: str = dspy.OutputField(
        desc="Le candidat sélectionné, recopié exactement comme réponse"
    )


class ValidateQA(dspy.Signature):
    """Vérifie la qualité d'une paire question-réponse générée."""

    question: str = dspy.InputField(desc="La question posée")
    answer: str = dspy.InputField(desc="La réponse proposée")
    context: str = dspy.InputField(desc="Le texte source")
    is_french: bool = dspy.OutputField(
        desc="True si la question ET la réponse sont en français, False sinon"
    )
    answer_in_context: bool = dspy.OutputField(
        desc="True si la réponse (ou son contenu factuel) est présente dans le contexte, "
        "False sinon"
    )
    answers_question: bool = dspy.OutputField(
        desc="True si la réponse répond DIRECTEMENT et CORRECTEMENT à la question posée, "
        "False si la réponse est hors sujet ou ne correspond pas à ce qui est demandé"
    )


def qa_reward_fn(inputs: dict, pred: dspy.Prediction) -> float:
    """Fonction de récompense pour valider la cohérence question-réponse.

    Système de points (sur 4):
    - 1 point: La question et la réponse sont en français
    - 1 point: La réponse est présente dans le contexte
    - 2 points: La réponse répond directement à la question

    Returns:
        Score normalisé entre 0 et 1.
    """
    max_points = 4
    points = 0

    validator = dspy.Predict(ValidateQA)
    result = validator(
        question=pred.question,
        answer=pred.answer,
        context=inputs["context"],
    )

    if result.is_french:
        points += 1

    if result.answer_in_context:
        points += 1

    if result.answers_question:
        points += 2

    return points / max_points


class QAGenerator(dspy.Module):
    """Module de génération Q&A en deux étapes: extraction puis sélection."""

    def __init__(self) -> None:
        super().__init__()
        extract_instructions = (
            "Lis attentivement ce texte juridique français. "
            "Extrait 3 faits clés, variés et vérifiables: dates, montants, noms, "
            "décisions, lieux, durées, etc. "
            "Chaque fait doit être distinct et factuel. "
            "IMPORTANT: Tout en français."
        )
        select_instructions = (
            "Parmi les 3 candidats, choisis celui qui permet de formuler "
            "la question la plus claire et naturelle. "
            "La question doit ressembler à une requête de recherche. "
            "IMPORTANT: La réponse doit être le candidat sélectionné, recopié exactement. "
            "IMPORTANT: La réponse DOIT répondre directement à la question."
        )
        self.extractor = dspy.ChainOfThought(
            ExtractCandidates, instructions=extract_instructions
        )
        self.selector = dspy.ChainOfThought(
            SelectAndGenerateQA, instructions=select_instructions
        )

    def forward(self, context: str) -> dspy.Prediction:
        extracted = self.extractor(context=context)
        candidates = (
            f"1. {extracted.candidate_1}\n"
            f"2. {extracted.candidate_2}\n"
            f"3. {extracted.candidate_3}"
        )
        selected = self.selector(
            context=context,
            main_topic=extracted.main_topic,
            candidates=candidates,
        )
        return dspy.Prediction(
            question=selected.question,
            answer=selected.answer,
            main_topic=extracted.main_topic,
        )


class QAAgent(dspy.Module):
    """Agent DSPy avec extraction de candidats et auto-vérification via Refine.

    Étape 1: Extrait 3 faits candidats du document
    Étape 2: Sélectionne le meilleur et génère une question
    Étape 3: Vérifie la cohérence Q&A, réessaie avec feedback si nécessaire
    """

    def __init__(self, max_retries: int = 3) -> None:
        super().__init__()
        self.generator = dspy.Refine(
            module=QAGenerator(),
            N=max_retries,
            reward_fn=qa_reward_fn,
            threshold=1.0,
        )

    def forward(self, context: str) -> dspy.Prediction:
        pred = self.generator(context=context)
        return dspy.Prediction(
            question=pred.question,
            answer=pred.answer,
            main_topic=pred.main_topic,
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
        doc_id_column: str,
        max_context_chars: int,
    ) -> None:
        self.agent = agent
        self.text_column = text_column
        self.doc_id_column = doc_id_column
        self.max_context_chars = max_context_chars

    def __call__(self, row: dict, idx: int) -> dict | None:
        """Process a single row and return a record or None if failed."""
        try:
            context = (row[self.text_column] or "").strip()
            doc_id = str(row[self.doc_id_column])
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

            if not question or not answer:
                logging.warning(
                    "Skipping doc_id %s because question/answer is empty.", doc_id
                )
                return None

            span_start, span_end = find_span(context_excerpt, answer)
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
    doc_id_column: str,
    max_context_chars: int,
    num_threads: int = 4,
    max_records: int | None = None,
    log_every: int = 20,
) -> list[tuple]:
    """Build QA records from rows using parallel execution."""
    rows_list = list(rows)
    if max_records is not None and max_records > 0:
        rows_list = rows_list[:max_records]

    processor = RowProcessor(agent, text_column, doc_id_column, max_context_chars)
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
        doc_id_column=cfg.doc_id_column,
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
