# Design Document: French Legal Agentic RAG

## Overview

### Objective
Deliver a notebook-first RAG experience that answers French legal questions by searching a dense index of jurisprudence and quoting the full decisions it uses.

### Goals
- Ground answers in retrieved decisions (ids, scores, and full text).
- Keep setup simple for Colab and local runs: install, download/extract assets, ask questions.
- Allow switching generator endpoints (HF router or any OpenAI-compatible server).
- Reuse retrieval assets via either a zipped index or a local index directory.

### Non-Goals
- Production hosting or high availability.
- Browser UI; the primary interface is the notebook.
- Coverage beyond the indexed French legal corpus.

### Key Principles
- Clear offline/online split: index building happens offline; inference consumes prebuilt assets.
- Artifact reuse: encoder and index can be provided as zips or existing directories; layout is normalized before use.
- Explicit scope in prompts: default instructions focus on constitutional jurisprudence but can be swapped with another index/prompt pair.
- Retrieval-first answers: always cite retrieved documents and show full decision text via the lookup tool.

## System Architecture

### High-Level Flow
- User question (French) enters the agent.
- Agent tool calls:
  - `search_legal_docs`: encodes the query with ColBERT and retrieves top-k chunk IDs/scores from the PLAID index.
  - `lookup_legal_doc`: given a chunk ID (+ optional score), parses the parent document ID and renders the full decision from the HF dataset.
- Agent responds in French, citing the decisions used.

### Chunk ID Format
Documents are split into chunks for indexing. Each chunk has an ID in the format `{parent_doc_id}-{chunk_index}`:
- `parent_doc_id`: Original document ID from the dataset (e.g., `JURITEXT000007022836`)
- `chunk_index`: 0-based index of the chunk within the document (e.g., `0`, `1`, `2`)
- Example: `JURITEXT000007022836-0`, `JURITEXT000007022836-1`

The `parse_chunk_id` function in `tools.py` extracts the parent document ID from a chunk ID for document lookup.

### Runtime Components
- Generator LM: configured through `build_language_model` in `src/legal_rag/agent.py`; supports HF Inference (token) or any OpenAI-compatible `api_base`. Defaults: `mistralai/Magistral-Small-2509` in code; `demo.py` exposes other options (e.g., Qwen 4B).
- Retrieval stack:
  - Encoder: `maastrichtlawtech/colbert-legal-french`.
  - Index: PLAID format with `fast_plaid_index` and `doc_mapping.json`.
  - Dataset: `artefactory/Argimi-Legal-French-Jurisprudence` loaded to resolve ids to full text.
- Notebook glue (`demo.py`): installs dependencies, downloads/extracts encoder/index if needed, configures the LM endpoint, builds the agent, and runs sample queries.

### Offline Assets
- Indexer (`src/legal_rag/indexer.py`, `scripts/indexer.py`):
  - Chunks documents using `chonkie.SentenceChunker` with a 511-token limit.
  - First chunk includes minimal metadata header (`Titre: {title} | Date: {decision_date}`).
  - Subsequent chunks contain only content text.
  - Each chunk is indexed with ID format `{doc_id}-{chunk_idx}` (e.g., `JURITEXT000007022836-0`).
  - Encodes chunks with ColBERT and builds a PLAID index (`legal_french_index` by default).
- Assets helper (`src/legal_rag/assets.py`): downloads or copies zips, extracts them, and resolves the correct root directories.

## Module Responsibilities
- `src/legal_rag/agent.py`: Builds the language model, loads encoder/retriever, and wires the `LegalReActAgent` with search/lookup tools and trajectory capture.
- `src/legal_rag/tools.py`: Implements the tools and utilities:
  - `parse_chunk_id(chunk_id)`: parses `{doc_id}-{chunk_idx}` into `(parent_doc_id, chunk_index)`.
  - `search_legal_docs(query, k)`: encode + retrieve, returns chunk IDs/scores.
  - `lookup_legal_doc(chunk_id, dataset, score)`: parses chunk ID, retrieves parent document, renders full text.
- `src/legal_rag/retriever.py`: Loads the ColBERT encoder and PLAID retriever, constraining worker count for constrained environments.
- `src/legal_rag/assets.py`: Fetches/extracts encoder/index zips and normalizes directory layouts.
- `src/legal_rag/indexer.py`: Offline PLAID builder with chunking, batching, and embedding:
  - `chunk_document(content, doc_id, title, decision_date, chunker)`: splits document into chunks with IDs.
  - `preprocess(sample)`: extracts fields for chunking.
  - `build_index(cfg)`: orchestrates chunking, encoding, and PLAID index creation.
- `demo.py`: Py:percent notebook that installs deps, prepares assets, configures generator endpoints, builds the agent, and runs sample queries.

## Data and Index Layout
- Index root must contain `<index_name>/fast_plaid_index` (defaults to `legal_french_index/fast_plaid_index`).
- `doc_mapping.json` stored at the index root:
  - `dataset`, `split`, `config`: identifiers for `load_dataset`.
  - `entries`: `{index_id: doc_id}` mapping PLAID index IDs to original document IDs.
- Dataset defaults: `artefactory/Argimi-Legal-French-Jurisprudence`, config `juri` unless overridden at index build time.
- Document ID column: defined by `ID_COLUMN` constant in `src/legal_rag/tools.py`, used consistently across indexer, question generation, and evaluation scripts.

## Notebook Execution Flow (demo.py)
1. Install dependencies (requirements.txt) and the package (no-deps).
2. Resolve generator endpoint from form fields (`GENERATOR_MODEL_ID`, `GENERATOR_API_KEY`, `GENERATOR_API_BASE`).
3. Prepare assets:
   - Download/extract encoder zip unless already present.
   - Download/extract index zip or reuse a provided directory; validate `fast_plaid_index`.
4. Build the agent with the encoder, retriever, dataset, and prompt instructions (default prompt emphasizes constitutional jurisprudence).
5. Run sample questions; display answers and tool trajectories.

## Question Generation Pipeline (DSPy)

### Overview
The `scripts/generate_question.py` script generates question-answer pairs from French legal documents using DSPy. It produces grounded QA pairs where answers are exact spans from the source text.

### Current Architecture
- **SummarizeAndExtractTopic**: Summarizes legal text and identifies main topic.
- **QAWithSpan**: Generates a question with an exact-span answer.
- **ValidateFrench**: Validates output language.
- **QAAgent**: Chains the above with manual retry logic when span matching fails.

### Recommended Improvements (No Prompt Optimization)

#### 1. Replace Manual Retry with `dspy.Refine`
Current approach uses ad-hoc retry when span isn't found. Replace with DSPy's built-in refinement:

```python
def qa_reward(args, pred) -> float:
    """Reward function: checks span presence and language."""
    context = args["context"]
    answer = (pred.answer or "").strip()
    span_start, _ = find_span(context, answer)

    score = 0.0
    if span_start is not None:
        score += 0.5  # answer found in context
    if pred.is_french:
        score += 0.5  # French language
    return score

refined_qa = dspy.Refine(
    module=qa_generator,
    N=3,
    reward_fn=qa_reward,
    threshold=1.0
)
```

#### 2. Use `Literal` Types for Constrained Outputs
Enforce language constraints at the signature level:

```python
from typing import Literal

class QAWithSpan(dspy.Signature):
    """Génère une question courte en français avec une réponse exacte tirée du texte."""

    context: str = dspy.InputField(desc="Texte source pour extraire la réponse")
    question: str = dspy.OutputField(
        desc="Question courte (max 15 mots) en français uniquement"
    )
    answer: str = dspy.OutputField(
        desc="Citation EXACTE du texte (max 10 mots), doit apparaître verbatim dans le contexte"
    )
    language: Literal["français"] = dspy.OutputField(
        desc="La langue de sortie - toujours français"
    )
```

#### 3. Use Pydantic Models for Structured Validation
Add schema-level validation:

```python
import pydantic

class QAPair(pydantic.BaseModel):
    question: str = pydantic.Field(max_length=150)
    answer: str = pydantic.Field(max_length=100)
    main_topic: str

    @pydantic.validator('question')
    def question_is_french(cls, v):
        french_patterns = ['qui', 'que', 'quoi', 'comment', 'pourquoi', 'où', 'quel']
        if not any(p in v.lower() for p in french_patterns + ['?']):
            raise ValueError("Question must be in French")
        return v

class QASignature(dspy.Signature):
    """Génère une paire question-réponse en français."""
    context: str = dspy.InputField()
    qa_pair: QAPair = dspy.OutputField()
```

#### 4. Use `dspy.BestOfN` for Quality Selection
Generate multiple candidates and select the best:

```python
def span_grounding_reward(args, pred) -> float:
    """Reward based on exact span matching."""
    context = args["context"]
    answer = pred.answer.strip()

    if answer in context:
        return 1.0
    if answer.lower() in context.lower():
        return 0.8
    return 0.0

best_qa = dspy.BestOfN(
    module=dspy.ChainOfThought(QAWithSpan),
    N=3,
    reward_fn=span_grounding_reward,
    threshold=1.0
)
```

#### 5. Merge Signatures to Reduce API Calls
Combine summarization and QA into a single multi-output signature:

```python
class FullQAExtraction(dspy.Signature):
    """Analyse un texte juridique français et génère une question-réponse fondée."""

    context: str = dspy.InputField(desc="Texte d'une décision juridique en français")
    summary: str = dspy.OutputField(desc="Résumé concis en 2-3 phrases")
    main_topic: str = dspy.OutputField(desc="Sujet principal en quelques mots")
    question: str = dspy.OutputField(desc="Question courte (max 15 mots) sur le sujet principal")
    answer: str = dspy.OutputField(desc="Citation EXACTE du texte (max 10 mots)")
```

### Improvement Summary

| Current Approach | Improved Approach |
|-----------------|-------------------|
| Manual retry loop | `dspy.Refine` with reward function |
| Separate `ValidateFrench` call | Integrated `Literal` types or reward function |
| Two separate signatures | Single combined signature |
| No quality selection | `dspy.BestOfN` for candidate selection |
| String outputs | Typed Pydantic models |
| Ad-hoc error handling | Structured `fail_count` parameter |

## Retrieval Evaluation Pipeline

### Overview
The `scripts/batch_query.py` script evaluates retrieval quality by querying the PLAID index with generated questions and outputting results in MSMARCO/TREC format for evaluation with pyserini.

### Workflow
1. Load questions from a generated Q&A dataset (produced by `scripts/generate_question.py`).
2. Load the `doc_mapping.json` which maps `index_id -> doc_id`.
3. Encode queries and retrieve top-k results from the PLAID index.
4. Output two files:
   - `run.txt`: TREC format (`qid Q0 doc_id rank score run_name`)
   - `qrels.txt`: Ground truth (`qid 0 doc_id 1`)

### Evaluation
Use pyserini to compute recall metrics:
```bash
python -m pyserini.eval.trec_eval -c -m recall.1,5,10 qrels.txt run.txt
```

### ID Naming Conventions
- `qid`: Query ID (UUID from question generation)
- `doc_id`: Original document ID from the HuggingFace dataset (e.g., `JURITEXT000006999374`)
- `chunk_id`: Indexed chunk ID in format `{doc_id}-{chunk_idx}` (e.g., `JURITEXT000006999374-0`)
- `chunk_idx`: 0-based index of chunk within parent document

## Current Gaps and Considerations
- Indexer assumes single-config builds; multi-config support would need batching/metadata extensions.
- Instructions are tuned to a constitutional subset; adjust prompt + index together for other domains.
- Index building is resource-intensive; intended for offline/GPU environments, not the Colab runtime.
