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
  - `search_legal_docs`: encodes the query with ColBERT and retrieves top-k ids/scores from the PLAID index.
  - `lookup_legal_doc`: given an id (+ optional score), renders the full decision using `doc_mapping.json` to access the HF dataset row.
- Agent responds in French, citing the decisions used.

### Runtime Components
- Generator LM: configured through `build_language_model` in `src/legal_rag/agent.py`; supports HF Inference (token) or any OpenAI-compatible `api_base`. Defaults: `mistralai/Magistral-Small-2509` in code; `demo.py` exposes other options (e.g., Qwen 4B).
- Retrieval stack:
  - Encoder: `maastrichtlawtech/colbert-legal-french`.
  - Index: PLAID format with `fast_plaid_index` and `doc_mapping.json`.
  - Dataset: `artefactory/Argimi-Legal-French-Jurisprudence` loaded to resolve ids to full text.
- Notebook glue (`demo.py`): installs dependencies, downloads/extracts encoder/index if needed, configures the LM endpoint, builds the agent, and runs sample queries.

### Offline Assets
- Indexer (`src/legal_rag/indexer.py`, `scripts/indexer.py`):
  - Templates metadata into each decision (`TEMPLATE_DOCUMENT`).
  - Encodes documents with ColBERT and builds a PLAID index (`legal_french_index` by default).
  - Writes `doc_mapping.json` with `{dataset, split, config, entries: plaid_id -> dataset_idx}` for lookups.
- Assets helper (`src/legal_rag/assets.py`): downloads or copies zips, extracts them, and resolves the correct root directories.

## Module Responsibilities
- `src/legal_rag/agent.py`: Builds the language model, loads encoder/retriever, and wires the `LegalReActAgent` with search/lookup tools and trajectory capture.
- `src/legal_rag/tools.py`: Implements the two tools:
  - `search_legal_docs(query, k)`: encode + retrieve, returns ids/scores.
  - `lookup_legal_doc(doc_id, mapping_entries, dataset, score)`: render full text for a retrieved id.
- `src/legal_rag/retriever.py`: Loads the ColBERT encoder and PLAID retriever, constraining worker count for constrained environments.
- `src/legal_rag/assets.py`: Fetches/extracts encoder/index zips and normalizes directory layouts.
- `src/legal_rag/indexer.py`: Offline PLAID builder with templating, batching, embedding, and `doc_mapping.json` creation.
- `demo.py`: Py:percent notebook that installs deps, prepares assets, configures generator endpoints, builds the agent, and runs sample queries.

## Data and Index Layout
- Index root must contain `<index_name>/fast_plaid_index` (defaults to `legal_french_index/fast_plaid_index`).
- `doc_mapping.json` stored at the index root:
  - `dataset`, `split`, `config`: identifiers for `load_dataset`.
  - `entries`: `{plaid_id: dataset_idx}` for lookups.
- Dataset defaults: `artefactory/Argimi-Legal-French-Jurisprudence`, config `juri` unless overridden at index build time.

## Notebook Execution Flow (demo.py)
1. Install dependencies (requirements.txt) and the package (no-deps).
2. Resolve generator endpoint from form fields (`GENERATOR_MODEL_ID`, `GENERATOR_API_KEY`, `GENERATOR_API_BASE`).
3. Prepare assets:
   - Download/extract encoder zip unless already present.
   - Download/extract index zip or reuse a provided directory; validate `fast_plaid_index`.
4. Build the agent with the encoder, retriever, dataset, and prompt instructions (default prompt emphasizes constitutional jurisprudence).
5. Run sample questions; display answers and tool trajectories.

## Current Gaps and Considerations
- No automated evaluation or optimization pipeline is present.
- Indexer assumes single-config builds; multi-config support would need batching/metadata extensions.
- Instructions are tuned to a constitutional subset; adjust prompt + index together for other domains.
- Index building is resource-intensive; intended for offline/GPU environments, not the Colab runtime.
