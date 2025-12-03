# French Legal Agentic RAG

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/artefactory-argimi/legal_rag/blob/main/demo.ipynb)

A notebook-first demo that answers French legal questions by searching a dense index of jurisprudence, then quoting the full decisions it finds.

## What you can do

- Ask legal questions in French and get answers grounded in retrieved decisions.
- See which documents were used: the agent lists document ids and shows full text via the lookup tool.
- Run it quickly in Colab: the notebook installs dependencies, fetches the index zip if needed, and wires the agent.
- Swap providers: use a Hugging Face token or point to any OpenAI-compatible endpoint for generation.
- Bring your own index: pass a local index folder or a zip URL; the default uses the published ColBERT index built from `artefactory/Argimi-Legal-French-Jurisprudence`.

## Try it in Colab

1. Open the notebook with the badge above.
2. Fill `GENERATOR_API_KEY` (HF token or OpenAI-style key) and optionally change `GENERATOR_API_BASE` (HF router by default).
3. Keep the default encoder (`maastrichtlawtech/colbert-legal-french`) and choose an index: the shipped zip URL works out of the box, or point to a local directory.
4. Run the cells and adapt the sample questions; responses are produced in French and cite the retrieved decisions.

## Run locally

```bash
git clone https://github.com/artefactory-argimi/legal_rag.git
cd legal_rag
pip install -e .
# Open demo.py with Jupytext/Jupyter and set your generator credentials + index path
```

Ensure an index folder with `fast_plaid_index` and `doc_mapping.json` is available locally; set `HF_API_TOKEN` or provide an OpenAI-compatible endpoint for generation.

## How it works (briefly)

- Agent: a lightweight question-answering loop that alternates between searching (`search_legal_docs`) and reading full documents (`lookup_legal_doc`).
- Retrieval: ColBERT encoder (`maastrichtlawtech/colbert-legal-french`) with a PLAID index and a mapping back to the Hugging Face dataset for full-text rendering.
- Assets: helper utilities download/extract the encoder and index zips in the notebook; the offline indexer templatizes metadata into each decision before encoding.
- Scope: the default prompt in `demo.py` focuses on constitutional jurisprudence; swap the index or instructions to target other subsets.

## Repository layout

- `demo.py`: Py:percent notebook entrypoint used in Colab or locally.
- `src/legal_rag/agent.py`: Wires the language model, retrieval tools, and agent loop.
- `src/legal_rag/tools.py`: Search and lookup helpers that return ids, scores, and full decisions.
- `src/legal_rag/retriever.py`: Loads the ColBERT encoder and PLAID retriever.
- `src/legal_rag/assets.py`: Downloads/extracts encoder and index zips and normalizes their layout.
- `src/legal_rag/indexer.py` and `scripts/indexer.py`: Offline pipeline to build a PLAID index and `doc_mapping.json` from `artefactory/Argimi-Legal-French-Jurisprudence`.

## About ArGiMi

This demo is part of the ArGiMi initiative (“Digital Commons for Generative Artificial Intelligence,” supported by France 2030) led by Artefact, Giskard, Mistral AI, INA, and BnF to build domain-specialized language models.
