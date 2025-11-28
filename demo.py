# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---

# %% [markdown]
"""
# French Legal Agent Demo

Colab-ready notebook (py:percent via Jupytext) for running the French Legal RAG
agent. It wires the DSPy agent, and lets you ask one or many questions. The LM can
run via Hugging Face Serverless Inference (with `HF_TOKEN`) or a local
OpenAI-compatible server (provide `GENERATOR_API_BASE`). DSPy (via LiteLLM)
auto-switches providers based on whether an API base is provided.
"""

# %% [markdown]
"""
## Configuration (Colab form)

All editable constants live here. Colab renders `@param` comments as form fields.
Tokens can come from login (`interpreter_login`) or manual entry; other fields use
these form values. Set `GENERATOR_API_KEY` to your own HF token, or point
`GENERATOR_API_BASE` to your OpenAI-compatible server to bypass HF serverless.
"""

# %%
GENERATOR_API_KEY = "local"  # @param {type:"string"}
GENERATOR_API_BASE = "http://localhost:8000/v1"  # @param {type:"string"}
GENERATOR_MODEL_ID = (
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503"  # @param {type:"string"}
)
# Encoder zip (URL/path) and extracted path must match the model used for indexing.
ENCODER_ZIP_URI = "https://github.com/artefactory-argimi/legal_rag/releases/download/data-juri-v1/colbert-encoder.zip"  # @param {type:"string"}
ENCODER_MODEL_PATH = "./encoder_model"  # path where the encoder zip will be extracted (model files at root)
# Index source built offline from the same encoder. Can be a remote zip or a
# local directory path to a pre-extracted index.
INDEX_URI = "https://github.com/artefactory-argimi/legal_rag/releases/download/data-juri-v1/index.zip"  # @param {type:"string"}
SEARCH_K = 5  # @param {type:"integer"}
MAX_NEW_TOKENS = 512  # @param {type:"integer"}
TEMPERATURE = 0.2  # @param {type:"number"}
MAX_ITERS = 4  # @param {type:"integer"}
INSTRUCTIONS = (
    "Tu es un agent RAG spécialisé en jurisprudence française (jeu de données artefactory/Argimi-Legal-French-Jurisprudence). "
    "Pour chaque question, appelle d'abord search_legal_docs pour trouver des décisions puis lookup_legal_doc pour lire les textes en intégralité. "
    "Chaque réponse doit citer explicitement la jurisprudence utilisée (titre ou référence) et la date de la décision. "
    "Présente d'abord les éléments juridiques pertinents (faits, fondement, dispositif, articles cités), puis formule une réponse synthétique. "
    "La réponse doit être une interprétation fondée uniquement sur les décisions récupérées, jamais sur ta mémoire du modèle. "
    "Si aucune décision pertinente n'est récupérée ou si les éléments ne permettent pas de répondre, indique clairement que tu n'as pas les informations nécessaires pour répondre à la question. "
    "Réponds en français de façon précise et utile."
)  # @param {type:"string"}
configured_index = None

# %%
import os
from pathlib import Path
from typing import Iterable

# Ensure the package is installed before importing.
REPO_URL = "https://github.com/artefactory-argimi/legal_rag.git"  # change if you fork
try:
    import legal_rag as _  # noqa: F401
except ImportError:
    try:
        get_ipython().run_line_magic(  # type: ignore[name-defined]
            "pip",
            f"install --quiet --upgrade git+{REPO_URL}",
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Failed to install legal_rag via %pip; please install manually.") from exc
    import legal_rag as _  # noqa: F401

from etils import epath
from legal_rag.assets import prepare_assets

# %% [markdown]
"""
## Encoder and index assets (load first)
Provide assets as a remote zip (URL) or local path. The encoder zip must contain
the ColBERT checkpoint used for indexing. For the index, you can point to a zip
(local path or URL) or directly to a local index directory that already
contains the PLAID files (e.g., an `index/` folder).
"""

# %%
# Run asset prep early so downstream cells only depend on local paths.
index_source = Path(INDEX_URI).expanduser()
encoder_path, configured_index = prepare_assets(
    encoder_zip_uri=ENCODER_ZIP_URI,
    index_zip_uri=INDEX_URI,
    encoder_dest=Path(ENCODER_MODEL_PATH),
    index_dest=Path("./index"),
)
print(f"✓ Encoder ready at {encoder_path}")
if index_source.is_dir():
    print(f"✓ Using local index at {configured_index}")
else:
    print(f"✓ Index ready at {configured_index}")

# %% [markdown]
"""
## Hugging Face login (Serverless Inference)

If running without a local generator and using the Hugging Face provider without
an `HF_TOKEN` set, prompt for a token using `huggingface_hub.interpreter_login()`.
"""

# %%
from huggingface_hub import get_token, interpreter_login

if not GENERATOR_API_BASE and not GENERATOR_API_KEY:
    # Default to HF serverless; prompt for token once if none was supplied.
    interpreter_login()
    GENERATOR_API_KEY = get_token() or ""

# Increase HF download timeout to reduce transient failures when fetching models.
os.environ.setdefault("HF_HUB_TIMEOUT", "60")

# %% [markdown]
"""
## Encoder ↔ Index coupling
The encoder loaded here must be the same model (or local snapshot path) used when
building the index. If you indexed with a different ColBERT checkpoint, update
the encoder assets accordingly to avoid mismatched embeddings.

## Agent configuration
We build the DSPy ReAct agent using the helpers in `agent.py`.

- Encoder: local, GPU if available (`torch.cuda.is_available()`), no API keys.
- Generator: defaults to Hugging Face Serverless (`huggingface/<model>` with token from
  `interpreter_login`) and falls back to a local OpenAI-compatible server when
  `GENERATOR_API_BASE` is provided.
"""

# %%
from legal_rag.agent import build_agent

generator_api_key = GENERATOR_API_KEY or None
# If no API base is set, default to HF Serverless and try to pick up a saved token.
if not GENERATOR_API_BASE and not generator_api_key:
    generator_api_key = get_token()
generator_api_base = GENERATOR_API_BASE or None

agent = build_agent(
    student_model=GENERATOR_MODEL_ID,
    encoder_model=encoder_path,
    generator_api_key=generator_api_key,
    generator_api_base=generator_api_base,
    index_folder=configured_index,  # used by ColBERT retriever in agent.py
    search_k=SEARCH_K,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    instructions=INSTRUCTIONS,
    max_iters=MAX_ITERS,
)

# %% [markdown]
"""
## Ask questions
Provide a single question as a string or multiple questions as an iterable.
The agent will search the index, optionally call lookup, and return grounded
answers. Adjust `queries` below and re-run the cell.
"""

# %%
queries: Iterable[str] | str = [
    "Quelles sont les obligations principales de l'employeur en matière de sécurité au travail ?",
    "Dans quel cas un contrat peut-il être résilié pour imprévision selon le droit français ?",
]

if isinstance(queries, str):
    questions = [queries]
else:
    questions = list(queries)

for idx, question in enumerate(questions, start=1):
    print(f"\n=== Question {idx} ===")
    print(question)
    prediction = agent(question=question)
    print("\n--- Réponse ---")
    print(prediction.answer)
