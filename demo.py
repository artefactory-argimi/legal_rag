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

# %%
import os
import sys
import zipfile
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

from etils import epath

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
# Index zip (URL/path) built offline from the same encoder.
INDEX_ZIP_URI = "https://github.com/artefactory-argimi/legal_rag/releases/download/data-juri-v1/index.zip"  # @param {type:"string"}
SEARCH_K = 5  # @param {type:"integer"}
MAX_NEW_TOKENS = 512  # @param {type:"integer"}
TEMPERATURE = 0.2  # @param {type:"number"}
MAX_ITERS = 4  # @param {type:"integer"}
INSTRUCTIONS = (
    "Tu es un agent RAG spécialisé en jurisprudence française (jeu de données artefactory/Argimi-Legal-French-Jurisprudence). "
    "Pour toute question juridique, commence par appeler search_legal_docs pour obtenir des ids et aperçus, puis lookup_legal_doc pour lire les décisions en intégralité. "
    "Formule ta réponse en t'appuyant sur le texte récupéré et cite le titre de la décision utilisée. "
    "Réponds en français de façon précise et utile. "
    "Exemples : "
    '- Q: "Quelles obligations de confraternité s\'imposent à un chirurgien-dentiste en campagne électorale ?" '
    "R: Résume la décision du Conseil national de l'Ordre des chirurgiens-dentistes (Titre: campagne électorale 1996) et rappelle les articles 21, 52, 54 du code déontologique. "
    '- Q: "Un avertissement disciplinaire peut-il être annulé pour critiques internes ?" '
    "R: Explique que des critiques vives mais sans imputations précises, diffusées en interne, n'ont pas dépassé les limites de la polémique électorale (Titre: avertissement annulé). "
    "- Q: \"Quelles limites à la liberté d'expression d'un praticien pendant une élection ordinale ?\" "
    "R: Mentionne que le devoir de confraternité subsiste mais doit être concilié avec la liberté d'expression syndicale (Titre: campagne électorale 1996)."
)  # @param {type:"string"}
configured_index = None

# %% [markdown]
"""
## Encoder and index assets (load first)
Provide zipped assets (local path or URL, e.g., GitHub release assets). The
encoder zip must contain the ColBERT checkpoint used for indexing. The index zip
must contain the PLAID index (e.g., an `index/` folder).
"""


# %%
def _fetch_zip(uri: str, dest: Path) -> Path:
    if not uri:
        raise FileNotFoundError("No URI provided for zip asset.")
    if uri.startswith("http://") or uri.startswith("https://"):
        dest.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                print(f"Downloading {uri} -> {dest} (attempt {attempt + 1}/3)")
                urlretrieve(uri, dest)
                break
            except Exception:
                if attempt == 2:
                    raise
        return dest
    src = Path(uri)
    if not src.exists():
        raise FileNotFoundError(f"Zip file not found: {src}")
    return src


def _extract_zip(zip_path: Path, target_dir: Path) -> Path:
    if target_dir.exists():
        # Clean existing contents to avoid mixing versions.
        for child in target_dir.iterdir():
            if child.is_file():
                child.unlink()
            else:
                import shutil

                shutil.rmtree(child)
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(target_dir))
    return target_dir


def prepare_assets() -> tuple[str, Path]:
    # Fetch and extract encoder.
    encoder_zip = _fetch_zip(ENCODER_ZIP_URI, Path("./encoder.zip"))
    encoder_dir = _extract_zip(encoder_zip, Path(ENCODER_MODEL_PATH))
    # Model files are expected at the root (no extra subfolder).
    encoder_path = str(encoder_dir.resolve())
    print(f"✓ Encoder extracted to {encoder_path}")

    # Fetch and extract index.
    index_zip = _fetch_zip(INDEX_ZIP_URI, Path("./index.zip"))
    index_dir = _extract_zip(index_zip, Path("./index"))
    print(f"✓ Index extracted to {index_dir}")
    return encoder_path, index_dir


# Run asset prep early so downstream cells only depend on local paths.
encoder_path, configured_index = prepare_assets()

# %% [markdown]
"""
## Repo setup

When opened directly from GitHub, this notebook installs the full repo so that
`agent.py` and utilities are importable. If the package is already installed,
the cell is a no-op.
"""

# %%
REPO_URL = "https://github.com/artefactory-argimi/legal_rag.git"  # change if you fork

try:
    import legal_rag as _  # noqa: F401
except ImportError:
    import subprocess

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--upgrade",
            f"git+{REPO_URL}",
        ],
        check=True,
    )

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
