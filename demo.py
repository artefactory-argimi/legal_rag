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
#       jupytext_version: 1.18.1
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
import sys
import zipfile
from pathlib import Path
from typing import Iterable

from etils import epath

"""Detect Colab early; avoid hard imports elsewhere."""
try:
    import google.colab  # noqa: ignore

    IN_COLAB = True
except Exception:
    IN_COLAB = False

# %% [markdown]
"""
## Configuration (Colab form)

All editable constants live here. Colab renders `@param` comments as form fields.
Tokens can come from login (`interpreter_login`) or manual entry; other fields use
these form values. Set `GENERATOR_API_KEY` to your own HF token, or point
`GENERATOR_API_BASE` to your OpenAI-compatible server to bypass HF serverless.
"""

# %%
GENERATOR_API_KEY = ""  # @param {type:"string"}
GENERATOR_API_BASE = ""  # @param {type:"string"}
GENERATOR_MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"  # @param {type:"string"}
ENCODER_MODEL_ID = "maastrichtlawtech/colbert-legal-french"  # @param {type:"string"}
SEARCH_K = 5  # @param {type:"integer"}
MAX_NEW_TOKENS = 512  # @param {type:"integer"}
TEMPERATURE = 0.2  # @param {type:"number"}
MAX_ITERS = 4  # @param {type:"integer"}
INSTRUCTIONS = "First call search_legal_docs to find candidate ids and previews. Then call lookup_legal_doc on specific ids you want to read in full. Ground your answer in the retrieved text and cite the document ids you used."  # @param {type:"string"}
INDEX_PATH = "/content/index" if IN_COLAB else "./index"  # @param {type:"string"}
configured_index = epath.Path(INDEX_PATH)

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

If running in Colab and using the Hugging Face provider without an `HF_TOKEN`
set, prompt for a token using `huggingface_hub.interpreter_login()`.
"""

# %%
from huggingface_hub import get_token, interpreter_login

if not GENERATOR_API_BASE and not GENERATOR_API_KEY:
    # Default to HF serverless; prompt for token once if none was supplied.
    interpreter_login()
    GENERATOR_API_KEY = get_token() or ""

# %% [markdown]
"""
## Index loading
We use a single configured path for the index. No duplicated paths between local and Colab.

If the index folder is missing, you must upload a zipped archive (e.g.
`legal_rag_index.zip`); the upload flow will unpack it into `INDEX_PATH`.
"""

# %%
if not configured_index.exists():
    if not IN_COLAB:
        raise FileNotFoundError(
            "No index found. Run indexing locally or upload a zipped index in Colab."
        )

    from google.colab import files  # type: ignore

    uploaded = files.upload()
    if not uploaded:
        raise FileNotFoundError("No index found and no archive uploaded.")
    # Pick the first uploaded file.
    fn, data = next(iter(uploaded.items()))
    local_path = f"/content/{fn}"
    print(f'User uploaded file "{fn}" with length {len(data)} bytes')

    archive = epath.Path(local_path)
    if not archive.name.lower().endswith(".zip"):
        raise ValueError(f"Uploaded file is not a zip archive: {archive}")

    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall("/content")

    if not configured_index.exists():
        # If the archive contained a folder with a different name, try to locate it.
        for child in epath.Path("/content").iterdir():
            if child.is_dir() and (child / "doc_mapping.json").exists():
                configured_index = child
                break
    if not configured_index.exists():
        raise FileNotFoundError(
            f"Archive extracted but index folder missing at {configured_index}"
        )
    print(f"✓ Index extracted to {configured_index}")

# %% [markdown]
"""
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
    encoder_model=ENCODER_MODEL_ID,
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
