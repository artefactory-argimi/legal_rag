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
"""# French Legal Agent Demo

Colab-ready notebook (py:percent via Jupytext) for running the French Legal RAG
agent. It mounts Google Drive when available, pulls the prebuilt index from
Drive, wires the DSPy agent, and lets you ask one or many questions."""

# %%
import os
import sys
from pathlib import Path
from typing import Iterable

from etils import epath

"""Detect Colab early; avoid hard imports elsewhere."""
try:
    import google.colab  # type: ignore

    IN_COLAB = True
except Exception:
    IN_COLAB = False

# %% [markdown]
"""## Configuration (Colab form)

All editable constants live here. Colab renders `@param` comments as form fields.
Tokens can still come from environment variables; other fields use these form values."""

# %%
GENERATOR_API_KEY = os.getenv("HF_TOKEN", "")  # @param {type:"string"}
ENCODER_API_KEY = ""  # @param {type:"string"}
GENERATOR_API_BASE = ""  # @param {type:"string"}
ENCODER_API_BASE = ""  # @param {type:"string"}
GENERATOR_MODEL_ID = "mistralai/Magistral-Small-2509"  # @param {type:"string"}
ENCODER_MODEL_ID = "maastrichtlawtech/colbert-legal-french"  # @param {type:"string"}
SEARCH_K = 5  # @param {type:"integer"}
MAX_NEW_TOKENS = 512  # @param {type:"integer"}
TEMPERATURE = 0.2  # @param {type:"number"}
MAX_ITERS = 4  # @param {type:"integer"}
INSTRUCTIONS = "First call search_legal_docs to find candidate ids and previews. Then call lookup_legal_doc on specific ids you want to read in full. Ground your answer in the retrieved text and cite the document ids you used."  # @param {type:"string"}
INDEX_PATH = (
    "/content/drive/MyDrive/legal_rag/index" if IN_COLAB else "./index"
)  # @param {type:"string"}
configured_index = epath.Path(INDEX_PATH)

# %% [markdown]
"""## Drive mounting (Colab only)
Mount Google Drive only when running in Colab."""

# %%
if IN_COLAB:
    drive_mount = epath.Path("/content/drive")
    from google.colab import drive  # type: ignore

    drive.mount(str(drive_mount), force_remount=False)

# %% [markdown]
"""## Index loading
We use a single configured path for the index. No duplicated paths between local and Colab."""

# %%
if not configured_index.exists():
    raise FileNotFoundError("No index found. Provide an index at the configured path.")

# %% [markdown]
"""## Agent configuration
We build the DSPy ReAct agent using the helpers in `agent.py`. The generator and
encoder each accept their own API key/base; set a base for local servers or omit
to use remote inference when a key is provided."""

# %%
from agent import build_agent

generator_api_key = GENERATOR_API_KEY or None
generator_api_base = GENERATOR_API_BASE or None
encoder_api_key = ENCODER_API_KEY or None
encoder_api_base = ENCODER_API_BASE or None

agent = build_agent(
    student_model=GENERATOR_MODEL_ID,
    encoder_model=ENCODER_MODEL_ID,
    encoder_api_key=encoder_api_key,
    encoder_api_base=encoder_api_base,
    generator_api_key=generator_api_key,
    generator_api_base=generator_api_base,
    index_folder=INDEX_PATH,
    search_k=SEARCH_K,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    instructions=INSTRUCTIONS,
    max_iters=MAX_ITERS,
)

# %% [markdown]
"""## Ask questions
Provide a single question as a string or multiple questions as an iterable.
The agent will search the index, optionally call lookup, and return grounded
answers. Adjust `queries` below and re-run the cell."""

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
