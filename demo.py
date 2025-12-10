# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     cell_metadata_filter: tags,jupyter,hide_input,-all
#     cell_metadata_json: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Préparation de l'environnement
"""

# %% {"tags": ["hide_code", "hide-input"], "jupyter": {"source_hidden": true}, "hide_input": true}

# Install dependencies (requirements.txt) and legal_rag (no deps). Safe for Colab/Jupyter.
import shutil
import subprocess
import sys
from pathlib import Path

REQ_FILE = Path("requirements.txt")
REQ_URL = "https://github.com/artefactory-argimi/legal_rag/releases/download/data-juri-v1/requirements.txt"
REPO_URL = "https://github.com/artefactory-argimi/legal_rag.git"  # change if you fork
UV_BIN = shutil.which("uv")
PIP_CMD = [UV_BIN, "pip"] if UV_BIN else [sys.executable, "-m", "pip"]
FORCE_FLAGS = ["--upgrade", "--force-reinstall"]


def _run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


# Install full dependencies (with deps) from published requirements, then install legal_rag sans deps.
_run([*PIP_CMD, "install", *FORCE_FLAGS, "-r", REQ_URL])
_run([*PIP_CMD, "install", *FORCE_FLAGS, "--no-deps", f"git+{REPO_URL}"])

# Verify import
import legal_rag as _  # noqa: F401

# %% [markdown]
"""
# Paramètres et instructions
"""

# %%
import os

DEFAULT_TOKEN = os.environ.get("HF_API_TOKEN", "")
GENERATOR_API_KEY = DEFAULT_TOKEN  # @param {type:"string"}
# OpenAI-compatible generator endpoint. HF router /v1 by default; Litellm will append /chat/completions.
# Choose the default router, or override with OpenAI or a local server.
GENERATOR_API_BASE = "http://localhost:8000/v1"  # @param ["https://router.huggingface.co/v1", "https://api.openai.com/v1", "http://localhost:8000/v1"]  # noqa: E501
GENERATOR_MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"  # @param ["Qwen/Qwen3-4B-Thinking-2507", "mistralai/Mistral-Small-3.1-24B-Instruct-2503", "HuggingFaceTB/SmolLM3-3B"]  # noqa: E501

# Encoder Hugging Face repo id (must match the model used to build the index).
ENCODER_MODEL_ID = "maastrichtlawtech/colbert-legal-french"  # @param {type:"string"}
# Index source built offline from the same encoder. Can be a remote zip or a
# local directory path to a pre-extracted index. INDEX_PATH controls where a zip
# is extracted if a local directory is not passed directly.
INDEX_URI = "https://github.com/artefactory-argimi/legal_rag/releases/download/data-juri-v1/index.zip"  # @param {type:"string"}
INDEX_PATH = "./downloads/index_legal_constit/"  # @param {type:"string"}
SEARCH_K = 10  # @param {type:"integer"}
MAX_NEW_TOKENS = 512  # @param {type:"integer"}
TEMPERATURE = 0.2  # @param {type:"number"}
MAX_ITERS = 4  # @param {type:"integer"}
INSTRUCTIONS = """Tu es un agent RAG spécialisé en jurisprudence constitutionnelle française (sous-ensemble 'constit' du jeu artefactory/Argimi-Legal-French-Jurisprudence).
Flow obligatoire des outils: (1) appelle search_legal_docs(query="…", k=10) puis (2) appelle lookup_legal_doc(chunk_id="…", score=…) sur les résultats renvoyés dès qu'au moins un score est suffisant. N'emploie jamais de clés args/kwargs ni d'autres paramètres, uniquement query/k ou chunk_id/score. Exemples exacts: search_legal_docs(query="résiliation contrat imprévision", k=5) puis lookup_legal_doc(chunk_id="JURITEXT000007022836-0", score=4.91). Les IDs retournés par search_legal_docs sont des chunk IDs au format "docid-chunkidx" (ex: "JURITEXT000007022836-0"). Toujours réutiliser tel quel le chunk_id (string) et le score renvoyés par search_legal_docs.
Pour chaque requête, formule un libellé de recherche précis: inclure le nom de la partie principale, la date (ou l'année) de la décision, la formation « Conseil constitutionnel », la nature de la question (QPC, contrôle a priori), les articles ou notions clés (ex. extradition, écrou, liberté individuelle), et un verbe d'action (contester, encadrer, garantir). Évite les formulations vagues; préfère les combinaisons de termes factuels et juridiques.
Périmètre: toute question mentionnant le Conseil constitutionnel, une QPC, la Constitution ou un article 61/61-1 est automatiquement considérée dans le périmètre: appelle search_legal_docs puis lookup_legal_doc. Ne renvoie jamais le message hors périmètre dans ces cas.
Hors périmètre: uniquement pour les sujets manifestement sans lien (ex. agriculture, technique, autres branches du droit). Dans ce seul cas, n'appelle aucun outil et répond strictement: "Le domaine demandé n'est pas couvert par cet agent (jurisprudence constitutionnelle française uniquement)."
Scores faibles: si tous les résultats d'un search_legal_docs ont un score inférieur ou égal à 8, ne lance pas lookup_legal_doc, reformule la requête et relance search_legal_docs; répète au maximum trois recherches (requête initiale + deux reformulations) avant de conclure à l'absence de résultats pertinents.
Inspection complète: dès qu'au moins un résultat a un score strictement supérieur à 8, effectue lookup_legal_doc sur tous les résultats renvoyés (dans l'ordre) jusqu'à disposer de suffisamment d'éléments pour répondre; si l'information manque, poursuivre jusqu'au dernier résultat avant de conclure.
Si search_legal_docs renvoie "No results.", indique clairement qu'aucune décision pertinente n'a été trouvée après tes tentatives.
Chaque réponse doit citer explicitement la jurisprudence utilisée (titre ou référence) et la date de la décision. Présente d'abord les éléments juridiques pertinents (faits, fondement, dispositif, articles cités), puis formule une réponse synthétique. La réponse doit être une interprétation fondée uniquement sur les décisions récupérées, jamais sur ta mémoire du modèle. Si aucune décision pertinente n'est récupérée ou si les éléments ne permettent pas de répondre, indique clairement que tu n'as pas les informations nécessaires pour répondre à la question. Réponds en français de façon précise et utile."""
configured_index = None

# %% [markdown]
"""
# Chargement des utilitaires
"""

# %% {"tags": ["hide_code", "hide-input"], "jupyter": {"source_hidden": true}, "hide_input": true}
import os
from pathlib import Path
from pprint import pformat
from typing import Iterable

from etils import ecolab
from rich import print as rprint

ecolab.auto_display()
ecolab.auto_inspect()
try:
    import google.colab  # type: ignore

    IN_COLAB = True
except Exception:
    IN_COLAB = False

# Avoid optional vision deps when loading text models.
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("DISABLE_TRANSFORMERS_AV", "1")
from legal_rag.assets import extract_zip, fetch_zip


def find_existing_index_root(base: Path) -> Path | None:
    """Return the index root if a fast_plaid_index layout exists under base.

    This avoids re-downloading an index when the destination already holds
    a valid index structure. The index root is resolved as the parent directory
    above the fast_plaid_index folder, matching prepare_index expectations.
    """

    if not base.exists():
        return None

    fast_plaid_paths = sorted(
        base.glob("**/fast_plaid_index"),
        key=lambda p: len(p.relative_to(base).parts),
    )
    if not fast_plaid_paths:
        return None

    index_root = fast_plaid_paths[0].parent.parent
    return index_root if index_root.exists() else None


# %% {"tags": ["hide_code", "hide-input"], "jupyter": {"source_hidden": true}, "hide_input": true}
def prepare_index(index_uri: str | Path, index_dest: Path) -> Path:
    """Fetch and extract the index assets; return the resolved index directory."""
    index_override = Path(index_uri).expanduser()
    index_dest = index_dest.expanduser()

    # Reuse an existing local index if present (either override or destination).
    existing_root = find_existing_index_root(
        index_override if index_override.is_dir() else index_dest
    )
    if existing_root:
        return existing_root

    # Download and extract when no valid index is found locally.
    index_dest.mkdir(parents=True, exist_ok=True)
    idx_zip_path = index_dest.parent / "_idx.zip"
    index_zip = fetch_zip(str(index_uri), idx_zip_path)
    try:
        extract_zip(index_zip, index_dest)
    finally:
        idx_zip_path.unlink(missing_ok=True)

    index_root = find_existing_index_root(index_dest)
    if not index_root:
        raise ValueError(
            f"Invalid index layout at {index_dest}. Expected a fast_plaid_index directory under this folder."
        )
    return index_root


# %% {"tags": ["hide_code"]}
# Run asset prep early so downstream cells only depend on local paths.
index_source = Path(INDEX_URI).expanduser()
configured_index = prepare_index(index_uri=INDEX_URI, index_dest=Path(INDEX_PATH))
print(f"✓ Encoder will be loaded from Hugging Face repo {ENCODER_MODEL_ID}")
print(f"✓ Index path resolved to {configured_index}")
if index_source.is_dir():
    print(f"✓ Using local index at {configured_index}")
else:
    print(f"✓ Index ready at {configured_index}")

# Increase HF download timeout to reduce transient failures when fetching models.
os.environ.setdefault("HF_HUB_TIMEOUT", "60")

# %% [markdown]
"""
# Construction de l'agent
"""

# %%
from legal_rag.agent import build_agent

generator_api_key = GENERATOR_API_KEY or None
generator_api_base = (GENERATOR_API_BASE or "https://router.huggingface.co/v1").rstrip(
    "/"
)
clean_model_id = GENERATOR_MODEL_ID

agent = build_agent(
    student_model=clean_model_id,
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
# Questions de démonstration
"""

# %%
queries: Iterable[str] | str = [
    "Comment régler l'arrosage goutte-à-goutte de tomates en serre pendant une canicule ?",
    "Que dit le Conseil constitutionnel, dans sa décision du 9 septembre 2016 sur M. Mukhtar A. (QPC 2016-561/562), au sujet des garanties de représentation à vérifier avant l'écrou extraditionnel et du contrôle de la durée raisonnable de cette détention ?",
]

# %%
for idx, question in enumerate(queries, start=1):
    rprint(f"[bold cyan]Question {idx}[/]: {question}")

    prediction = agent(question=question)
    rprint(f"[bold green]Réponse[/]\n{prediction.answer}")
    if IN_COLAB:
        ecolab.json(prediction.trajectory)  # interactive view in Colab
    else:
        rprint(
            f"[bold magenta]Trajectoire[/]\n{pformat(prediction.trajectory, width=100, sort_dicts=False)}"
        )

# %%
rprint(f"[bold cyan]Question {idx}[/]: {question}")

prediction = agent(question=question)
rprint(f"[bold green]Réponse[/]\n{prediction.answer}")
if IN_COLAB:
    ecolab.json(prediction.trajectory)  # interactive view in Colab
else:
    rprint(
        f"[bold magenta]Trajectoire[/]\n{pformat(prediction.trajectory, width=100, sort_dicts=False)}"
    )
