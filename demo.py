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
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%

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

# %%
GENERATOR_API_KEY = ""  # @param {type:"string"}
GENERATOR_API_BASE = ""  # @param {type:"string"}
GENERATOR_MODEL_ID = "Qwen/Qwen3-0.6B"  # @param ["Qwen/Qwen3-0.6B", "mistralai/Mistral-Small-3.1-24B-Instruct-2503"]  # noqa: E501

# Encoder Hugging Face repo id (must match the model used to build the index).
ENCODER_MODEL_ID = "maastrichtlawtech/colbert-legal-french"  # @param {type:"string"}
# Index source built offline from the same encoder. Can be a remote zip or a
# local directory path to a pre-extracted index. INDEX_PATH controls where a zip
# is extracted if a local directory is not passed directly.
INDEX_URI = "https://github.com/artefactory-argimi/legal_rag/releases/download/data-juri-v1/index.zip"  # @param {type:"string"}
INDEX_PATH = "./downloaded/index_legal_constit/"  # @param {type:"string"}
SEARCH_K = 5  # @param {type:"integer"}
MAX_NEW_TOKENS = 512  # @param {type:"integer"}
TEMPERATURE = 0.2  # @param {type:"number"}
MAX_ITERS = 4  # @param {type:"integer"}
INSTRUCTIONS = """Tu es un agent RAG spécialisé en jurisprudence constitutionnelle française (sous-ensemble 'constit' du jeu artefactory/Argimi-Legal-French-Jurisprudence).
Flow obligatoire: (1) search_legal_docs(query="…", k=5) pour obtenir des ids et scores puis (2) lookup_legal_doc(doc_id="…", score=…) sur les ids utiles avant de répondre.
N'emploie jamais de clés args/kwargs ni d'autres paramètres, uniquement query/k ou doc_id/score.
Exemples exacts: search_legal_docs(query="résiliation contrat imprévision", k=5) puis lookup_legal_doc(doc_id="2491", score=4.91).
Toujours réutiliser tel quel le doc_id (string) et le score renvoyés par search_legal_docs.
Si search_legal_docs renvoie "No results.", indique clairement qu'aucune décision pertinente n'a été trouvée.
Si la question dépasse la jurisprudence constitutionnelle (autres branches du droit), explique que le domaine n'est pas couvert et n'appelle pas les outils.
Chaque réponse doit citer explicitement la jurisprudence utilisée (titre ou référence) et la date de la décision.
Présente d'abord les éléments juridiques pertinents (faits, fondement, dispositif, articles cités), puis formule une réponse synthétique.
La réponse doit être une interprétation fondée uniquement sur les décisions récupérées, jamais sur ta mémoire du modèle.
Si aucune décision pertinente n'est récupérée ou si les éléments ne permettent pas de répondre, indique clairement que tu n'as pas les informations nécessaires pour répondre à la question.
Réponds en français de façon précise et utile."""
configured_index = None

"""Run the install cell (demo_install.py) before executing this notebook."""

# %%
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


# %%
def prepare_index(index_uri: str | Path, index_dest: Path) -> Path:
    """Fetch and extract the index assets; return the resolved index directory."""
    index_override = Path(index_uri).expanduser()
    index_dest = index_dest.expanduser()
    index_ready = index_dest.exists() and any(index_dest.iterdir())
    if index_override.is_dir():
        index_dir = index_override
    elif index_ready:
        index_dir = index_dest
    else:
        index_dest.mkdir(parents=True, exist_ok=True)
        idx_zip_path = index_dest.parent / "_idx.zip"
        index_zip = fetch_zip(str(index_uri), idx_zip_path)
        try:
            index_dir = extract_zip(index_zip, index_dest)
        finally:
            idx_zip_path.unlink(missing_ok=True)
    fast_plaid_paths = list(index_dir.glob("**/fast_plaid_index"))
    if not fast_plaid_paths:
        raise ValueError(
            f"Invalid index layout at {index_dir}. Expected a fast_plaid_index directory under this folder."
        )
    fast_plaid = sorted(
        fast_plaid_paths, key=lambda p: len(p.relative_to(index_dir).parts)
    )[0]
    index_root = fast_plaid.parent.parent
    if not index_root.exists():
        raise ValueError(
            f"Could not resolve index root from fast_plaid_index at {fast_plaid}"
        )
    return index_root


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

# %%
from legal_rag.agent import build_agent

generator_api_key = GENERATOR_API_KEY or None
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

# %%
queries: Iterable[str] | str = [
    "Comment obtenir un permis de conduire au Canada ?",  # hors domaine (non constitutionnel français)
    "Quelles sont les conditions de recevabilité d'une question prioritaire de constitutionnalité par le Conseil constitutionnel ?",
]

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
