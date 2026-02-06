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
# Agent RAG pour la Jurisprudence Constitutionnelle Française

Agent DSPy spécialisé dans l'analyse de la jurisprudence du Conseil constitutionnel.

## Architecture

| Composant | Rôle |
|-----------|------|
| **Retriever ColBERT** | Recherche sémantique dense (`maastrichtlawtech/colbert-legal-french`) |
| **LLM Générateur** | Synthèse des réponses (Mistral, Qwen, etc.) |

## Outils de l'agent

| Outil | Description |
|-------|-------------|
| `search_legal_docs` | Recherche N chunks (ordre aléatoire pour reranking) |
| `lookup_chunk` | Récupère un chunk avec contexte environnant |
| `lookup_legal_doc` | Récupère le document complet |
"""

# %% [markdown]
"""
# 1. Préparation de l'environnement

Installation automatique des dépendances si `legal_rag` n'est pas installé.
"""

# %% {"tags": ["hide_code", "hide-input"], "jupyter": {"source_hidden": true}, "hide_input": true}

# Install dependencies only if legal_rag is not already installed.
import os
import shutil
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

# Enable CPU fallback for unsupported MPS ops (must be set before PyTorch import).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

if find_spec("legal_rag") is None:
    REQ_URL = "https://github.com/artefactory-argimi/legal_rag/releases/download/data-juri-v1/requirements.txt"
    REPO_URL = "https://github.com/artefactory-argimi/legal_rag.git"
    UV_BIN = shutil.which("uv")
    PIP_CMD = [UV_BIN, "pip"] if UV_BIN else [sys.executable, "-m", "pip"]
    subprocess.check_call([*PIP_CMD, "install", "-r", REQ_URL])
    subprocess.check_call([*PIP_CMD, "install", "--no-deps", f"git+{REPO_URL}"])

import legal_rag as _  # noqa: F401

# %% [markdown]
"""
# 2. Configuration de l'Agent

Modifiez les paramètres ci-dessous pour adapter l'agent à votre environnement.

| Paramètre | Description |
|-----------|-------------|
| `GENERATOR_API_BASE` | Endpoint LLM (local `localhost:8000` ou Hugging Face) |
| `GENERATOR_MODEL_ID` | Modèle de génération (Mistral, Qwen, etc.) |
| `SEARCH_K` | Nombre de chunks récupérés pour le reranking |
| `TEMPERATURE` | Créativité (0 = déterministe, 1 = créatif) |
| `MAX_ITERS` | Iterations max de l'agent (recherche + lookup) |
"""

# %%
import os

# === Paramètres du Générateur (LLM) ===
DEFAULT_TOKEN = os.environ.get("HF_API_TOKEN", "")
GENERATOR_API_KEY = DEFAULT_TOKEN  # @param {type:"string"}
GENERATOR_API_BASE = "http://localhost:8000/v1"  # @param ["https://router.huggingface.co/v1", "https://api.openai.com/v1", "http://localhost:8000/v1"]  # noqa: E501
GENERATOR_MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"  # @param ["mistralai/Magistral-Small-2509", "Qwen/Qwen3-4B-Thinking-2507", "HuggingFaceTB/SmolLM3-3B"]  # noqa: E501

# === Paramètres du Retriever (ColBERT) ===
ENCODER_MODEL_ID = "maastrichtlawtech/colbert-legal-french"  # @param {type:"string"}
INDEX_URI = "https://github.com/artefactory-argimi/legal_rag/releases/download/data-juri-v1/index.zip"  # @param {type:"string"}
INDEX_PATH = "./downloads/index_legal_constit/"  # @param {type:"string"}
INDEX_NAME = "colbert_legal_french_constit_index"  # @param {type:"string"}
DATASET_CONFIG = "constit"  # @param ["constit", "juri", "cetat"] {type:"string"}
SEARCH_K = 20  # @param {type:"integer"}

# === Paramètres de Génération ===
TEMPERATURE = 0.2  # @param {type:"number"}
MAX_ITERS = 4  # @param {type:"integer"}

configured_index = None

# %% [markdown]
"""
# 3. Validation et Préparation

Cette section télécharge l'index ColBERT et vérifie la connectivité du serveur LLM.
"""

# %% {"tags": ["hide_code", "hide-input"], "jupyter": {"source_hidden": true}, "hide_input": true}
from pathlib import Path

import httpx
from legal_rag.assets import extract_zip, fetch_zip
from rich import print as rprint


def download_index(uri: str, path: Path, name: str) -> None:
    """Download and extract the index if not present."""
    root = path.expanduser() / name
    if root.exists():
        return
    path.mkdir(parents=True, exist_ok=True)
    zip_path = path.parent / "_idx.zip"
    try:
        extract_zip(fetch_zip(uri, zip_path), path)
    finally:
        zip_path.unlink(missing_ok=True)


def validate_index(path: Path, name: str) -> Path:
    """Validate PLAID index structure."""
    root = path / name
    fast_plaid = root / "fast_plaid_index"
    if not root.exists():
        raise FileNotFoundError(f"Index missing: {root}")
    if not fast_plaid.exists() or not (fast_plaid / "metadata.json").exists():
        raise FileNotFoundError(f"Invalid index: {root}")
    if not list(fast_plaid.glob("*.codes.npy")):
        raise FileNotFoundError(f"Incomplete index: {fast_plaid}")
    return root


def validate_llm() -> None:
    """Validate LLM server connectivity."""
    if "localhost" in GENERATOR_API_BASE or "127.0.0.1" in GENERATOR_API_BASE:
        base = GENERATOR_API_BASE.rstrip("/").rsplit("/v1", 1)[0]
        r = httpx.get(f"{base}/health", timeout=5)
        if r.status_code != 200:
            raise ConnectionError(f"LLM unhealthy: {r.status_code}")
    elif "huggingface" in GENERATOR_API_BASE and not GENERATOR_API_KEY:
        raise ValueError("HF_API_TOKEN required")


download_index(INDEX_URI, Path(INDEX_PATH), INDEX_NAME)
configured_index = validate_index(Path(INDEX_PATH), INDEX_NAME)
validate_llm()
rprint(f"[green]✓[/] Index: {configured_index.name}, LLM: {GENERATOR_MODEL_ID}")

# %% {"tags": ["hide_code", "hide-input"], "jupyter": {"source_hidden": true}, "hide_input": true}
import os
from typing import Iterable

from etils import ecolab

ecolab.auto_display()
ecolab.auto_inspect()

# Avoid optional vision deps when loading text models.
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("DISABLE_TRANSFORMERS_AV", "1")
os.environ.setdefault("HF_HUB_TIMEOUT", "60")

# %% [markdown]
"""
# 4. Construction de l'Agent RAG

L'agent utilise le workflow défini dans `LegalRAGSignature` :

```
Recherche → Reranking → Validation → Formulation (→ Reformulation si nécessaire)
```

Les instructions et le comportement sont encodés dans les signatures DSPy, pas dans des prompts manuels.
"""

# %% {"tags": ["hide_code", "hide-input"], "jupyter": {"source_hidden": true}, "hide_input": true}
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
    index_folder=configured_index.parent,  # PLAID expects the parent folder
    index_name=configured_index.name,  # Index name is the folder containing fast_plaid_index
    search_k=SEARCH_K,
    temperature=TEMPERATURE,
    max_iters=MAX_ITERS,
    dataset_config=DATASET_CONFIG,
)

# %% [markdown]
"""
# 5. Démonstration : Questions et Réponses

Nous testons l'agent avec deux types de questions :

| Question | Type | Comportement attendu |
|----------|------|---------------------|
| Agriculture | Hors périmètre | Refus poli sans appel d'outils |
| QPC Eric RAOULT | Dans le périmètre | Recherche → Reranking → Lookup → Réponse structurée |
"""

# %% [markdown]
"""
## Définition des questions de test

Modifiez cette liste pour tester d'autres questions sur la jurisprudence constitutionnelle.
"""

# %%
queries: Iterable[str] | str = [
    # Question hors périmètre - l'agent doit refuser poliment
    "Comment régler l'arrosage goutte-à-goutte de tomates en serre pendant une canicule ?",
    # Question dans le périmètre - l'agent doit rechercher et répondre
    "Quelle requête M. Eric RAOULT a-t-il présentée au Conseil constitutionnel",
]

# %% [markdown]
"""
## Exécution et résultats

Pour chaque question, l'agent retourne :
- `prediction.answer` : La réponse textuelle générée
- `prediction.trajectory` : L'historique des appels d'outils (recherche, reranking, lookup)
"""

# %%
for idx, question in enumerate(queries, start=1):
    rprint(f"[bold cyan]Question {idx}[/]: {question}")

    prediction = agent(question=question)
    rprint(f"[bold green]Réponse[/]\n{prediction.answer}")
    rprint("[bold magenta]Trajectoire[/]")
    ecolab.json(prediction.trajectory)
