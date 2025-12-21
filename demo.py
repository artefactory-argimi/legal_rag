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

Cette démonstration présente un agent RAG (Retrieval-Augmented Generation) spécialisé
dans l'analyse de la jurisprudence constitutionnelle française.

## Architecture de l'agent

L'agent combine deux composants principaux :
1. **Retriever ColBERT** : Recherche sémantique dense utilisant le modèle `maastrichtlawtech/colbert-legal-french`
2. **LLM Générateur** : Modèle de langage (Mistral, Qwen, etc.) pour synthétiser les réponses

## Outils disponibles

L'agent dispose de 3 outils pour la recherche documentaire :
- `search_legal_docs` : Recherche 100 chunks avec leurs extraits (aperçu)
- `lookup_chunk` : Récupère un chunk spécifique avec son contexte environnant
- `lookup_legal_doc` : Récupère le document complet (pour analyse approfondie)

## Flux de travail en deux étapes

```
1. Recherche    : search_legal_docs(query) → 100 chunks avec aperçus
2. Inspection   : lookup_chunk(chunk_id) → chunk + contexte environnant
3. (Optionnel)  : lookup_legal_doc(chunk_id) → document complet
4. Synthèse    : LLM génère la réponse basée sur le contenu récupéré
```
"""

# %% [markdown]
"""
# 1. Préparation de l'environnement

Cette cellule installe automatiquement les dépendances nécessaires :
- Le fichier `requirements.txt` depuis les releases GitHub
- Le package `legal_rag` (sans ses dépendances car déjà installées)

**Note** : L'installation utilise `uv` si disponible (plus rapide), sinon `pip`.
"""

# %% {"tags": ["hide_code", "hide-input"], "hide_input": true}

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

## Paramètres du générateur (LLM)
Ces paramètres configurent le modèle de langage qui génère les réponses :
- `GENERATOR_API_KEY` : Clé API pour accéder au service LLM
- `GENERATOR_API_BASE` : Endpoint de l'API (Hugging Face, OpenAI, ou serveur local)
- `GENERATOR_MODEL_ID` : Identifiant du modèle à utiliser

## Paramètres du retriever (ColBERT)
Ces paramètres configurent la recherche sémantique :
- `ENCODER_MODEL_ID` : Modèle d'encodage pour la recherche (doit correspondre à l'index)
- `INDEX_URI` : URL ou chemin vers l'index pré-construit
- `SEARCH_K` : Nombre de documents à récupérer par requête

## Paramètres de génération
- `TEMPERATURE` : Contrôle la créativité (0 = déterministe, 1 = créatif)
- `MAX_ITERS` : Nombre maximum d'itérations de l'agent (recherche + lookup)
"""

# %%
import os

# === Paramètres du Générateur (LLM) ===
DEFAULT_TOKEN = os.environ.get("HF_API_TOKEN", "")
GENERATOR_API_KEY = DEFAULT_TOKEN  # @param {type:"string"}
GENERATOR_API_BASE = "http://localhost:8000/v1"  # @param ["https://router.huggingface.co/v1", "https://api.openai.com/v1", "http://localhost:8000/v1"]  # noqa: E501
GENERATOR_MODEL_ID = "mistralai/Magistral-Small-2509"  # @param ["mistralai/Magistral-Small-2509", "Qwen/Qwen3-4B-Thinking-2507", "HuggingFaceTB/SmolLM3-3B"]  # noqa: E501

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

# === Instructions système de l'agent ===
# Ces instructions définissent le comportement de l'agent : comment utiliser les outils,
# quand reformuler les requêtes, et comment structurer les réponses.
INSTRUCTIONS = f"""Tu es un agent RAG spécialisé en jurisprudence constitutionnelle française (sous-ensemble 'constit' du jeu artefactory/Argimi-Legal-French-Jurisprudence).

Outils disponibles:
- search_legal_docs(query="…"): Retourne {SEARCH_K} chunks avec leurs extraits (ordre aléatoire, ensemble non ordonné).
- lookup_chunk(chunk_id="…"): Récupère un chunk avec son contexte environnant pour vérifier sa pertinence.
- lookup_legal_doc(chunk_id="…"): Récupère le document complet pour formuler la réponse finale.

Workflow obligatoire:
1. Appelle search_legal_docs(query="…") pour obtenir {SEARCH_K} chunks avec extraits.
2. RERANKING OBLIGATOIRE: Les chunks sont retournés dans un ordre aléatoire (ensemble non ordonné).
   Tu DOIS analyser TOUS les extraits et les classer par pertinence selon ces critères:
   - Correspondance avec les termes spécifiques de la question (noms, dates, références QPC/articles)
   - Pertinence du contexte juridique (type de décision, juridiction, matière)
   - Qualité de l'extrait pour répondre à la question posée
3. Appelle lookup_chunk(chunk_id="…") uniquement sur les 3-5 chunks les plus pertinents après ton analyse.
4. Si un chunk confirme un document pertinent, appelle lookup_legal_doc(chunk_id="…") pour obtenir le document complet et formuler ta réponse.
5. Si aucun chunk n'est pertinent, reformule la requête et relance search_legal_docs (max 3 tentatives).

Important: N'utilise lookup_legal_doc que pour formuler la réponse finale, jamais pour la recherche. Le document complet ne doit être lu qu'après confirmation de sa pertinence via lookup_chunk.
Les IDs retournés par search_legal_docs sont des chunk IDs au format "docid-chunkidx" (ex: "JURITEXT000007022836-0"). Toujours réutiliser tel quel le chunk_id renvoyé.

FORMULATION DES REQUÊTES (CRITIQUE pour le retrieval):
La requête doit être SPÉCIFIQUE pour retrouver le bon document parmi des milliers. Elle DOIT inclure des termes distinctifs:
- Noms propres: nom de la partie (M. Mukhtar A.), nom de loi (loi sur le foncier public)
- Références: numéro de QPC (2016-561/562), numéro d'article (article 696-11)
- Dates: année ou date précise (9 septembre 2016)
- Notions clés: termes juridiques spécifiques (écrou extraditionnel, garde à vue, liberté individuelle)

MAUVAISES requêtes (trop génériques, retournent des milliers de résultats):
- "Quel article a été déclaré conforme ?"
- "QPC liberté individuelle"
- "décision Conseil constitutionnel"

BONNES requêtes (spécifiques, permettent de retrouver LE document):
- "QPC 2016-561 Mukhtar écrou extraditionnel liberté individuelle septembre 2016"
- "article 696-11 code procédure pénale extradition garanties représentation"
- "loi foncier public articles déclarés conformes Constitution"

Combine plusieurs termes distinctifs pour maximiser la précision du retrieval.

Périmètre: toute question mentionnant le Conseil constitutionnel, une QPC, la Constitution ou un article 61/61-1 est automatiquement considérée dans le périmètre. Ne renvoie jamais le message hors périmètre dans ces cas.
Hors périmètre: uniquement pour les sujets manifestement sans lien (ex. agriculture, technique, autres branches du droit). Dans ce seul cas, n'appelle aucun outil et répond strictement: "Le domaine demandé n'est pas couvert par cet agent (jurisprudence constitutionnelle française uniquement)."
Si search_legal_docs renvoie "No results.", indique clairement qu'aucune décision pertinente n'a été trouvée après tes tentatives.
Chaque réponse doit citer explicitement la jurisprudence utilisée (titre ou référence) et la date de la décision. Présente d'abord les éléments juridiques pertinents (faits, fondement, dispositif, articles cités), puis formule une réponse synthétique. La réponse doit être une interprétation fondée uniquement sur les décisions récupérées, jamais sur ta mémoire du modèle. Si aucune décision pertinente n'est récupérée ou si les éléments ne permettent pas de répondre, indique clairement que tu n'as pas les informations nécessaires pour répondre à la question. Réponds en français de façon précise et utile."""
configured_index = None

# %% [markdown]
"""
# 2.1 Validation et Préparation
"""

# %% {"tags": ["hide_code", "hide-input"], "hide_input": true}
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

L'agent est construit avec la fonction `build_agent` qui initialise :
1. **Le retriever ColBERT** : Charge le modèle d'encodage et l'index
2. **Le client LLM** : Configure la connexion au service de génération
3. **Les 3 outils** : `search_legal_docs`, `lookup_chunk`, `lookup_legal_doc`

L'agent utilise un workflow en deux étapes :
1. **Recherche** : `search_legal_docs` retourne 100 chunks avec aperçus
2. **Reranking** : L'agent analyse les extraits et classe par pertinence
3. **Validation** : `lookup_chunk` confirme l'utilité des chunks sélectionnés
4. **Formulation** : `lookup_legal_doc` récupère le document complet pour la réponse
5. **Reformulation** : Si aucun résultat pertinent, nouvelle requête (max 3 tentatives)
"""

# %% {"tags": ["hide_code", "hide-input"], "hide_input": true}
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
    instructions=INSTRUCTIONS,
    max_iters=MAX_ITERS,
    dataset_config=DATASET_CONFIG,
)

# %% [markdown]
"""
# 5. Démonstration : Questions et Réponses

Nous testons l'agent avec deux types de questions :

## Question 1 : Hors périmètre (agriculture)
Une question sans rapport avec la jurisprudence constitutionnelle.

**Comportement attendu** : L'agent détecte que la question est hors périmètre et répond
sans appeler les outils de recherche.

## Question 2 : Dans le périmètre (QPC)
Une question précise sur une décision du Conseil constitutionnel.

**Comportement attendu** :
1. `search_legal_docs` : L'agent formule une requête précise et obtient 100 chunks avec aperçus
2. **Reranking** : L'agent analyse les extraits et sélectionne les plus pertinents
3. `lookup_chunk` : L'agent vérifie la pertinence des chunks sélectionnés avec leur contexte
4. `lookup_legal_doc` : L'agent récupère le document complet pour formuler sa réponse
5. **Réponse** : L'agent génère une réponse structurée citant la décision et sa date
"""

# %% {"tags": ["hide_code", "hide-input"], "hide_input": true}
queries: Iterable[str] | str = [
    # Question hors périmètre - l'agent doit refuser poliment
    "Comment régler l'arrosage goutte-à-goutte de tomates en serre pendant une canicule ?",
    # Question dans le périmètre - l'agent doit rechercher et répondre
    "Quelle requête M. Eric RAOULT a-t-il présentée au Conseil constitutionnel",
]

# %% [markdown]
"""
## Exécution des questions

Pour chaque question, l'agent retourne :
- `prediction.answer` : La réponse textuelle générée
- `prediction.trajectory` : L'historique des appels d'outils (recherche, lookup, etc.)
"""

# %%
for idx, question in enumerate(queries, start=1):
    rprint(f"[bold cyan]Question {idx}[/]: {question}")

    prediction = agent(question=question)
    rprint(f"[bold green]Réponse[/]\n{prediction.answer}")
    rprint("[bold magenta]Trajectoire[/]")
    ecolab.json(prediction.trajectory)

# %%
