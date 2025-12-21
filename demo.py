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
GENERATOR_MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"  # @param ["Qwen/Qwen3-4B-Thinking-2507", "mistralai/Mistral-Small-3.1-24B-Instruct-2503", "HuggingFaceTB/SmolLM3-3B"]  # noqa: E501

# === Paramètres du Retriever (ColBERT) ===
ENCODER_MODEL_ID = "maastrichtlawtech/colbert-legal-french"  # @param {type:"string"}
INDEX_URI = "https://github.com/artefactory-argimi/legal_rag/releases/download/data-juri-v1/index.zip"  # @param {type:"string"}
INDEX_PATH = "./downloads/index_legal_constit/"  # @param {type:"string"}
SEARCH_K = 100  # @param {type:"integer"}

# === Paramètres de Génération ===
TEMPERATURE = 0.2  # @param {type:"number"}
MAX_ITERS = 4  # @param {type:"integer"}

# === Instructions système de l'agent ===
# Ces instructions définissent le comportement de l'agent : comment utiliser les outils,
# quand reformuler les requêtes, et comment structurer les réponses.
INSTRUCTIONS = """Tu es un agent RAG spécialisé en jurisprudence constitutionnelle française (sous-ensemble 'constit' du jeu artefactory/Argimi-Legal-French-Jurisprudence).
Outils disponibles:
- search_legal_docs(query="…"): Retourne 100 chunks avec leurs extraits. Exemple: search_legal_docs(query="QPC extradition écrou liberté individuelle 2016").
- lookup_chunk(chunk_id="…"): Récupère un chunk avec son contexte environnant pour vérifier sa pertinence. Exemple: lookup_chunk(chunk_id="JURITEXT000007022836-0").
- lookup_legal_doc(chunk_id="…"): Récupère le document complet pour formuler la réponse finale. Exemple: lookup_legal_doc(chunk_id="JURITEXT000007022836-0").
Workflow obligatoire:
1. Appelle search_legal_docs(query="…") pour obtenir 100 chunks avec extraits.
2. Analyse les extraits retournés et classe-les par pertinence (reranking).
3. Appelle lookup_chunk(chunk_id="…") sur les chunks les plus prometteurs pour confirmer leur utilité.
4. Si un chunk confirme un document pertinent, appelle lookup_legal_doc(chunk_id="…") pour obtenir le document complet et formuler ta réponse.
5. Si aucun chunk n'est pertinent, reformule la requête et relance search_legal_docs (max 3 tentatives).
Important: N'utilise lookup_legal_doc que pour formuler la réponse finale, jamais pour la recherche. Le document complet ne doit être lu qu'après confirmation de sa pertinence via lookup_chunk.
Les IDs retournés par search_legal_docs sont des chunk IDs au format "docid-chunkidx" (ex: "JURITEXT000007022836-0"). Toujours réutiliser tel quel le chunk_id renvoyé.
Pour chaque requête, formule un libellé de recherche précis: inclure le nom de la partie principale, la date (ou l'année) de la décision, la formation « Conseil constitutionnel », la nature de la question (QPC, contrôle a priori), les articles ou notions clés (ex. extradition, écrou, liberté individuelle), et un verbe d'action (contester, encadrer, garantir). Évite les formulations vagues; préfère les combinaisons de termes factuels et juridiques.
Périmètre: toute question mentionnant le Conseil constitutionnel, une QPC, la Constitution ou un article 61/61-1 est automatiquement considérée dans le périmètre. Ne renvoie jamais le message hors périmètre dans ces cas.
Hors périmètre: uniquement pour les sujets manifestement sans lien (ex. agriculture, technique, autres branches du droit). Dans ce seul cas, n'appelle aucun outil et répond strictement: "Le domaine demandé n'est pas couvert par cet agent (jurisprudence constitutionnelle française uniquement)."
Si search_legal_docs renvoie "No results.", indique clairement qu'aucune décision pertinente n'a été trouvée après tes tentatives.
Chaque réponse doit citer explicitement la jurisprudence utilisée (titre ou référence) et la date de la décision. Présente d'abord les éléments juridiques pertinents (faits, fondement, dispositif, articles cités), puis formule une réponse synthétique. La réponse doit être une interprétation fondée uniquement sur les décisions récupérées, jamais sur ta mémoire du modèle. Si aucune décision pertinente n'est récupérée ou si les éléments ne permettent pas de répondre, indique clairement que tu n'as pas les informations nécessaires pour répondre à la question. Réponds en français de façon précise et utile."""
configured_index = None

# %% [markdown]
"""
# 3. Chargement de l'Index ColBERT

Cette section télécharge et prépare l'index de recherche pré-construit.

**L'index contient** :
- Les embeddings ColBERT des décisions du Conseil constitutionnel
- La structure `fast_plaid_index` pour une recherche rapide

**Temps estimé** : ~2-3 minutes pour le premier téléchargement (~500 Mo)
"""

# %% {"tags": ["hide_code", "hide-input"], "jupyter": {"source_hidden": true}, "hide_input": true}
import os
from pathlib import Path
from typing import Iterable

from etils import ecolab
from rich import print as rprint

ecolab.auto_display()
ecolab.auto_inspect()

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
    index_folder=configured_index,  # used by ColBERT retriever in agent.py
    search_k=SEARCH_K,
    temperature=TEMPERATURE,
    instructions=INSTRUCTIONS,
    max_iters=MAX_ITERS,
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

# %% {"tags": ["hide_code", "hide-input"], "jupyter": {"source_hidden": true}, "hide_input": true}
queries: Iterable[str] | str = [
    # Question hors périmètre - l'agent doit refuser poliment
    "Comment régler l'arrosage goutte-à-goutte de tomates en serre pendant une canicule ?",
    # Question dans le périmètre - l'agent doit rechercher et répondre
    "Que dit le Conseil constitutionnel, dans sa décision du 9 septembre 2016 sur M. Mukhtar A. (QPC 2016-561/562), au sujet des garanties de représentation à vérifier avant l'écrou extraditionnel et du contrôle de la durée raisonnable de cette détention ?",
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
