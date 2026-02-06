"""DSPy-based ReAct agent wiring for the Legal RAG demo (new dspy API)."""

from pathlib import Path

import dspy
from datasets import load_dataset
from etils import epath

from legal_rag.chunking import DocumentChunkCache
from legal_rag.retriever import build_encoder, build_retriever
from legal_rag.tools import (
    DEFAULT_DOC_ID_COLUMN,
    lookup_chunk as _lookup_chunk,
    lookup_legal_doc as _lookup_legal_doc,
    search_legal_docs as _search_legal_docs,
)

# Defaults aligned with the design doc; adjust via function arguments as needed.
DEFAULT_GENERATOR_MODEL = "mistralai/Magistral-Small-2509"
DEFAULT_ENCODER_MODEL = "maastrichtlawtech/colbert-legal-french"
DEFAULT_INDEX_FOLDER = epath.Path("./index")
DEFAULT_INDEX_NAME = "legal_french_index"
DEFAULT_SEARCH_K = 100  # Return 100 chunks for reranking
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.2
DEFAULT_DATASET = "artefactory/Argimi-Legal-French-Jurisprudence"
DEFAULT_CONFIG = "juri"
DEFAULT_SPLIT = "constit"


class LegalRAGSignature(dspy.Signature):
    """Agent RAG spécialisé en jurisprudence constitutionnelle française.

    Domaine couvert: décisions du Conseil constitutionnel, QPC, articles 61/61-1.
    Hors périmètre: agriculture, technique, autres branches du droit non constitutionnel.
    Pour les questions hors périmètre, répondre: "Le domaine demandé n'est pas couvert."

    Workflow obligatoire:
    1. Recherche avec search_legal_docs (chunks en ordre aléatoire, ensemble non ordonné)
    2. Analyse et reranking de TOUS les extraits par pertinence avant lookup
    3. Validation avec lookup_chunk sur les 3-5 chunks les plus pertinents
    4. Récupération du document complet avec lookup_legal_doc pour la réponse finale
    5. Reformulation de la requête si aucun résultat pertinent (max 3 tentatives)
    """

    question: str = dspy.InputField(
        desc="Question juridique sur la jurisprudence constitutionnelle française. "
        "Peut contenir: noms de parties, numéros QPC, dates, articles de loi."
    )
    answer: str = dspy.OutputField(
        desc="Réponse structurée en français citant explicitement la jurisprudence "
        "(titre ou référence, date de la décision). "
        "Présente les éléments juridiques: faits, fondement, dispositif, articles cités. "
        "Basée uniquement sur les documents récupérés, jamais sur la mémoire du modèle. "
        "Si aucune décision pertinente n'est trouvée, l'indiquer clairement."
    )


def build_language_model(
    *,
    student_model: str = DEFAULT_GENERATOR_MODEL,
    api_key: str | None = None,
    api_base: str | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dspy.LM:
    """Instantiate the generator LM using the current dspy.LM API.

    - Hugging Face Inference (default): student_model as the HF repo id,
      api_key is the HF token. Used when api_base is not provided.
    - Local OpenAI-compatible server (e.g., SGLang): student_model as the model id
      exposed by the server, api_base points to it, api_key optional, model_type="chat".
    """
    if api_base:
        lm_id = f"openai/{student_model}"
        return dspy.LM(
            lm_id,
            api_base=api_base,
            api_key=api_key,
            model_type="chat",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    if not api_key:
        raise ValueError("api_key (HF token) is required for Hugging Face inference.")
    lm_id = f"huggingface/{student_model}"
    return dspy.LM(
        lm_id,
        api_key=api_key,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


class LegalReActAgent(dspy.Module):
    """DSPy ReAct agent with two-stage retrieval.

    The agent uses a workflow:
    1. Search: Retrieve 100 chunks with previews
    2. Lookup: Examine selected chunks with surrounding context
    3. Synthesize: Generate answer based on retrieved content
    """

    def __init__(
        self,
        search_tool,
        lookup_chunk_tool,
        lookup_doc_tool,
        max_iters: int = 6,
    ) -> None:
        super().__init__()
        self.search_tool = search_tool
        self.lookup_chunk_tool = lookup_chunk_tool
        self.lookup_doc_tool = lookup_doc_tool

        self.react = dspy.ReAct(
            LegalRAGSignature,
            tools=[self.search_tool, self.lookup_chunk_tool, self.lookup_doc_tool],
            max_iters=max_iters,
        )

    def forward(self, question: str) -> dspy.Prediction:
        prediction = self.react(question=question)
        trajectory = prediction.trajectory

        # Collect retrieved documents and chunks
        documents: dict[str, str] = {}
        chunks: dict[str, str] = {}

        idx = 0
        while True:
            name_key = f"tool_name_{idx}"
            args_key = f"tool_args_{idx}"
            obs_key = f"observation_{idx}"

            if name_key not in trajectory:
                break

            tool_name = trajectory[name_key]
            args = trajectory[args_key]
            observation = trajectory[obs_key]

            if tool_name == "lookup_legal_doc" and isinstance(args, dict):
                chunk_id = args.get("chunk_id", args.get("id", None))
                if isinstance(chunk_id, str) and isinstance(observation, str):
                    doc_id = chunk_id.rsplit("-", 1)[0]
                    documents[doc_id] = observation

            elif tool_name == "lookup_chunk" and isinstance(args, dict):
                chunk_id = args.get("chunk_id", None)
                if isinstance(chunk_id, str) and isinstance(observation, str):
                    chunks[chunk_id] = observation

            idx += 1

        prediction.documents = documents
        prediction.chunks = chunks
        return prediction


def _validate_index(index_folder: epath.Path | str, index_name: str) -> None:
    """Validate that a PLAID index exists at the expected location.

    Args:
        index_folder: Parent directory containing the index.
        index_name: Name of the index folder (contains fast_plaid_index/).

    Raises:
        FileNotFoundError: If the index structure is invalid or missing.
    """
    index_path = Path(str(index_folder)).expanduser().resolve() / index_name
    fast_plaid_path = index_path / "fast_plaid_index"
    metadata_path = fast_plaid_path / "metadata.json"

    if not index_path.exists():
        raise FileNotFoundError(f"Index folder not found: {index_path}")
    if not fast_plaid_path.exists():
        raise FileNotFoundError(f"fast_plaid_index not found in: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Invalid PLAID index (missing metadata.json): {fast_plaid_path}"
        )


def build_agent(
    student_model: str = DEFAULT_GENERATOR_MODEL,
    encoder_model: str = DEFAULT_ENCODER_MODEL,
    generator_api_key: str | None = None,
    generator_api_base: str | None = None,
    index_folder: epath.Path = DEFAULT_INDEX_FOLDER,
    index_name: str = DEFAULT_INDEX_NAME,
    search_k: int = DEFAULT_SEARCH_K,
    mode: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_iters: int = 6,
    dataset_config: str = DEFAULT_CONFIG,
) -> LegalReActAgent:
    """Factory that wires LM, retrieval, and ReAct agent.

    Creates a two-stage retrieval agent:
    1. search_legal_docs: Returns 100 chunks with previews
    2. lookup_chunk: Retrieves specific chunk with surrounding context
    3. lookup_legal_doc: Retrieves full document (for deep analysis)
    """
    lm = build_language_model(
        student_model=student_model,
        api_key=generator_api_key or api_key,
        api_base=generator_api_base or api_base,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    dspy.configure(lm=lm)

    encoder = build_encoder(encoder_model=encoder_model)
    _validate_index(index_folder, index_name)
    retriever = build_retriever(
        index_folder=index_folder,
        index_name=index_name,
    )
    dataset = load_dataset(DEFAULT_DATASET, dataset_config, split=DEFAULT_SPLIT)

    # Create chunk cache for efficient runtime chunking
    chunk_cache = DocumentChunkCache(
        dataset=dataset,
        doc_id_column=DEFAULT_DOC_ID_COLUMN,
    )

    # Create tool wrappers with clean signatures for DSPy introspection.
    # Using closures instead of functools.partial preserves proper function signatures.
    def search_legal_docs(query: str) -> str:
        """Recherche des documents juridiques. Retourne des chunks en ordre aléatoire.

        Args:
            query: Requête spécifique avec noms, références QPC, dates ou notions clés.

        Returns:
            Liste de chunks avec extraits (ensemble non ordonné à analyser).
        """
        return _search_legal_docs(
            query=query,
            encoder=encoder,
            retriever=retriever,
            k=search_k,
            chunk_cache=chunk_cache,
        )

    def lookup_chunk(chunk_id: str) -> str:
        """Récupère un chunk avec son contexte environnant pour vérifier sa pertinence.

        Args:
            chunk_id: ID au format "docid-chunkidx" (ex: "JURITEXT000007022836-0").

        Returns:
            Texte du chunk avec contexte.
        """
        return _lookup_chunk(
            chunk_id=chunk_id,
            chunk_cache=chunk_cache,
            include_context=True,
            context_chunks=1,
        )

    def lookup_legal_doc(chunk_id: str) -> str:
        """Récupère le document complet avec métadonnées pour la réponse finale.

        Args:
            chunk_id: ID au format "docid-chunkidx".

        Returns:
            Document complet (titre, date, juridiction, contenu).
        """
        return _lookup_legal_doc(
            chunk_id=chunk_id,
            chunk_cache=chunk_cache,
            doc_id_column=DEFAULT_DOC_ID_COLUMN,
        )

    return LegalReActAgent(
        search_legal_docs,
        lookup_chunk,
        lookup_legal_doc,
        max_iters=max_iters,
    )
