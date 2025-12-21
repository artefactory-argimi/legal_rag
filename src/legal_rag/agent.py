"""DSPy-based ReAct agent wiring for the Legal RAG demo (new dspy API)."""

from functools import partial
from pathlib import Path

import dspy
from datasets import load_dataset
from etils import epath

from legal_rag.chunking import DocumentChunkCache
from legal_rag.retriever import build_encoder, build_retriever
from legal_rag.tools import (
    DEFAULT_DOC_ID_COLUMN,
    lookup_chunk,
    lookup_legal_doc,
    search_legal_docs,
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
DEFAULT_SPLIT = "train"


class RerankChunks(dspy.Signature):
    """Analyse les résultats de recherche et sélectionne les chunks les plus pertinents.

    Tu reçois une liste de chunks avec leurs extraits. Analyse chaque extrait et
    sélectionne les chunk_ids les plus pertinents pour répondre à la question.
    Ordonne-les du plus pertinent au moins pertinent.
    """

    question: str = dspy.InputField(desc="La question posée par l'utilisateur")
    search_results: str = dspy.InputField(
        desc="Résultats de recherche avec chunk_id, score et extrait de texte"
    )
    selected_chunk_ids: list[str] = dspy.OutputField(
        desc="Liste ordonnée des chunk_ids les plus pertinents (max 10), "
        "du plus pertinent au moins pertinent"
    )
    reasoning: str = dspy.OutputField(
        desc="Explication du choix des chunks sélectionnés"
    )


class SynthesizeAnswer(dspy.Signature):
    """Synthétise une réponse à partir des chunks analysés.

    Utilise les informations des chunks pour répondre à la question.
    Cite les sources (chunk_ids ou doc_ids) utilisées dans la réponse.
    """

    question: str = dspy.InputField(desc="La question posée")
    chunks_content: str = dspy.InputField(
        desc="Contenu des chunks sélectionnés avec contexte"
    )
    answer: str = dspy.OutputField(
        desc="Réponse complète et bien structurée à la question"
    )
    sources: list[str] = dspy.OutputField(
        desc="Liste des chunk_ids ou doc_ids utilisés comme sources"
    )


DEFAULT_INSTRUCTIONS = (
    "Tu es un agent juridique spécialisé dans la recherche documentaire française. "
    "Workflow en deux étapes:\n"
    "1. Appelle search_legal_docs pour obtenir 100 chunks avec leurs extraits.\n"
    "2. Analyse les extraits et appelle lookup_chunk sur les chunks les plus pertinents "
    "(avec contexte environnant) pour les examiner en détail.\n"
    "3. Si tu as besoin du document complet, appelle lookup_legal_doc.\n"
    "Base ta réponse sur le texte récupéré et cite les document IDs utilisés."
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
        instructions: str = DEFAULT_INSTRUCTIONS,
    ) -> None:
        super().__init__()
        self.search_tool = search_tool
        self.lookup_chunk_tool = lookup_chunk_tool
        self.lookup_doc_tool = lookup_doc_tool

        signature = dspy.Signature(
            "question -> answer:str",
            instructions=instructions,
        )
        self.react = dspy.ReAct(
            signature,
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


def _resolve_index_dir(base: epath.Path | str) -> epath.Path:
    """Return the index root (parent of the folder that contains fast_plaid_index)."""
    base_path = Path(str(base)).expanduser().resolve()
    fast_paths = sorted(
        (p for p in base_path.glob("**/fast_plaid_index") if p.is_dir()),
        key=lambda p: len(p.relative_to(base_path).parts),
    )
    if not fast_paths:
        raise FileNotFoundError(f"No fast_plaid_index found under {base_path}")
    fast_dir = fast_paths[0]
    return epath.Path(fast_dir.parent.parent)


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
    instructions: str = DEFAULT_INSTRUCTIONS,
    max_iters: int = 6,
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
    resolved_index_folder = _resolve_index_dir(index_folder)
    retriever = build_retriever(
        index_folder=resolved_index_folder,
        index_name=index_name,
    )
    dataset = load_dataset(DEFAULT_DATASET, DEFAULT_CONFIG, split=DEFAULT_SPLIT)

    # Create chunk cache for efficient runtime chunking
    chunk_cache = DocumentChunkCache(
        dataset=dataset,
        doc_id_column=DEFAULT_DOC_ID_COLUMN,
    )

    # Search tool: returns 100 chunks with previews
    search_tool = partial(
        search_legal_docs,
        encoder=encoder,
        retriever=retriever,
        k=search_k,
        chunk_cache=chunk_cache,
    )
    search_tool.__name__ = "search_legal_docs"

    # Chunk lookup tool: returns specific chunk with context
    lookup_chunk_tool = partial(
        lookup_chunk,
        chunk_cache=chunk_cache,
        include_context=True,
        context_chunks=1,
    )
    lookup_chunk_tool.__name__ = "lookup_chunk"

    # Document lookup tool: returns full document
    lookup_doc_tool = partial(
        lookup_legal_doc,
        chunk_cache=chunk_cache,
        doc_id_column=DEFAULT_DOC_ID_COLUMN,
    )
    lookup_doc_tool.__name__ = "lookup_legal_doc"

    return LegalReActAgent(
        search_tool,
        lookup_chunk_tool,
        lookup_doc_tool,
        max_iters=max_iters,
        instructions=instructions,
    )
