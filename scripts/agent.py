"""DSPy-based ReAct agent wiring for the Legal RAG demo (new dspy API)."""

from enum import Enum

import dspy
from etils import epath
from pylate import indexes, models, retrieve

from scripts import indexer
from scripts.tools import lookup_legal_doc, search_legal_docs_metadata

# Defaults aligned with the design doc; adjust via function arguments as needed.
DEFAULT_STUDENT_MODEL = "mistralai/Magistral-Small-2509"
DEFAULT_ENCODER_MODEL = "maastrichtlawtech/colbert-legal-french"
DEFAULT_INDEX_FOLDER = epath.Path("./index")
DEFAULT_INDEX_NAME = "legal_french_index"
DEFAULT_SEARCH_K = 5
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.2
DEFAULT_INSTRUCTIONS = (
    "First call search_legal_docs to find candidate ids and previews. "
    "Then call lookup_legal_doc on specific ids you want to read in full. "
    "Ground your answer in the retrieved text and cite the document ids you used."
)


class LMMode(str, Enum):
    HUGGINGFACE = "huggingface"
    LOCAL_OPENAI = "local_openai"


def build_language_model(
    *,
    student_model: str = DEFAULT_STUDENT_MODEL,
    mode: LMMode = LMMode.HUGGINGFACE,
    api_key: str | None = None,
    api_base: str | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dspy.LM:
    """Instantiate the LM using the current dspy.LM API.

    - Hugging Face Inference: mode="huggingface", student_model as the HF repo id,
      api_key is the HF token.
      Example: dspy.LM("huggingface/mistralai/Magistral-Small-2509", api_key=...).
    - Local OpenAI-compatible server (e.g., SGLang): mode="local_openai",
      student_model as the model id exposed by the server, api_base points to it,
      api_key any non-empty string, model_type="chat".
      Example: dspy.LM("openai/meta-llama/Llama-3.1-8B-Instruct",
                       api_base="http://localhost:7501/v1",
                       api_key="local",
                       model_type="chat").
    """
    if mode == LMMode.HUGGINGFACE:
        if not api_key:
            raise ValueError("api_key (HF token) is required for Hugging Face inference.")
        lm_id = f"huggingface/{student_model}"
        return dspy.LM(
            lm_id,
            api_key=api_key,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    if mode == LMMode.LOCAL_OPENAI:
        if not api_base:
            raise ValueError("api_base is required for local_openai mode.")
        if not api_key:
            raise ValueError("api_key is required for local_openai mode (use a dummy string if needed).")
        lm_id = f"openai/{student_model}"
        return dspy.LM(
            lm_id,
            api_base=api_base,
            api_key=api_key,
            model_type="chat",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported mode '{mode}'. Use LMMode.HUGGINGFACE or LMMode.LOCAL_OPENAI.")


def build_retrieval(
    encoder_model: str = DEFAULT_ENCODER_MODEL,
    index_folder: epath.Path = DEFAULT_INDEX_FOLDER,
    index_name: str = DEFAULT_INDEX_NAME,
) -> tuple[models.ColBERT, retrieve.ColBERT]:
    """Load encoder and retriever from disk."""
    encoder = models.ColBERT(
        model_name_or_path=encoder_model,
        document_length=496,
    )
    encoder = indexer.fix_colbert_embeddings(encoder)

    index = indexes.PLAID(
        index_folder,
        index_name=index_name,
        override=False,
        show_progress=False,
    )
    retriever = retrieve.ColBERT(index=index)
    return encoder, retriever


class LegalSearchTool:
    """Callable search tool for use inside ReAct (returns ids + previews)."""

    def __init__(
        self,
        encoder: models.ColBERT,
        retriever: retrieve.ColBERT,
        index_folder: epath.Path,
        k: int,
    ) -> None:
        self.encoder = encoder
        self.retriever = retriever
        self.index_folder = index_folder
        self.k = k
        self.__name__ = "search_legal_docs"

    def __call__(self, query: str, k: int | None = None) -> str:
        """Return formatted search results for the LM."""
        results = search_legal_docs_metadata(
            query=query,
            encoder=self.encoder,
            retriever=self.retriever,
            index_folder=self.index_folder,
            k=k or self.k,
        )
        if not results:
            return "No results."

        return "\n\n".join(
            f"[{idx}] id={res['id']} score={res['score']:.4f}\nPreview: {res['preview']}"
            for idx, res in enumerate(results)
        )


class LegalLookupTool:
    """Callable lookup tool to fetch full text by document id."""

    def __init__(self, index_folder: epath.Path) -> None:
        self.index_folder = index_folder
        self.__name__ = "lookup_legal_doc"

    def __call__(self, doc_id: str) -> str:
        return lookup_legal_doc(doc_id=doc_id, index_folder=self.index_folder)


class LegalReActAgent(dspy.Module):
    """DSPy ReAct agent with a retrieval tool."""

    def __init__(
        self,
        search_tool: LegalSearchTool,
        lookup_tool: LegalLookupTool,
        max_iters: int = 4,
        instructions: str = DEFAULT_INSTRUCTIONS,
    ) -> None:
        super().__init__()
        self.search_tool = search_tool
        self.lookup_tool = lookup_tool
        signature = dspy.Signature(
            "question -> answer:str",
            instructions=instructions,
        )
        self.react = dspy.ReAct(
            signature, tools=[self.search_tool, self.lookup_tool], max_iters=max_iters
        )

    def forward(self, question: str) -> dspy.Prediction:
        return self.react(question=question)


def build_agent(
    student_model: str = DEFAULT_STUDENT_MODEL,
    encoder_model: str = DEFAULT_ENCODER_MODEL,
    index_folder: epath.Path = DEFAULT_INDEX_FOLDER,
    index_name: str = DEFAULT_INDEX_NAME,
    search_k: int = DEFAULT_SEARCH_K,
    mode: LMMode = LMMode.HUGGINGFACE,
    api_key: str | None = None,
    api_base: str | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    instructions: str = DEFAULT_INSTRUCTIONS,
    max_iters: int = 4,
) -> LegalReActAgent:
    """Factory that wires LM, retrieval, and ReAct agent."""
    lm = build_language_model(
        student_model=student_model,
        mode=mode,
        api_key=api_key,
        api_base=api_base,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    dspy.configure(lm=lm)

    encoder, retriever = build_retrieval(
        encoder_model=encoder_model,
        index_folder=index_folder,
        index_name=index_name,
    )
    search_tool = LegalSearchTool(
        encoder=encoder,
        retriever=retriever,
        index_folder=index_folder,
        k=search_k,
    )
    lookup_tool = LegalLookupTool(index_folder=index_folder)
    return LegalReActAgent(
        search_tool,
        lookup_tool,
        max_iters=max_iters,
        instructions=instructions,
    )
