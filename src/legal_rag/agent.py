"""DSPy-based ReAct agent wiring for the Legal RAG demo (new dspy API)."""

from functools import partial
from pathlib import Path

import dspy
from datasets import load_dataset
from etils import epath

from legal_rag.retriever import build_encoder, build_retriever
from legal_rag.tools import DEFAULT_DOC_ID_COLUMN, lookup_legal_doc, search_legal_docs

# Defaults aligned with the design doc; adjust via function arguments as needed.
DEFAULT_GENERATOR_MODEL = "mistralai/Magistral-Small-2509"
DEFAULT_ENCODER_MODEL = "maastrichtlawtech/colbert-legal-french"
DEFAULT_INDEX_FOLDER = epath.Path("./index")
DEFAULT_INDEX_NAME = "legal_french_index"
DEFAULT_SEARCH_K = 5
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.2
DEFAULT_DATASET = "artefactory/Argimi-Legal-French-Jurisprudence"
DEFAULT_CONFIG = "juri"
DEFAULT_SPLIT = "train"
DEFAULT_INSTRUCTIONS = (
    "First call search_legal_docs to find candidate ids and previews. "
    "Then call lookup_legal_doc on specific ids you want to read in full. "
    "Ground your answer in the retrieved text and cite the document ids you used."
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
    """DSPy ReAct agent with a retrieval tool."""

    def __init__(
        self,
        search_tool,
        lookup_tool,
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
        prediction = self.react(question=question)
        trajectory = prediction.trajectory
        documents: dict[str, str] = {}
        idx = 0
        while True:
            name_key = f"tool_name_{idx}"
            args_key = f"tool_args_{idx}"
            obs_key = f"observation_{idx}"
            if name_key not in trajectory:
                break
            if trajectory.get(name_key) == "lookup_legal_doc":
                args = trajectory.get(args_key) or {}
                if isinstance(args, dict):
                    doc_id = args.get("doc_id") or args.get("id")
                else:
                    doc_id = None
                observation = trajectory.get(obs_key)
                if isinstance(doc_id, str) and isinstance(observation, str):
                    documents[doc_id] = observation
            idx += 1
        prediction.documents = documents
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
    max_iters: int = 4,
) -> LegalReActAgent:
    """Factory that wires LM, retrieval, and ReAct agent."""
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

    lookup_tool = partial(
        lookup_legal_doc,
        dataset=dataset,
        doc_id_column=DEFAULT_DOC_ID_COLUMN,
    )
    lookup_tool.__name__ = "lookup_legal_doc"
    search_tool = partial(
        search_legal_docs,
        encoder=encoder,
        retriever=retriever,
        k=search_k,
    )
    search_tool.__name__ = "search_legal_docs"
    return LegalReActAgent(
        search_tool,
        lookup_tool,
        max_iters=max_iters,
        instructions=instructions,
    )
