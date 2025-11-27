# Design Document: French Legal Agentic RAG

## Overview

### Objective
This document outlines the design and implementation of a specialized Legal RAG (Retrieval-Augmented Generation) agent for answering complex questions related to French law. The system uses an Agentic ReAct architecture to autonomously retrieve and reason over the French Civil Code and other legal texts.

### Goals
- Provide accurate, context-aware answers to French legal queries.
- Implement a ReAct agent capable of autonomously deciding when to search its knowledge base.
- Ensure the system is easily deployable and runnable within a Google Colab environment.
- Create a modular codebase that separates indexing, inference, and evaluation.
- Persist critical artifacts (e.g., search index, source code, configurations) on Google Drive to ensure statefulness across sessions.

### Non-Goals
- A production-grade, highly available service; the focus is a robust demonstration within the Colab ecosystem.
- A user-facing graphical web application; the primary interface will be an interactive Python notebook.
- Support for legal domains outside of French law.

### Key Architectural Principles
- Hybrid deployment with two modes for flexibility and resource management:
  1. Remote mode, activated by an `HF_TOKEN`, uses the Hugging Face Inference API to minimize local GPU memory consumption.
  2. Local mode (default) launches a local SGLang inference server on the Colab instance.
- Persistence on Google Drive for all generated artifacts, including the search index, compiled agent programs, and source code.
- Separation of concerns between offline (indexing) and online (agent inference) processes.
- Modularity via distinct Python modules imported by the main notebook.
- Preservation of legal nuance by returning full-text legal documents rather than snippets to maintain legal accuracy.

## System Architecture

### High-Level Diagram Description
The system is orchestrated by the DSPy framework. A user query is first processed by the ReAct agent (student model). The agent can use its retrieval tool to search the ColBERT index. The retrieval tool, powered by RAGatouille, queries the index and returns relevant legal documents. The agent uses this retrieved context to reason and generate a final, evidence-based answer.

An offline optimization process uses a teacher model to generate synthetic data and compile the agent's program for improved performance.

### Core Components

- Inference layer
  - Student model (agent): `mistralai/Magistral-Small-2509`, chosen for strong French-language reasoning in a manageable size.
  - Teacher model (optimizer): `openai/gpt-oss-120b`, used to generate high-quality synthetic data for agent training and optimization.
  - Provider switching: DSPy (via LiteLLM) handles routing. Use `huggingface/<model>` with `HF_TOKEN` when no `api_base` is provided; set an OpenAI-compatible `api_base` to route to a local server. Default stays on the mistral student model; switching happens inside DSPy based solely on the presence of `api_base` and credentials.
- Retrieval layer
  - Engine: RAGatouille, implementing the ColBERTv2 algorithm for dense retrieval.
  - Embedding model: `maastrichtlawtech/colbert-legal-french`, specialized for French legal terminology.
  - Index storage: Persisted on Google Drive and loaded into memory at runtime for fast access.
- Data layer
  - Data source: `AgentPublic/legi` (French Civil Law corpus) and `maastrichtlawtech/colbert-legal-french`.
  - ETL: Google Grain for deterministic and reproducible data loading during the indexing phase.
- Orchestration layer
  - Framework: DSPy to define, optimize, and serve the agent; manages the ReAct loop, tool integration, and prompt compilation via its teleprompters.

## Codebase and Module Design

### Project Structure
All Python source code resides in a dedicated folder on Google Drive (e.g., `/content/drive/MyDrive/rag_project/`) and is added to the Python path at runtime.

### Module Breakdown

- `config.py` (configuration)
  - Purpose: Single source of truth for all system settings.
  - Implementation: Uses `attrs` and `environ-config` to load configuration; auto-detects `HF_TOKEN` and the Google Drive path to configure `USE_REMOTE_API` and `INDEX_ROOT`; defines constants like `SEARCH_K`, model IDs, and paths.
- `scripts/indexer.py` (offline corpus indexing)
  - Purpose: Builds the RAGatouille/ColBERT search index; run once or when the data source is updated.
  - Implementation: Checks for a pre-existing index on Google Drive and offers a fast track to unzip it if available; otherwise streams data via Google Grain and builds the index from scratch.
- `tools.py` (agent retrieval tools)
  - Purpose: Provides the agent with callable functions to interact with the retrieval layer.
  - Implementation: Implements a singleton pattern to load the RAGatouille index into memory only once; exposes `search_legal_docs(query: str)` to return the full text of the top `K` documents.
- `agent.py` (core ReAct agent)
  - Purpose: Defines the cognitive architecture of the agent.
  - Implementation: Contains a `dspy.ReAct` module with the signature `question -> answer`; the DSPy language model (`dspy.LM`) is configured dynamically based on the `USE_REMOTE_API` flag.
- `evaluate.py` (agent evaluation)
  - Purpose: Measures agent performance against a predefined benchmark.
  - Implementation: Loads `test.jsonl`; runs the agent (zero-shot MVP or optimized version) against test questions and calculates metrics, including answer accuracy and tool-use correctness.
- `optimize.py` (agent compilation and optimization)
  - Purpose: Improves the agent's prompts and logic.
  - Implementation:
    1. Data generation connects to the teacher LLM to create a synthetic `train.jsonl` dataset for training.
    2. Compilation uses the `dspy.MIPROv2` teleprompter to run the optimization process and saves the compiled agent as `optimized_agent.json` to Google Drive.
- `main.ipynb` (interactive deployment notebook)
  - Purpose: Main entry point and user interface for interacting with the agent.
  - Implementation: Mounts Google Drive, adds the project to `sys.path`, conditionally launches the SGLang server, loads the desired agent (zero-shot or compiled), and runs an interactive chat loop.

## Implementation Plan

### Phase 1: Minimum Viable Product (MVP)
Goal: A functioning interactive chatbot on Colab that can search the legal index and answer questions using zero-shot reasoning.

- Tasks:
  1. Project setup: Create `config.py` with logic to detect Drive, `HF_TOKEN`, and define all paths and model IDs.
  2. Indexing: Implement `scripts/indexer.py` with Google Grain for data loading and ColBERT for indexing; include logic to use a pre-built zipped index.
  3. Tooling: Implement `tools.py` with the singleton index loader and the `search_legal_docs` tool.
  4. Agent: Implement `agent.py` with a zero-shot `dspy.ReAct` module.
  5. Deployment: Create `main.ipynb` to mount Drive, launch the SGLang server if needed, and run the interactive chat.

### Phase 2: MVP Evaluation
Goal: Quantify the baseline performance of the zero-shot MVP agent.

- Tasks:
  1. Test set creation: Manually or semi-automatically create a high-quality `test.jsonl` file containing representative legal questions and reference answers.
  2. Evaluation script: Implement `evaluate.py`, load the MVP agent and the test set, run the evaluation, and report baseline metrics for answer quality and tool usage.

### Phase 3: Optimization and Tuning
Goal: Improve the agent's reasoning and tool-use performance through automated compilation and then verify the improvement.

- Tasks:
  1. Synthetic data generation: Implement data generation within `optimize.py` to create a `train.jsonl` file using the teacher LLM.
  2. Metric definition: Define a validation metric within the optimizer that aligns with the metrics used in `evaluate.py`.
  3. Agent compilation: Implement and run the `dspy.MIPROv2` optimization loop in `optimize.py`; save the resulting `optimized_agent.json` to Drive.
  4. Comparative evaluation: Re-run `evaluate.py`, loading `optimized_agent.json`, and compare performance against the MVP baseline to quantify improvement.
