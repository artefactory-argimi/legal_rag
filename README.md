# French Legal Agentic RAG

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/artefactory-argimi/legal_rag/blob/main/demo.py)

This project provides a specialized Retrieval-Augmented Generation (RAG) agent for answering complex questions about French law. It leverages a DSPy ReAct agent that can autonomously search a knowledge base of legal documents (the French Civil Code) to provide accurate, evidence-based answers.

The system is designed to be highly accessible and is best experienced through the interactive Google Colab notebook.

## About the ArGiMi Project

This demo is part of the ArGiMi project, an initiative for "Digital Commons for Generative Artificial Intelligence" supported by France 2030. The ArGiMi consortium, comprising Artefact, Giskard, Mistral AI, Institut national de l'audiovisuel (INA), and Bibliothèque nationale de France (BnF), was formed to develop specialized language models (LLMs) for business needs.

## Key Features

-   **Agentic Reasoning:** Uses a `dspy.ReAct` agent (`mistralai/Mistral-Small-3.1-24B-Instruct-2503`) that intelligently decides when to use its search tool to gather evidence.
-   **Specialized Retrieval:** Employs RAGatouille with a ColBERTv2 model (`maastrichtlawtech/colbert-legal-french`) trained specifically for French legal text, ensuring high-quality retrieval.
-   **Colab-Ready:** Designed for seamless execution in Google Colab, with helpers for environment setup, dependency management, and file uploads.
-   **Flexible Inference:** Supports two modes for Large Language Model (LLM) inference:
    -   **Remote:** Hugging Face Serverless Inference API (default, requires an HF token).
    -   **Local:** Any OpenAI-compatible server (e.g., SGLang, vLLM).
-   **Modular Codebase:** A clean, organized structure that separates concerns for indexing, agent logic, and tools.

## Architecture

The system is orchestrated by the DSPy framework. When a user asks a question, it is passed to a ReAct agent. The agent can iteratively use a `search_legal_docs` tool to query a ColBERT index of legal documents. The retrieved documents provide the necessary context for the agent to reason and generate a final answer grounded in legal text.

## Getting Started in Google Colab

The easiest way to run this demo is by using the provided Google Colab notebook.

### Step 1: Open the Notebook

Click the "Open in Colab" badge at the top of this README to launch the interactive demo.

### Step 2: Prepare the Search Index

The agent requires a pre-built ColBERT search index to function.

1.  When you execute the **"Index loading"** cell in the notebook, it will check if the index exists.
2.  If the index is not found, a file upload prompt will appear.
3.  Please upload your zipped index file (e.g., `legal_rag_index.zip`). The notebook will automatically unzip it to the correct location (`/content/index`).

> **Note:** The process for building the index from raw legal texts is defined in `src/legal_rag/indexer.py` but is not part of the interactive demo, which focuses on inference.

### Step 3: Configure the LLM Provider

The notebook can use either the Hugging Face API or a custom OpenAI-compatible endpoint.

-   **To use Hugging Face (Default):**
    -   Leave the `GENERATOR_API_BASE` field in the form empty.
    -   If you do not provide a Hugging Face token in the `GENERATOR_API_KEY` field, a login prompt will appear when you run the **"Hugging Face login"** cell.

-   **To use a Custom Server:**
    -   Fill in the `GENERATOR_API_BASE` with your server's URL (e.g., `http://localhost:30000/v1`).

### Step 4: Ask Questions

1.  Navigate to the final cell, **"Ask questions"**.
2.  Modify the `queries` list with your own legal questions in French.
3.  Run the cell to see the agent retrieve documents and generate grounded answers.

## Local Development

While Colab is the recommended environment for this demo, you can also run the agent locally.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/artefactory-argimi/legal_rag.git
    cd legal_rag
    ```
2.  **Install dependencies:**
    ```bash
    pip install -e .
    ```
3.  **Provide the index:** Ensure you have a search index available at the path specified in `demo.py` (defaults to `./index`), or update the path accordingly.
4.  **Configure environment:** Set your `HF_TOKEN` as an environment variable or modify the `demo.py` script to point to your local inference server.
5.  **Run the notebook:** Use a tool like VS Code with the Jupyter extension to run the `demo.py` notebook.

## Codebase Structure

The project is organized into the following key components:

-   `demo.py`: The main interactive Jupytext notebook for running the agent.
-   `src/legal_rag/`: The core Python source code.
    -   `agent.py`: Defines the DSPy `ReAct` agent and its configuration.
    -   `tools.py`: Implements the `search_legal_docs` tool for the agent.
    -   `indexer.py`: Contains the offline logic for building the ColBERT index.
    -   `colbert_utils.py`: Utilities for working with the ColBERT index.
-   `pyproject.toml`: Project metadata and dependencies.
