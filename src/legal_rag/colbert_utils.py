"""
Utilities for ColBERT models used in the Legal RAG demo.
"""


def fix_colbert_embeddings(model):
    """
    Fix the token embedding size issue in ColBERT models.

    The bug: PyLate adds special tokens ([Q], [D]) but doesn't always
    properly resize the embedding layer, causing token IDs to be out of bounds.

    Some models (e.g., Jina ColBERT v2) use custom architectures that don't
    support get_input_embeddings. These models handle their own token
    embeddings and don't need this fix.
    """
    # Get current sizes
    vocab_size = len(model.tokenizer)
    try:
        embedding_layer = model[0].auto_model.get_input_embeddings()
    except NotImplementedError:
        print("✓ Model uses custom embeddings, skipping embedding fix")
        return model
    embedding_size = embedding_layer.num_embeddings

    # Get special token IDs
    query_id = model.query_prefix_id
    doc_id = model.document_prefix_id

    print(f"Tokenizer vocab size: {vocab_size}")
    print(f"Embedding layer size: {embedding_size}")
    print(f"Query prefix '{model.query_prefix}': ID {query_id}")
    print(f"Document prefix '{model.document_prefix}': ID {doc_id}")

    # Calculate required size
    max_token_id = max(vocab_size - 1, query_id, doc_id)
    required_size = max_token_id + 1

    # Resize if needed
    if required_size > embedding_size:
        print("\n⚠️  Token IDs exceed embedding size!")
        print(f"Resizing from {embedding_size} to {required_size}")
        model[0].auto_model.resize_token_embeddings(required_size)
        new_size = model[0].auto_model.get_input_embeddings().num_embeddings
        print(f"✓ Resized to {new_size}")

        # Verify fix
        assert query_id < new_size, f"Query ID {query_id} still out of bounds!"
        assert doc_id < new_size, f"Document ID {doc_id} still out of bounds!"
        print("✓ All token IDs are now valid")
    else:
        print("✓ Embedding size is already sufficient")

    return model
