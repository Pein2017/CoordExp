from __future__ import annotations

from src.tokens.row_offsets import (
    TOKEN_EMBEDDINGS_ADAPTER_NAME,
    TokenEmbeddingsAdapter,
    install_token_embeddings_adapter,
    reattach_token_embeddings_adapter_hooks,
)


def test_row_offset_module_exposes_token_embeddings_adapter_surface():
    assert TokenEmbeddingsAdapter.__name__ == "TokenEmbeddingsAdapter"
    assert TOKEN_EMBEDDINGS_ADAPTER_NAME == "token_embeddings_adapter"
    assert callable(install_token_embeddings_adapter)
    assert callable(reattach_token_embeddings_adapter_hooks)
