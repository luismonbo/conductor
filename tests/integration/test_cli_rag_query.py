from __future__ import annotations

import pytest

from harness.cli.rag_query import run_query
from harness.config.settings import Settings


@pytest.mark.asyncio
async def test_run_query_returns_answer_against_empty_in_memory_index():
    settings = Settings(embedding_backend="fake", llm_backend="fake")

    result = await run_query(
        settings, "what is this about?", collection="papers",
        vector_store_backend="in_memory", k=3,
    )

    assert result.answer
    assert result.retrieved == ()
