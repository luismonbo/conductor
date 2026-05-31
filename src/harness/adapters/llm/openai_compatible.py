"""OpenAI-compatible LLM adapter.

One adapter for every server that speaks the OpenAI Chat Completions API —
llama.cpp `llama-server` (local default), vLLM, Ollama, or any compatible
endpoint. The engine is selected purely by `base_url`, which is why a
dev-local Gemma on Metal and a future NVIDIA vLLM deployment need no code
change between them.

generate() and model_id are inherited from _OpenAIBaseClient (openai_wire.py).
The `client` parameter exists for tests (inject a stand-in); in normal use it
is left None and a real AsyncOpenAI is built lazily so the `openai` SDK is only
required when this backend is actually selected.
"""
from __future__ import annotations

from typing import Any

from harness.adapters.llm.openai_wire import _OpenAIBaseClient
from harness.core.llm.tool_parsing import ToolCallParser

# Most local servers (llama-server) ignore auth, but the SDK rejects an empty
# api_key, so we substitute a harmless placeholder when none is configured.
_PLACEHOLDER_KEY = "sk-no-key-required"


class OpenAICompatibleClient(_OpenAIBaseClient):
    def __init__(
        self,
        base_url: str,
        model: str,
        parser: ToolCallParser,
        api_key: str = "",
        temperature: float = 0.0,
        client: Any | None = None,
    ) -> None:
        if client is None:
            # Imported lazily so core/tests don't require the SDK installed.
            from openai import AsyncOpenAI

            client = AsyncOpenAI(base_url=base_url, api_key=api_key or _PLACEHOLDER_KEY)
        super().__init__(client, model, parser, temperature)
