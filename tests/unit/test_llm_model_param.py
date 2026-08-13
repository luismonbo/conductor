"""The port accepts a per-call model; None preserves the constructed default."""
from types import SimpleNamespace

from harness.adapters.llm.fake import FakeLLMClient
from harness.adapters.llm.openai_compatible import OpenAICompatibleClient
from harness.adapters.llm.parsers import NativeToolCallParser
from harness.core.types import LLMResponse, Message, Role


class _RecordingCompletions:
    def __init__(self) -> None:
        self.kwargs: list[dict] = []

    async def create(self, **kwargs):
        self.kwargs.append(kwargs)
        msg = SimpleNamespace(content="ok", tool_calls=None)
        choice = SimpleNamespace(message=msg, finish_reason="stop")
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return SimpleNamespace(choices=[choice], usage=usage)


def _client_with_recorder():
    completions = _RecordingCompletions()
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client = OpenAICompatibleClient(
        base_url="http://x/v1", model="default-model",
        parser=NativeToolCallParser(), client=sdk,
    )
    return client, completions


async def test_generate_uses_constructed_model_when_none():
    client, completions = _client_with_recorder()
    await client.generate([Message(role=Role.USER, content="hi")])
    assert completions.kwargs[0]["model"] == "default-model"


async def test_generate_uses_per_call_model_when_given():
    client, completions = _client_with_recorder()
    await client.generate([Message(role=Role.USER, content="hi")], model="claude")
    assert completions.kwargs[0]["model"] == "claude"


async def test_fake_records_requested_models():
    fake = FakeLLMClient([LLMResponse(text="a"), LLMResponse(text="b")])
    await fake.generate([Message(role=Role.USER, content="1")], model="gpt")
    async for _ in fake.stream([Message(role=Role.USER, content="2")], model="claude"):
        pass
    assert fake.requested_models == ["gpt", "claude"]
