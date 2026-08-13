"""repeat_last=True makes the served fake inexhaustible; default stays scripted."""
from harness.adapters.llm.fake import FakeLLMClient
from harness.core.types import LLMResponse, Message, Role


def _msg() -> list[Message]:
    return [Message(role=Role.USER, content="hi")]


async def test_default_exhausts_after_scripted_responses():
    fake = FakeLLMClient([LLMResponse(text="one")])
    assert (await fake.generate(_msg())).text == "one"
    assert (await fake.generate(_msg())).text == "(no scripted response left)"


async def test_repeat_last_replays_final_response_forever():
    fake = FakeLLMClient([LLMResponse(text="one"), LLMResponse(text="two")], repeat_last=True)
    assert (await fake.generate(_msg())).text == "one"
    assert (await fake.generate(_msg())).text == "two"
    assert (await fake.generate(_msg())).text == "two"
    items = [item async for item in fake.stream(_msg())]
    assert items[-1].text == "two"
