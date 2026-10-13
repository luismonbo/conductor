"""Errors an LLM adapter raises in the domain's own terms.

Adapters translate provider exceptions into these, so the loop and the API
recognise a failure without importing a vendor SDK (see core/types.py).
"""
from __future__ import annotations


class ContentFiltered(Exception):
    """The provider's content filter refused the prompt or cut the answer off.

    `stage` is "prompt" (the request was refused) or "completion" (the answer
    was cut off). `categories` names what tripped (e.g. "jailbreak") when the
    provider says; it is for the server log, never for a browser.
    """

    def __init__(self, stage: str, categories: tuple[str, ...] = ()) -> None:
        self.stage = stage
        self.categories = categories
        super().__init__(
            f"content filter ({stage}): {', '.join(categories) or 'unspecified'}"
        )
