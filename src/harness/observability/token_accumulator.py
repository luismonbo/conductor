from __future__ import annotations

from dataclasses import dataclass

# Rough characters per token, for calls the provider never reports usage for
# (a call cancelled mid-stream). About right for English text.
_CHARS_PER_TOKEN = 4


def estimate_tokens(chars: int) -> int:
    """Rounded-up token estimate for `chars` characters of text."""
    return -(-chars // _CHARS_PER_TOKEN)


@dataclass
class TokenAccumulator:
    """Mutable accumulator — intentionally stateful; accumulates across LLM calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    iterations: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        """Accumulate tokens from an LLM call."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.iterations += 1

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.input_tokens + self.output_tokens
