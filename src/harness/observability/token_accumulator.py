from __future__ import annotations

from dataclasses import dataclass


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
