from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenAccumulator:
    input_tokens: int = 0
    output_tokens: int = 0
    iterations: int = 0

    def add(self, input: int, output: int) -> None:
        self.input_tokens += input
        self.output_tokens += output
        self.iterations += 1
