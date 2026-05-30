"""Calculator tool: deterministic, no network — ideal for testing the loop and
for giving a small model an easy, reliable first tool to call."""
from __future__ import annotations

import ast
import operator
from typing import Any

from harness.core.types import ToolSpec

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("non-numeric constant")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


class CalculatorTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="calculator",
            description="Evaluate a basic arithmetic expression and return the result.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression, e.g. '3 * (4 + 2)'",
                    }
                },
                "required": ["expression"],
            },
        )

    async def run(self, arguments: dict[str, Any]) -> str:
        expr = arguments["expression"]
        tree = ast.parse(expr, mode="eval")
        return str(_safe_eval(tree.body))
