"""
Tool 9: Arithmetic Calculator — simple expression evaluator.
"""

from typing import Dict, Any


def evaluate_arithmetic(expression: str) -> str:
    """
    Evaluate a simple arithmetic expression.

    Supports +, -, *, / and functions: exp, log, sqrt, sin, cos, round.

    Args:
        expression: The arithmetic expression to evaluate.

    Returns:
        Formatted string with the expression and its result.
    """
    from .legacy_tools.arithmetic import evaluate_arithmetic as _evaluate_arithmetic_impl
    return _evaluate_arithmetic_impl(expression)


TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "evaluate_arithmetic",
        "description": (
            "Evaluate a simple arithmetic expression. Supports basic math operations "
            "(+, -, *, /) and functions: exp, log, sqrt, sin, cos, round."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "The arithmetic expression to evaluate."}
            },
            "required": ["expression"],
            "additionalProperties": False,
        }
    }
}
