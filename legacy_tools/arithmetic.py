import math

from simpleeval import simple_eval


def evaluate_arithmetic(expression: str) -> str:
    """Evaluate a simple arithmetic expression.

    Args:
        expression (str): The arithmetic expression to evaluate.

    Examples:
        >>> evaluate_arithmetic("2 + 2 * 3")
        >>> evaluate_arithmetic("sin(0) + cos(0)")
        >>> evaluate_arithmetic("log(10) + exp(1)")
        >>> evaluate_arithmetic("sqrt(16) + round(3.6)")

    Returns:
        str: A formatted string containing the evaluated expression and its result.
    """

    functions = {
        "exp": math.exp,
        "log": math.log,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "round": round,
    }

    try:
        result = float(simple_eval(expression, functions=functions))
        return f"Arithmetic Evaluation Result:\n- expression: {expression}\n- result: {result}"
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression '{expression}': {e}") from e


ARITHMETIC_OPENAI_TOOLS = [{
    'type': 'function',
    'function': {
        'name': 'evaluate_arithmetic',
        'description': 'Evaluate a simple arithmetic expression. Supports basic math operations (+, -, *, /) and functions like exp, log, sqrt, sin, cos, and round. Examples of valid expressions: "2 + 2 * 3", "sin(0) + cos(0)", "log(10) + exp(1)", "sqrt(16) + round(3.6)".',
        'parameters': {
            'type': 'object',
            'properties': {
                'expression': {
                    'type': 'string',
                    'description': 'The arithmetic expression to evaluate.'
                }
            },
            'required': [
                'expression'
            ]
        }
    }
}]


if __name__ == "__main__":
    print(evaluate_arithmetic("-2.72 + 0.71*1.2 - 0.0061*432"))