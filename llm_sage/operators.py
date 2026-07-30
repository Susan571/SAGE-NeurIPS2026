from __future__ import annotations

import re
from enum import IntEnum


class MathOperator(IntEnum):
    SIMPLIFICATION = 0
    SUBSTITUTION = 1
    NUMERICAL_EVALUATION = 2
    EQUATION_FORMATION = 3
    FORMULA_INVOCATION = 4
    CASE_SPLIT = 5
    CONSTRAINT_CHECKING = 6
    FINAL_ANSWER = 7


class NaturalOperator(IntEnum):
    FACTUAL_RETRIEVAL = 0
    COMPARISON = 1
    ELIMINATION = 2
    AGGREGATION = 3
    INFERENCE = 4
    FORMAT_NORMALIZATION = 5
    FINAL_ANSWER = 6


_MATH_RULES = [
    (MathOperator.FINAL_ANSWER, r"\\boxed|final answer|therefore.*(?:answer|=)"),
    (MathOperator.CASE_SPLIT, r"\bcase(?:s)?\b|otherwise|if .* then"),
    (MathOperator.SUBSTITUTION, r"substitut|plug(?:ging)? in|replace .* with"),
    (MathOperator.CONSTRAINT_CHECKING, r"check|satisf(?:y|ies)|constraint|valid"),
    (MathOperator.FORMULA_INVOCATION, r"formula|theorem|identity|by (?:vieta|bayes)"),
    (MathOperator.EQUATION_FORMATION, r"(?:let|set) .*=|equation|we have"),
    (MathOperator.NUMERICAL_EVALUATION, r"calculat|evaluat|=\s*-?\d+(?:\.\d+)?"),
]

_NATURAL_RULES = [
    (NaturalOperator.FINAL_ANSWER, r"final answer|therefore|thus the answer"),
    (NaturalOperator.COMPARISON, r"compar|whereas|than|difference|similar"),
    (NaturalOperator.ELIMINATION, r"eliminat|rule out|cannot be|exclude"),
    (NaturalOperator.AGGREGATION, r"combine|overall|in sum|collectively"),
    (NaturalOperator.FORMAT_NORMALIZATION, r"format|json|schema|requested form"),
    (NaturalOperator.FACTUAL_RETRIEVAL, r"according to|known that|fact|recall"),
]


def classify_operator(text: str, task_family: str = "math") -> int:
    """Deterministic, label-free coarse operator parser used for pseudo-labels."""
    rules = _MATH_RULES if task_family == "math" else _NATURAL_RULES
    fallback = (
        MathOperator.SIMPLIFICATION
        if task_family == "math"
        else NaturalOperator.INFERENCE
    )
    lowered = text.lower()
    for operator, pattern in rules:
        if re.search(pattern, lowered, flags=re.DOTALL):
            return int(operator)
    return int(fallback)


def operator_count(task_family: str) -> int:
    return len(MathOperator) if task_family == "math" else len(NaturalOperator)
