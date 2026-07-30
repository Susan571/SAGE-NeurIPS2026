from __future__ import annotations

import re
from typing import Callable, Sequence

import torch

from .structural_prior import semantic_entropy


def _last_boxed(text: str) -> str | None:
    starts = [match.start() for match in re.finditer(r"\\boxed\{", text)]
    for start in reversed(starts):
        depth = 0
        content_start = start + len(r"\boxed{")
        for index in range(content_start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                if depth == 0:
                    return text[content_start:index]
                depth -= 1
    return None


def canonical_answer(text: str) -> str:
    boxed = _last_boxed(text)
    lines = text.strip().splitlines()
    answer = boxed if boxed is not None else (lines[-1] if lines else "")
    if boxed is None:
        explicit = re.findall(
            r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*([^\n]+)",
            text,
            flags=re.IGNORECASE,
        )
        if explicit:
            answer = explicit[-1]
    return re.sub(r"\s+", " ", answer.strip().lower())


def deterministic_clusters(outputs: Sequence[str]) -> list[int]:
    ids: dict[str, int] = {}
    clusters = []
    for output in outputs:
        answer = canonical_answer(output)
        if answer not in ids:
            ids[answer] = len(ids)
        clusters.append(ids[answer])
    return clusters


def semantic_clusters(
    question: str,
    outputs: Sequence[str],
    equivalent: Callable[[str, str, str], bool],
) -> list[int]:
    representatives: list[str] = []
    assignments: list[int] = []
    for output in outputs:
        cluster = next(
            (
                idx
                for idx, representative in enumerate(representatives)
                if equivalent(question, representative, output)
            ),
            None,
        )
        if cluster is None:
            cluster = len(representatives)
            representatives.append(output)
        assignments.append(cluster)
    return assignments


def embedding_clusters(
    embeddings: torch.Tensor, similarity_threshold: float = 0.85
) -> list[int]:
    """Binary semantic-equivalence proxy over frozen reference embeddings."""
    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError("embeddings must be a non-empty rank-2 tensor")
    normalized = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    representatives: list[torch.Tensor] = []
    assignments: list[int] = []
    for embedding in normalized:
        cluster = next(
            (
                index
                for index, representative in enumerate(representatives)
                if float(embedding @ representative) >= similarity_threshold
            ),
            None,
        )
        if cluster is None:
            cluster = len(representatives)
            representatives.append(embedding)
        assignments.append(cluster)
    return assignments


def retain_by_entropy(cluster_ids: Sequence[int], low: float, high: float) -> bool:
    entropy = semantic_entropy(cluster_ids)
    return low < entropy < high
