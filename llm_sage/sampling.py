from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import torch


@dataclass
class CandidateStep:
    text: str
    token_ids: torch.Tensor
    token_logprobs: torch.Tensor
    next_hidden: torch.Tensor
    terminal: bool = False

    @property
    def length_normalized_logprob(self) -> torch.Tensor:
        return self.token_logprobs.mean()


def split_reasoning_steps(text: str) -> list[str]:
    """Deterministic implementation of the Appendix E.2 step boundaries."""
    pieces = re.findall(r".*?(?:\n+|(?<=\.)\s+|$)", text, flags=re.DOTALL)
    return [piece for piece in pieces if piece.strip()]


def guided_candidate_probabilities(
    candidates: Sequence[CandidateStep],
    psi_p: torch.Tensor,
    psi_h: torch.Tensor,
    lambda_psi: float,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    """Finite-candidate reweighting rule from Appendix E.2."""
    if not candidates:
        raise ValueError("at least one candidate is required")
    base = torch.stack([candidate.length_normalized_logprob for candidate in candidates])
    potential = alpha * psi_p.to(base) + gamma * psi_h.to(base)
    return torch.softmax(base + lambda_psi * potential, dim=0)


def choose_candidate(
    candidates: Sequence[CandidateStep],
    psi_p: torch.Tensor,
    psi_h: torch.Tensor,
    lambda_psi: float,
    alpha: float = 1.0,
    gamma: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[CandidateStep, torch.Tensor]:
    probabilities = guided_candidate_probabilities(
        candidates, psi_p, psi_h, lambda_psi, alpha, gamma
    )
    index = int(torch.multinomial(probabilities, 1, generator=generator))
    return candidates[index], probabilities
