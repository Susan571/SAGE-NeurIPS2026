from __future__ import annotations

import re
from fractions import Fraction

import torch

from .filtering import canonical_answer


def _normalized_scalar(text: str) -> str:
    value = canonical_answer(text)
    is_percent = value.rstrip().endswith(r"\%") or value.rstrip().endswith("%")
    value = value.replace(",", "").replace("$", "").replace(r"\%", "").replace("%", "")
    value = re.sub(
        r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", value
    )
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = re.sub(r"^(?:answer\s*:|is\s+)", "", value).strip(" .")
    fraction = re.fullmatch(r"\(([-+]?\d+)\)/\(([-+]?\d+)\)", value)
    if fraction:
        value = f"{fraction.group(1)}/{fraction.group(2)}"
    try:
        number = Fraction(value)
        return str(number / 100 if is_percent else number)
    except (ValueError, ZeroDivisionError):
        return value


def exact_answer_reward(completion: str, answer: str) -> float:
    """Terminal-only exact/canonical answer reward for closed-form tasks."""
    return float(_normalized_scalar(completion) == _normalized_scalar(answer))


def multiple_choice_reward(completion: str, answer: str) -> float:
    canonical = canonical_answer(completion).upper().strip()
    explicit = re.findall(
        r"(?:FINAL\s+ANSWER|ANSWER)\s*(?:IS|:)?\s*[\(\[]?([A-Z])[\)\]]?",
        completion.upper(),
    )
    trailing = re.search(r"(?:^|[\s(\[])*([A-Z])[\s)\].]*$", canonical)
    predicted = explicit[-1] if explicit else (trailing.group(1) if trailing else "")
    return float(predicted == answer.strip().upper())


def sage_trajectory_reward(
    terminal_reward: float | torch.Tensor,
    step_potentials: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    """Equation (E.2): terminal outcome plus mean trajectory potential."""
    terminal = torch.as_tensor(
        terminal_reward, dtype=step_potentials.dtype, device=step_potentials.device
    )
    structural = (
        step_potentials.mean()
        if step_potentials.numel()
        else torch.zeros((), dtype=terminal.dtype, device=terminal.device)
    )
    return terminal + eta * structural
