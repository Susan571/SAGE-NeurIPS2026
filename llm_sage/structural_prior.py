from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .operators import classify_operator, operator_count


def radial_project(z: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    """Appendix E.3 radial projection into the Poincare ball."""
    if curvature <= 0:
        raise ValueError("curvature must be positive")
    norm = torch.linalg.vector_norm(z, dim=-1, keepdim=True).clamp_min(1e-12)
    root_c = curvature**0.5
    projected = torch.tanh(root_c * norm) * z / (root_c * norm)
    # tanh(x) rounds to exactly 1 for large x in finite precision. Keep the
    # result strictly inside the open ball so the distance denominator is valid.
    projected_norm = torch.linalg.vector_norm(
        projected, dim=-1, keepdim=True
    ).clamp_min(1e-12)
    radius = (1.0 - 1e-6) / root_c
    return projected * torch.clamp(radius / projected_norm, max=1.0)


def poincare_distance(x: torch.Tensor, y: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    """Poincare-ball geodesic distance with curvature -c."""
    x2 = (x * x).sum(dim=-1)
    y2 = (y * y).sum(dim=-1)
    delta2 = ((x - y) ** 2).sum(dim=-1)
    denom = (1.0 - curvature * x2).clamp_min(1e-7) * (
        1.0 - curvature * y2
    ).clamp_min(1e-7)
    argument = (1.0 + 2.0 * curvature * delta2 / denom).clamp_min(1.0)
    return torch.acosh(argument) / curvature**0.5


def semantic_entropy(cluster_ids: Sequence[int]) -> float:
    """Appendix E.5 entropy of meaning-cluster frequencies."""
    if not cluster_ids:
        raise ValueError("cluster_ids must not be empty")
    ids = torch.as_tensor(cluster_ids, dtype=torch.long)
    counts = torch.bincount(ids).float()
    probabilities = counts[counts > 0] / len(cluster_ids)
    return float(-(probabilities * probabilities.log()).sum())


def pseudo_label(prefix: str, candidate: str, task_family: str) -> torch.Tensor:
    """Construct rollout-local labels without correctness or outcome supervision.

    The vector contains operator type, lexical constraint coverage,
    answer-schema status, and format consistency, matching E.4 at an
    implementable level. Dataset adapters may provide richer parsers.
    """
    n_ops = operator_count(task_family)
    label = torch.zeros(n_ops + 3)
    label[classify_operator(candidate, task_family)] = 1.0
    prefix_terms = set(prefix.lower().split())
    candidate_terms = set(candidate.lower().split())
    label[n_ops] = len(prefix_terms & candidate_terms) / max(len(prefix_terms), 1)
    label[n_ops + 1] = float(
        "\\boxed" in candidate
        or "final answer" in candidate.lower()
        or "answer:" in candidate.lower()
    )
    label[n_ops + 2] = float(candidate.count("(") == candidate.count(")"))
    return label


@dataclass
class LabelFreeStructuralPrior:
    """Frozen Appendix E.3-E.4 prior fitted only from training rollouts."""

    projection: torch.Tensor
    feature_mean: torch.Tensor
    residual_probe: torch.Tensor
    subspaces: list[torch.Tensor]
    anchor: torch.Tensor
    task_family: str
    curvature: float = 1.0
    kappa: float = 1.0
    epsilon: float = 1e-8
    prompt_anchors: dict[str, torch.Tensor] | None = None

    @classmethod
    def fit(
        cls,
        hidden_states: torch.Tensor,
        prefixes: Sequence[str],
        steps: Sequence[str],
        task_family: str,
        structural_dim: int = 64,
        ridge_xi: float = 1e-3,
        subspace_rank: int = 4,
        curvature: float = 1.0,
        kappa: float = 1.0,
        epsilon: float = 1e-8,
        prompt_anchor_hidden_states: dict[str, torch.Tensor] | None = None,
    ) -> "LabelFreeStructuralPrior":
        if hidden_states.ndim != 2 or len(hidden_states) != len(prefixes) or len(prefixes) != len(steps):
            raise ValueError("hidden_states, prefixes, and steps must have the same batch length")
        feature_mean = hidden_states.float().mean(0)
        d_e = min(structural_dim, *hidden_states.shape)
        _, _, principal_directions = torch.pca_lowrank(
            hidden_states.float(), q=d_e, center=True
        )
        projection = principal_directions.T

        labels = torch.stack(
            [pseudo_label(prefix, step, task_family) for prefix, step in zip(prefixes, steps)]
        ).to(hidden_states)
        probe_features = hidden_states.float()
        # Equivalent primal/dual ridge solutions; the dual form avoids a
        # hidden_size x hidden_size inverse for the usual n_rollouts << d_h.
        if probe_features.shape[0] <= probe_features.shape[1]:
            eye = torch.eye(
                probe_features.shape[0],
                device=probe_features.device,
                dtype=probe_features.dtype,
            )
            probe = labels.T @ torch.linalg.solve(
                probe_features @ probe_features.T + ridge_xi * eye,
                probe_features,
            )
        else:
            eye = torch.eye(
                probe_features.shape[1],
                device=probe_features.device,
                dtype=probe_features.dtype,
            )
            probe = labels.T @ probe_features @ torch.linalg.solve(
                probe_features.T @ probe_features + ridge_xi * eye, eye
            )
        residuals = probe_features @ probe.T

        types = [classify_operator(step, task_family) for step in steps]
        subspaces: list[torch.Tensor] = []
        for operator in range(operator_count(task_family)):
            selected = residuals[torch.tensor(types, device=residuals.device) == operator]
            if len(selected) < 2:
                subspaces.append(torch.zeros(residuals.shape[1], 0, device=residuals.device))
                continue
            centered = selected - selected.mean(dim=0, keepdim=True)
            _, singular_values, op_vh = torch.linalg.svd(
                centered, full_matrices=False
            )
            nonzero_rank = int(
                (singular_values > torch.finfo(singular_values.dtype).eps).sum()
            )
            rank = min(subspace_rank, nonzero_rank)
            subspaces.append(op_vh[:rank].T)

        # Fallback anchor; scores() accepts a prompt-conditioned anchor hidden state.
        anchor = radial_project(feature_mean @ projection.T, curvature)
        prompt_anchors = None
        if prompt_anchor_hidden_states:
            prompt_anchors = {
                prompt: radial_project(
                    hidden.float() @ projection.T, curvature
                )
                for prompt, hidden in prompt_anchor_hidden_states.items()
            }
        return cls(
            projection,
            feature_mean,
            probe,
            subspaces,
            anchor,
            task_family,
            curvature,
            kappa,
            epsilon,
            prompt_anchors,
        )

    def anchor_for(self, prompt: str) -> torch.Tensor:
        if self.prompt_anchors and prompt in self.prompt_anchors:
            return self.prompt_anchors[prompt]
        return self.anchor

    def scores(
        self,
        state_hidden: torch.Tensor,
        next_hidden: torch.Tensor,
        candidates: Sequence[str],
        anchor_hidden: torch.Tensor | None = None,
        anchor_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.residual_probe @ state_hidden.float()
        denom = residual.square().sum().clamp_min(self.epsilon)
        psi_p = []
        for candidate in candidates:
            basis = self.subspaces[classify_operator(candidate, self.task_family)]
            projected = basis @ (basis.T @ residual) if basis.numel() else torch.zeros_like(residual)
            psi_p.append(projected.square().sum() / denom)
        z_next = next_hidden.float() @ self.projection.T
        embedded = radial_project(z_next, self.curvature)
        if anchor_embedding is not None:
            anchor = anchor_embedding
        elif anchor_hidden is None:
            anchor = self.anchor
        else:
            anchor = radial_project(
                anchor_hidden.float() @ self.projection.T,
                self.curvature,
            )
        anchor = anchor.expand_as(embedded)
        psi_h = torch.exp(-poincare_distance(embedded, anchor, self.curvature) / self.kappa)
        return torch.stack(psi_p), psi_h

    def state_dict(self) -> dict:
        return {
            "projection": self.projection,
            "feature_mean": self.feature_mean,
            "residual_probe": self.residual_probe,
            "subspaces": self.subspaces,
            "anchor": self.anchor,
            "task_family": self.task_family,
            "curvature": self.curvature,
            "kappa": self.kappa,
            "epsilon": self.epsilon,
            "prompt_anchors": self.prompt_anchors,
        }
