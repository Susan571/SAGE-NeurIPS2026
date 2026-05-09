"""
SAGE guidance utilities for ASC validity masking and TNSC topological potentials.
"""

import numpy as np

from trex.ac_solver.envs.ac_moves import ACMove
from trex.ac_solver.envs.utils import generate_trivial_states


def _compute_lengths(presentation: np.ndarray, n_gen: int = 2):
    """
    Compute relator lengths for a balanced presentation with `n_gen` generators.
    """
    max_relator_length = len(presentation) // n_gen
    lengths = [
        int(
            np.count_nonzero(
                presentation[i * max_relator_length : (i + 1) * max_relator_length]
            )
        )
        for i in range(n_gen)
    ]
    return lengths, max_relator_length


def build_trivial_targets(max_relator_length: int) -> np.ndarray:
    """
    Precompute trivial target states used for depth potential Ψ_H.
    """
    return generate_trivial_states(max_relator_length=max_relator_length)


def sage_validity_and_potentials_for_state(
    state: np.ndarray,
    trivial_targets: np.ndarray,
    lambda_width: float = 1.0,
    lambda_depth: float = 1.0,
):
    """
    For a single AC presentation `state`, compute:
    - validity mask over all discrete actions (ASC-style local admissibility)
    - width potential Ψ_P and depth potential Ψ_H per action

    Returns:
        valid_mask: np.ndarray of shape (num_actions,), bool
        psi_total: np.ndarray of shape (num_actions,), float32
    """
    num_actions = 12
    lengths, max_relator_length = _compute_lengths(state)
    base_total_len = sum(lengths)

    valid_mask = np.ones(num_actions, dtype=bool)
    psi_p = np.zeros(num_actions, dtype=np.float32)
    psi_h = np.zeros(num_actions, dtype=np.float32)

    # Small epsilon to avoid division by zero normalisation issues
    eps = 1e-6

    for a in range(num_actions):
        # simulate AC move
        new_state, new_lengths = ACMove(
            move_id=a,
            presentation=state,
            max_relator_length=max_relator_length,
            lengths=lengths.copy(),
        )

        # Treat actions that fail to change the presentation and lengths as locally invalid.
        # This approximates the "feasibility floor" ASC behaviour for AC moves that would
        # exceed the max_relator_length bound.
        if np.array_equal(new_state, state) and new_lengths == lengths:
            valid_mask[a] = False
            psi_p[a] = 0.0
            psi_h[a] = 0.0
            continue

        new_total_len = sum(new_lengths)

        # Width potential Ψ_P: prefer moves that reduce total relator length.
        delta_len = new_total_len - base_total_len
        psi_p[a] = -float(delta_len)

        # Depth potential Ψ_H: negative Euclidean distance in embedding space to the
        # closest trivial target presentation.
        diffs = trivial_targets.astype(np.float32) - new_state.astype(np.float32)
        dists = np.linalg.norm(diffs, axis=1)
        min_dist = float(dists.min()) if dists.size > 0 else 0.0
        psi_h[a] = -min_dist

    # Simple normalisation so that magnitudes are comparable across states
    if np.any(valid_mask):
        valid_psi_p = np.abs(psi_p[valid_mask])
        valid_psi_h = np.abs(psi_h[valid_mask])

        scale_p = float(valid_psi_p.max()) if valid_psi_p.size > 0 else 1.0
        scale_h = float(valid_psi_h.max()) if valid_psi_h.size > 0 else 1.0

        scale_p = scale_p if scale_p > eps else 1.0
        scale_h = scale_h if scale_h > eps else 1.0

        psi_p /= scale_p
        psi_h /= scale_h

    psi_total = lambda_width * psi_p + lambda_depth * psi_h
    return valid_mask, psi_total.astype(np.float32)


def sage_validity_and_potentials_batch(
    states: np.ndarray,
    trivial_targets: np.ndarray,
    lambda_width: float = 1.0,
    lambda_depth: float = 1.0,
):
    """
    Vectorised wrapper over `sage_validity_and_potentials_for_state` for a batch
    of AC presentations with the same max_relator_length.

    Args:
        states: np.ndarray with shape (batch_size, obs_dim)

    Returns:
        valid_mask: np.ndarray of shape (batch_size, num_actions), bool
        psi_total: np.ndarray of shape (batch_size, num_actions), float32
    """
    batch_size = states.shape[0]
    num_actions = 12
    valid_masks = np.zeros((batch_size, num_actions), dtype=bool)
    psi_totals = np.zeros((batch_size, num_actions), dtype=np.float32)

    for i in range(batch_size):
        vm, psi = sage_validity_and_potentials_for_state(
            state=states[i],
            trivial_targets=trivial_targets,
            lambda_width=lambda_width,
            lambda_depth=lambda_depth,
        )
        valid_masks[i] = vm
        psi_totals[i] = psi

    return valid_masks, psi_totals
