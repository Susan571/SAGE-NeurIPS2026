"""
SAGE (Topologically-guided Reinforcement Learning with EXploration) module.

This module implements the SAGE algorithm for AC problem solving, including:
- ASC: Active Symbolic Closure (validity masking)
- TNSC: Topological Neuro-Symbolic Compression (potentials)
- Topologically Guided Sampling
- Hybrid Reward shaping
- SAGE-style PPO updates
"""

from trex.policy import SAGEPolicy
from trex.train import train_sage

__all__ = ["SAGEPolicy", "train_sage"]
