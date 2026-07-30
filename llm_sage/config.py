from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import math


@dataclass
class SAGEConfig:
    """Configuration shared by guided rollout generation and GRPO.

    Appendix E specifies the algorithm but not every numerical value. Defaults
    here are runnable reproduction defaults, not reconstructed paper secrets.
    """

    model_name: str = "Qwen/Qwen3.5-2B"
    dataset_name: str = "openai/gsm8k"
    dataset_config: str | None = "main"
    train_split: str = "train"
    task_family: str = "math"
    output_dir: str = "outputs/llm_sage"
    seed: int = 42
    system_prompt: str = (
        "Please reason step by step, and put your final answer within \\boxed{}."
    )

    # E.2: finite candidate-step sampling
    group_size: int = 8
    candidates_per_step: int = 4
    max_steps: int = 16
    max_step_tokens: int = 128
    max_prompt_tokens: int = 1024
    temperature: float = 0.8
    top_p: float = 0.95
    lambda_psi: float = 1.0
    alpha: float = 1.0
    gamma: float = 1.0
    eta: float = 1.0

    # E.3: frozen representation and Poincare ball
    encoder_layer: int = -1
    encoder_batch_size: int = 8
    max_prior_states: int = 50000
    structural_dim: int = 64
    curvature: float = 1.0
    hyperbolic_kappa: float = 1.0

    # E.4: label-free ridge probe and PCA subspaces
    ridge_xi: float = 1e-3
    subspace_rank: int = 4
    epsilon: float = 1e-8

    # E.5: fixed reference-policy entropy filter
    reference_rollouts: int = 8
    entropy_low: float = 0.1
    entropy_high: float = 2.0
    semantic_similarity_threshold: float = 0.85

    # GRPO
    learning_rate: float = 1e-6
    beta_kl: float = 1e-3
    clip_epsilon: float = 0.2
    epochs: int = 1
    policy_update_epochs: int = 2
    max_train_prompts: int | None = None
    save_every_steps: int = 50
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.task_family not in {"math", "natural"}:
            raise ValueError("task_family must be 'math' or 'natural'")
        positive_ints = {
            "group_size": self.group_size,
            "candidates_per_step": self.candidates_per_step,
            "max_steps": self.max_steps,
            "max_step_tokens": self.max_step_tokens,
            "max_prompt_tokens": self.max_prompt_tokens,
            "encoder_batch_size": self.encoder_batch_size,
            "max_prior_states": self.max_prior_states,
            "structural_dim": self.structural_dim,
            "subspace_rank": self.subspace_rank,
            "reference_rollouts": self.reference_rollouts,
            "epochs": self.epochs,
            "policy_update_epochs": self.policy_update_epochs,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "save_every_steps": self.save_every_steps,
        }
        invalid = [name for name, value in positive_ints.items() if value < 1]
        if invalid:
            raise ValueError(f"Config values must be positive: {invalid}")
        if self.max_train_prompts is not None and self.max_train_prompts < 1:
            raise ValueError("max_train_prompts must be positive or null")
        if not 0 < self.temperature or not 0 < self.top_p <= 1:
            raise ValueError("temperature must be positive and top_p in (0, 1]")
        if not 0 <= self.entropy_low < self.entropy_high:
            raise ValueError("entropy thresholds must satisfy 0 <= low < high")
        if self.entropy_high > math.log(self.reference_rollouts) + 1e-6:
            raise ValueError("entropy_high exceeds the maximum entropy for G rollouts")
        if not -1 <= self.semantic_similarity_threshold <= 1:
            raise ValueError("semantic_similarity_threshold must be in [-1, 1]")
        nonnegative = {
            "lambda_psi": self.lambda_psi,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "eta": self.eta,
            "beta_kl": self.beta_kl,
        }
        if any(value < 0 for value in nonnegative.values()):
            raise ValueError(f"Config values must be nonnegative: {nonnegative}")
        if self.curvature <= 0 or self.hyperbolic_kappa <= 0 or self.epsilon <= 0:
            raise ValueError("curvature, hyperbolic_kappa, and epsilon must be positive")
        if self.ridge_xi <= 0:
            raise ValueError("ridge_xi must be positive")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if not 0 < self.clip_epsilon < 1:
            raise ValueError("clip_epsilon must be in (0, 1)")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SAGEConfig":
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("Install pyyaml to read YAML configs") from exc
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config not found: {path}")
        config_text = config_path.read_text()
        data: dict[str, Any] = yaml.safe_load(config_text)
        if not isinstance(data, dict):
            raise ValueError("Config YAML must contain a mapping")
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
