# SAGE: Mitigating Long-Horizon Reasoning Biases via Topological Structural Guidance

## Project Structure

```
sage/
├── train.py          # Main training entry point
├── training.py       # SAGE training loop
├── policy.py         # SAGEPolicy (actor-critic)
├── guidance.py       # ASC validity masking + TNSC potentials
├── config.py         # Command-line arguments
├── env_setup.py      # Environment initialization
├── utils.py          # Helper functions
├── eval/             # Evaluation scripts
└── ac_solver/        # AC problem generator and environment
    ├── envs/         # AC environment implementation
    ├── search/       # Classical search algorithms (BFS, Greedy)
    └── agents/       # PPO baseline for comparison
```

## Start

### Training

```bash
python -m sage.train --use-sage
```

### Full Training Example

```bash
python -m sage.train \
    --use-sage \
    --sage-lambda 1.5 \
    --sage-width-coef 1.0 \
    --sage-depth-coef 1.0 \
    --sage-beta-valid 0.1 \
    --sage-group-adv \
    --num-envs 8 \
    --total-timesteps 1000000 \
    --wandb-log
```

### Evaluation

```bash
python -m sage.eval.evaluate \
    --checkpoint_path out/sage_checkpoint.pt \
    --num_episodes 100 \
    --use_sage_guidance
```

## Key Arguments

### SAGE-Specific
- `--use-sage`: Enable SAGE features
- `--sage-lambda`: Global scale λ for topological potential (default: 1.0)
- `--sage-width-coef`: Weight for width potential Ψ_P (default: 1.0)
- `--sage-depth-coef`: Weight for depth potential Ψ_H (default: 1.0)
- `--sage-beta-valid`: Coefficient β for weak process reward (default: 0.0)
- `--sage-group-adv`: Enable group-relative advantage normalization

### Standard RL
- `--num-envs`: Number of parallel environments (default: 4)
- `--num-steps`: Steps per rollout (default: 2000)
- `--total-timesteps`: Total training timesteps (default: 200000)
- `--learning-rate`: Learning rate (default: 2.5e-4)
- `--gamma`: Discount factor (default: 0.99)

## Algorithm Components

### ASC: Active Symbolic Closure
Local validity checking and prefix-level pruning to keep trajectories in feasible set **F**. Invalid actions are masked before sampling.

### TNSC: Topological Neuro-Symbolic Compression
- **Width Potential (Ψ_P)**: Prefers actions that reduce total relator length
- **Depth Potential (Ψ_H)**: Measures distance to closest trivial target (Euclidean placeholder)
- **Combined**: `Ψ_total = λ_width · Ψ_P + λ_depth · Ψ_H`

### Topologically Guided Sampling
Policy logits augmented with topological potential: `logits = base_logits + λ · Ψ_total`

### Hybrid Reward
`R(τ) = 1[τ ∈ T*] + β · Σ_t 1[Valid_S(·) = ⊤]` - accumulates valid transitions until first violation.

### Policy Update
PPO clip + KL penalty with adaptive beta. Optional group-relative advantage normalization.

## Dependencies

- PyTorch
- NumPy
- Gymnasium
- WandB (optional, for logging)
- tqdm

## Notes

- **Depth Potential**: Uses Euclidean distance as placeholder (can be upgraded to Poincaré ball)
- **Hybrid Reward**: Accumulates valid transitions until first violation, then adds reward
- **AC Generator**: The `ac_solver` module provides the environment for AC problem  generation
