# SAGE: Mitigating Long-Horizon Reasoning Biases via Topological Guidance

## Project Structure

```text
SAGE-Long-Horizon-Reasoning/
├── train_llm.py      # LLM data, rollout, prior-fitting, and GRPO/SAGE entry point
├── eval_llm.py       # Direct-policy LLM evaluation
├── llm_sage/         # Candidate sampling, learned structural prior, and GRPO
├── train.py          # Legacy discrete AC training entry point
├── eval/             # Legacy discrete AC evaluation
├── ac_solver/        # AC environment, agents, and classical search
└── pyproject.toml    # Package metadata and dependencies
```

## LLM + GRPO/SAGE Pipeline

`train_llm.py` and `llm_sage/` implement the LLM post-training mechanism:
fixed-reference entropy filtering, finite candidate-step SAGE sampling, the
label-free learned structural prior, trajectory reward construction, and the
clipped step-level group-relative update. Structural modules are used during
training; `eval_llm.py` evaluates the trained policy directly.

Install the LLM dependencies:

```bash
pip install -e '.[llm]'
```

Provide a configuration, datasets, model weights, and output locations, then
run:

```bash
python train_llm.py prepare-data \
  --config <path-to-config.yaml> \
  --output <path-to-train.jsonl>

python train_llm.py collect \
  --config <path-to-config.yaml> \
  --input <path-to-train.jsonl> \
  --output <path-to-reference-rollouts.jsonl>

python train_llm.py fit-prior \
  --config <path-to-config.yaml> \
  --input <path-to-reference-rollouts.jsonl> \
  --output <path-to-structural-prior.pt>

python train_llm.py train \
  --config <path-to-config.yaml> \
  --input <path-to-reference-rollouts.jsonl> \
  --prior <path-to-structural-prior.pt> \
  --output <output-directory>

python eval_llm.py \
  --config <path-to-config.yaml> \
  --model <path-to-trained-model> \
  --input <path-to-test.jsonl> \
  --output <path-to-predictions.jsonl>
```

## Legacy Discrete AC Proof of Concept

The original discrete symbolic AC environment is packaged as `sage`. It is
separate from the LLM training pipeline above.

Install its dependencies:

```bash
pip install -e '.[ac]'
```

Train:

```bash
python -m sage.train --use-sage
```

Full training example:

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

Evaluate a supplied checkpoint:

```bash
python -m sage.eval.evaluate \
  --checkpoint_path <path-to-checkpoint.pt> \
  --num_episodes 100
```

## Legacy AC Arguments

SAGE-specific:

- `--use-sage`: enable SAGE features
- `--sage-lambda`: global scale for the topological potential
- `--sage-width-coef`: weight for the width potential
- `--sage-depth-coef`: weight for the depth potential
- `--sage-beta-valid`: coefficient for the weak process reward
- `--sage-group-adv`: enable group-relative advantage normalization

Standard RL:

- `--num-envs`: number of parallel environments
- `--num-steps`: steps per rollout
- `--total-timesteps`: total training timesteps
- `--learning-rate`: learning rate
- `--gamma`: discount factor

## Dependencies

- PyTorch
- NumPy
- Gymnasium
- tqdm
- Weights & Biases
- Transformers, Accelerate, Datasets, and PyYAML for the LLM path
