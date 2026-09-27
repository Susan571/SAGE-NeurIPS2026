# <a href="https://github.com/Susan571/SAGE-NeurIPS2026" style="color: black !important;">SAGE: Mitigating Long-Horizon Reasoning Biases via Topological Guidance</a>

**Xinyue Zeng, Jiawei Zhang, Yujun Yan, Dawei Zhou**

*Accepted by NeurIPS 2026*

<a href="https://arxiv.org/abs/2609.30192">[Paper]</a>

---

SAGE is a topological-guidance framework for improving multi-step reasoning.
Instead of treating every generated step as equally useful, SAGE combines
local validity constraints, structural potentials, and group-relative policy
updates to keep a reasoning trajectory on a more coherent path.

This repository contains two implementations:

- an LLM post-training pipeline based on frozen representations, a label-free
  structural prior, guided rollouts, and GRPO;
- a discrete Andrews--Curtis (AC) proof of concept with PPO-style training,
  symbolic validity masking, and topological action guidance.

> **Implementation status.** The code is an auditable research implementation.
> The defaults are runnable reproduction defaults and are not intended to
> represent unreleased experimental settings or reported benchmark results.

---

## 📌 Abstract

Long-horizon reasoning remains a central challenge for large language models
(LLMs) under sparse-reward regimes. We identify two biases induced by complex
reasoning spaces: an **exploration bias**, where models are drawn toward
locally plausible but structurally unstable branches, and a **compounding
bias**, where small local deviations accumulate across depth and suppress rare
rewards.

We introduce **Symbolic Closure Analysis (SCA)** as a theoretical lens for
characterizing how branching structures and sparse rewards induce these biases.
Motivated by this analysis, we propose **Structural Admissibility-Guided
Exploration (SAGE)**, a unified framework that injects structural guidance into
long-horizon reasoning. SAGE combines algebraic sparsification, which projects
locally admissible candidates onto operator-indexed subspaces to reduce
spurious branching, with hyperbolic structural guidance, which embeds reasoning
states into a negatively curved space to provide dense depth-wise signals.

The repository provides an auditable implementation of these ideas for LLM
post-training and for the Andrews-Curtis symbolic reasoning environment.

---

## 🧠 Core Insight: Long-Horizon Errors Are Structural

SAGE separates two complementary questions that are often conflated:

| Question | SAGE signal | Role |
| --- | --- | --- |
| Is this step locally admissible? | **ASC-style validity** | Masks actions that violate the local feasibility floor |
| Does this step move toward a coherent solution? | **TNSC-style potential** | Scores structural progress in width/depth or representation space |
| Is the complete answer correct? | **Terminal task reward** | Supplies the final outcome signal |

The combined trajectory reward used by the LLM pipeline is:

```text
R(trajectory) = terminal_reward + η · mean(step_potentials)
```

This keeps structural guidance from replacing the task objective: it shapes
the path while the terminal answer remains the correctness signal.

---

## 🔍 Method Overview

### LLM + GRPO/SAGE

The LLM implementation follows a label-separated pipeline:

1. **Prepare data** in a small JSONL schema.
2. **Collect stochastic reference rollouts** from a frozen Hugging Face model.
3. **Filter by semantic entropy** so the prior is fitted on informative rollouts.
4. **Fit a label-free structural prior** using frozen hidden states, a ridge
   probe, operator subspaces, and a Poincaré-ball representation.
5. **Generate guided candidate steps** and score them with the structural prior.
6. **Update the policy with GRPO**, using group-relative advantages, clipping,
   and a KL penalty.

The prior-fitting stage does not consume gold rationales, outcome labels, or
test labels. `eval_llm.py` evaluates the trained model directly without
training-time structural guidance.

### Discrete AC proof of concept

The original symbolic environment exposes the same idea in a directly
inspectable form:

- **ASC** masks locally invalid AC moves;
- **TNSC** combines a width potential (relator-length change) and a depth
  potential (distance to trivial target presentations);
- guided logits, hybrid rewards, and group-relative advantages are used by the
  PPO-style training loop.

---

## 🗂️ Project Structure

```text
SAGE-Long-Horizon-Reasoning/
├── train_llm.py          # LLM data, rollout, prior fitting, and GRPO entry point
├── eval_llm.py           # Direct-policy LLM evaluation
├── llm_sage/             # Sampling, structural prior, rewards, and GRPO
├── train.py              # Discrete AC training entry point
├── eval/                 # Discrete AC evaluation
├── ac_solver/            # AC environment, agents, and classical search
├── guidance.py           # ASC validity masks and TNSC potentials
├── policy.py             # SAGE actor-critic policy
├── pyproject.toml        # Package metadata and optional dependencies
└── README.md
```

---

## 🚀 Quick Start

### Installation

Install the LLM dependencies:

```bash
pip install -e '.[llm]'
```

For the discrete AC environment:

```bash
pip install -e '.[ac]'
```

### LLM + GRPO/SAGE pipeline

Prepare a dataset, collect reference rollouts, fit the structural prior, and
train the policy:

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
```

Evaluate a trained checkpoint:

```bash
python eval_llm.py \
  --config <path-to-config.yaml> \
  --model <path-to-trained-model> \
  --input <path-to-test.jsonl> \
  --output <path-to-predictions.jsonl>
```

The configuration supports mathematical exact-answer tasks and natural
multiple-choice tasks. See [`llm_sage/config.py`](llm_sage/config.py) for all
available parameters.

### Discrete AC training and evaluation

Minimal SAGE training:

```bash
python -m sage.train --use-sage
```

Evaluate a checkpoint:

```bash
python -m sage.eval.evaluate \
  --checkpoint_path <path-to-checkpoint.pt> \
  --num_episodes 100
```

The evaluation module documents result files, trajectory saving, and baseline
comparisons in [`eval/README.md`](eval/README.md).

---

## ⚙️ Main Configuration Knobs

LLM-specific parameters include:

- `group_size` and `candidates_per_step` for group-relative candidate sampling;
- `lambda_psi`, `alpha`, and `gamma` for structural guidance strength;
- `structural_dim`, `subspace_rank`, and `ridge_xi` for the learned prior;
- `reference_rollouts`, `entropy_low`, and `entropy_high` for entropy filtering;
- `beta_kl`, `clip_epsilon`, and `learning_rate` for GRPO optimization.

Discrete AC options include:

- `--use-sage` to enable SAGE guidance;
- `--sage-lambda` for the global topological-potential scale;
- `--sage-width-coef` and `--sage-depth-coef` for the two potential terms;
- `--sage-beta-valid` for the weak local-validity process reward;
- `--sage-group-adv` for group-relative advantage normalization.

---

## 📦 Dependencies

The core package depends on PyTorch. Optional dependency groups are kept
separate so that the LLM and discrete AC paths can be installed independently:

- **LLM:** Transformers, Accelerate, Datasets, and PyYAML
- **AC:** NumPy, Gymnasium, tqdm, and Weights & Biases

See [`pyproject.toml`](pyproject.toml) for the authoritative package metadata.
