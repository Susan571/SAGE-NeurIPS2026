# SAGE Evaluation

This module provides evaluation scripts for SAGE models on AC problems, following the structure of EMPO-main/eval_math.

## Quick Start

### Evaluate a Checkpoint

```bash
python -m trex.eval.evaluate \
    --checkpoint_path out/sage_checkpoint.pt \
    --output_dir ./eval_outputs \
    --num_episodes 100 \
    --states_type all \
    --use_sage_guidance \
    --save_trajectories
```

### Using Shell Script

```bash
bash trex/eval/eval.sh <checkpoint_path> [output_dir] [num_episodes] [states_type]
```

## Evaluation Features

- **Checkpoint Loading**: Loads trained SAGE checkpoints with full configuration
- **SAGE Guidance**: Optionally uses SAGE validity masking and topological potentials during evaluation
- **Deterministic Mode**: Greedy action selection for reproducible results
- **Trajectory Saving**: Save full action sequences for analysis
- **Failed Case Analysis**: Optionally save failed problems for debugging

## Output Format

### Results JSONL (`eval_results_<timestamp>.jsonl`)

Each line contains:
```json
{
  "idx": 0,
  "initial_state": [1, 1, -2, -2, -2, 1, 2, 1, -2, -1, -2, ...],
  "success": true,
  "path_length": 45,
  "trajectory": [0, 3, 5, 2, ...],
  "final_state": [1, 0, 2, 0, ...],
  "truncated": false
}
```

### Metrics JSON (`metrics_<timestamp>.json`)

```json
{
  "num_problems": 100,
  "num_solved": 75,
  "num_failed": 25,
  "success_rate": 0.75,
  "avg_path_length": 42.3,
  "min_path_length": 12,
  "max_path_length": 156,
  "path_lengths": [12, 15, 23, ...],
  "checkpoint_path": "out/sage_checkpoint.pt",
  "checkpoint_metadata": {...},
  "eval_config": {...}
}
```

## Comparison with Baselines

To compare SAGE with classical search algorithms:

1. **BFS**: Use `ac_solver/search/breadth_first.py`
2. **Greedy**: Use `ac_solver/search/greedy.py`
3. **PPO Baseline**: Train and evaluate using `ac_solver/agents/ppo.py`

## Evaluation Metrics

- **Success Rate**: Primary metric - percentage of problems solved
- **Path Length**: Number of AC moves required to solve
- **Efficiency**: Compare average path lengths across methods
- **Failure Analysis**: Examine unsolved problems to identify patterns
