"""
SAGE evaluation script for AC problem solving.

Follows the structure of EMPO-main/eval_math for consistency.
"""

import argparse
import os
import json
import time
from datetime import datetime
from distutils.util import strtobool
from tqdm import tqdm
import numpy as np
import torch

from trex.policy import SAGEPolicy
from trex.env_setup import get_env
from trex.eval.utils import set_seed, save_json, save_jsonl


def parse_args():
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate SAGE on AC problems")
    
    # Checkpoint and model
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to SAGE checkpoint file (.pt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_outputs",
        help="Directory to save evaluation results",
    )
    
    # Evaluation settings
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel environments for evaluation",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=2000,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--states_type",
        type=str,
        default="all",
        choices=["solved", "unsolved", "all"],
        help="Type of initial states to evaluate on",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for evaluation",
    )
    
    # SAGE-specific evaluation settings
    parser.add_argument(
        "--use_trex_guidance",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Use SAGE guidance during evaluation",
    )
    parser.add_argument(
        "--eval_deterministic",
        action="store_true",
        help="Use deterministic policy (greedy action selection)",
    )
    
    # Output options
    parser.add_argument(
        "--save_trajectories",
        action="store_true",
        help="Save full trajectories for analysis",
    )
    parser.add_argument(
        "--save_failed",
        action="store_true",
        help="Save failed cases for analysis",
    )
    
    args = parser.parse_args()
    return args


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Load SAGE checkpoint and return policy, config, and metadata.
    
    Returns:
        policy: SAGEPolicy model
        config_args: Args object with configuration
        metadata: Additional checkpoint metadata
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get("config", {})
    
    # Create dummy args object from config with defaults
    class Args:
        def __init__(self, config_dict):
            # Set defaults first
            self.nodes_counts = [256, 256]
            self.num_envs = 1
            self.horizon_length = 2000
            self.fixed_init_state = False
            self.states_type = "all"
            self.use_supermoves = False
            self.norm_rewards = False
            self.clip_rewards = False
            self.max_relator_length = 36
            
            # Override with config values
            for k, v in config_dict.items():
                setattr(self, k, v)
    
    config_args = Args(config)
    
    # Create a minimal environment to get observation/action spaces
    # Use a simple trivial state for initialization
    import numpy as np
    from trex.ac_solver.envs.ac_env import ACEnv, ACEnvConfig
    
    dummy_state = np.array([1, 0, 2, 0])  # Trivial state <x, y>
    env_config = ACEnvConfig(
        initial_state=dummy_state,
        horizon_length=config_args.horizon_length,
        use_supermoves=config_args.use_supermoves,
    )
    dummy_env = ACEnv(env_config)
    
    # Create vectorized env wrapper for policy initialization
    import gymnasium as gym
    envs = gym.vector.SyncVectorEnv([lambda: dummy_env])
    
    # Create policy
    policy = SAGEPolicy(envs, config_args.nodes_counts).to(device)
    
    # Load weights
    policy.actor.load_state_dict(checkpoint["actor"])
    policy.critic.load_state_dict(checkpoint["critic"])
    policy.eval()
    
    metadata = {
        "update": checkpoint.get("update", 0),
        "episode": checkpoint.get("episode", 0),
        "mean_return": checkpoint.get("mean_return", 0.0),
        "global_step": checkpoint.get("global_step", 0),
    }
    
    print(f"Loaded checkpoint: update={metadata['update']}, episode={metadata['episode']}")
    return policy, config_args, metadata


def evaluate_single_episode(
    policy,
    env,
    initial_state,
    max_steps,
    use_trex_guidance,
    device,
    deterministic=False,
):
    """
    Evaluate policy on a single AC problem.
    
    Returns:
        success: bool, whether problem was solved
        path_length: int, number of steps taken
        trajectory: list, sequence of actions (if save_trajectories)
        final_state: np.ndarray, final presentation state
    """
    from trex.guidance import build_trivial_targets, trex_validity_and_potentials_batch
    
    obs = torch.tensor(initial_state, dtype=torch.float32).to(device)
    env.reset(options={"starting_state": initial_state})
    
    trajectory = []
    done = False
    truncated = False
    step_count = 0
    
    # Precompute trivial targets for SAGE guidance
    if use_trex_guidance:
        max_relator_length = env.max_relator_length
        trivial_targets = build_trivial_targets(max_relator_length=max_relator_length)
    
    while not done and not truncated and step_count < max_steps:
        # Compute SAGE guidance if enabled
        valid_mask = None
        psi_total = None
        
        if use_trex_guidance:
            obs_np = obs.cpu().numpy().reshape(1, -1)
            valid_masks, psi_totals = trex_validity_and_potentials_batch(
                states=obs_np,
                trivial_targets=trivial_targets,
                lambda_width=1.0,
                lambda_depth=1.0,
            )
            valid_mask = valid_masks[0]
            psi_total = psi_totals[0]
        
        # Get action from policy
        with torch.no_grad():
            if deterministic:
                # Greedy: take action with highest probability
                logits = policy.actor(obs)
                if valid_mask is not None:
                    logits = logits.clone()
                    logits[~torch.tensor(valid_mask, device=device)] = float("-inf")
                action = logits.argmax().unsqueeze(0)
                _, logprob, _, value = policy.get_action_and_value(obs, action=action)
            else:
                action, logprob, _, value = policy.get_action_and_value(
                    obs,
                    valid_action_mask=valid_mask,
                    psi_total=psi_total,
                    trex_lambda=1.0 if use_trex_guidance else 0.0,
                )
        
        action_np = action.cpu().numpy()[0]
        trajectory.append(int(action_np))
        
        # Step environment
        next_obs, reward, done, truncated, info = env.step(action_np)
        obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
        
        step_count += 1
        
        if done:
            break
    
    success = done
    final_state = env.state.copy()
    
    return {
        "success": success,
        "path_length": step_count,
        "trajectory": trajectory if len(trajectory) > 0 else [],
        "final_state": final_state.tolist(),
        "truncated": truncated,
    }


def evaluate_trex(
    policy,
    initial_states,
    args,
    device,
    use_trex_guidance=True,
    deterministic=False,
):
    """
    Evaluate SAGE policy on a set of AC problems.
    
    Returns:
        results: list of evaluation results
        metrics: dict of aggregated metrics
    """
    from trex.ac_solver.envs.ac_env import ACEnv, ACEnvConfig
    
    results = []
    num_solved = 0
    path_lengths = []
    
    print(f"Evaluating on {len(initial_states)} problems...")
    
    # Ensure args has required attributes
    if not hasattr(args, "horizon_length"):
        args.horizon_length = getattr(args, "max_steps", 2000)
    if not hasattr(args, "use_supermoves"):
        args.use_supermoves = False
    
    for idx, initial_state in enumerate(tqdm(initial_states, desc="Evaluating")):
        # Create environment directly for this problem
        env_config = ACEnvConfig(
            initial_state=initial_state,
            horizon_length=args.horizon_length,
            use_supermoves=args.use_supermoves,
        )
        env = ACEnv(env_config)
        
        # Evaluate
        result = evaluate_single_episode(
            policy=policy,
            env=env,
            initial_state=initial_state,
            max_steps=getattr(args, "max_steps", args.horizon_length),
            use_trex_guidance=use_trex_guidance,
            device=device,
            deterministic=deterministic,
        )
        
        result["idx"] = idx
        result["initial_state"] = initial_state.tolist()
        
        if result["success"]:
            num_solved += 1
            path_lengths.append(result["path_length"])
        
        results.append(result)
    
    # Compute metrics
    success_rate = num_solved / len(initial_states) if initial_states else 0.0
    avg_path_length = np.mean(path_lengths) if path_lengths else 0.0
    min_path_length = min(path_lengths) if path_lengths else 0
    max_path_length = max(path_lengths) if path_lengths else 0
    
    metrics = {
        "num_problems": len(initial_states),
        "num_solved": num_solved,
        "num_failed": len(initial_states) - num_solved,
        "success_rate": float(success_rate),
        "avg_path_length": float(avg_path_length),
        "min_path_length": int(min_path_length),
        "max_path_length": int(max_path_length),
        "path_lengths": [int(x) for x in path_lengths],
    }
    
    return results, metrics


def run_evaluation(args):
    """Main evaluation function."""
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    policy, config_args, metadata = load_checkpoint(args.checkpoint_path, device)
    
    # Override config with eval args
    config_args.num_envs = args.num_envs
    config_args.max_steps = args.max_steps
    config_args.states_type = args.states_type
    config_args.horizon_length = args.max_steps
    # Ensure all required attributes exist
    if not hasattr(config_args, "gamma"):
        config_args.gamma = 0.99
    if not hasattr(config_args, "norm_rewards"):
        config_args.norm_rewards = False
    if not hasattr(config_args, "clip_rewards"):
        config_args.clip_rewards = False
    if not hasattr(config_args, "min_rew"):
        config_args.min_rew = -10
    if not hasattr(config_args, "max_rew"):
        config_args.max_rew = 1000
    
    # Load initial states
    from trex.utils import load_initial_states_from_text_file
    
    initial_states = load_initial_states_from_text_file(states_type=args.states_type)
    
    # Limit number of episodes if specified
    if args.num_episodes > 0:
        initial_states = initial_states[: args.num_episodes]
    
    print(f"Evaluating on {len(initial_states)} problems")
    
    # Run evaluation
    results, metrics = evaluate_trex(
        policy=policy,
        initial_states=initial_states,
        args=config_args,
        device=device,
        use_trex_guidance=args.use_trex_guidance,
        deterministic=args.eval_deterministic,
    )
    
    # Prepare output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = os.path.basename(args.checkpoint_path).replace(".pt", "")
    output_dir = os.path.join(args.output_dir, checkpoint_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    output_file = os.path.join(output_dir, f"eval_results_{timestamp}.jsonl")
    
    # Filter results based on save options
    results_to_save = results
    if not args.save_trajectories:
        results_to_save = [
            {k: v for k, v in r.items() if k != "trajectory"} for r in results
        ]
    
    save_jsonl(results_to_save, output_file)
    
    # Save metrics
    metrics_file = os.path.join(output_dir, f"metrics_{timestamp}.json")
    metrics["checkpoint_path"] = args.checkpoint_path
    metrics["checkpoint_metadata"] = metadata
    metrics["eval_config"] = vars(args)
    save_json(metrics, metrics_file)
    
    # Save failed cases if requested
    if args.save_failed:
        failed_results = [r for r in results if not r["success"]]
        if failed_results:
            failed_file = os.path.join(output_dir, f"failed_cases_{timestamp}.jsonl")
            save_jsonl(failed_results, failed_file)
            print(f"Saved {len(failed_results)} failed cases to {failed_file}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    print(f"Problems evaluated: {metrics['num_problems']}")
    print(f"Solved: {metrics['num_solved']}")
    print(f"Failed: {metrics['num_failed']}")
    print(f"Success rate: {metrics['success_rate']:.2%}")
    if metrics["num_solved"] > 0:
        print(f"Average path length: {metrics['avg_path_length']:.2f}")
        print(f"Min path length: {metrics['min_path_length']}")
        print(f"Max path length: {metrics['max_path_length']}")
    print("=" * 50)
    print(f"\nResults saved to: {output_dir}")
    
    return results, metrics


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
