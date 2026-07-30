"""
SAGE training loop implementing ASC, TNSC, guided sampling, hybrid rewards, and SAGE-style PPO updates.
"""

import math
import random
import uuid
import wandb
from collections import deque
from tqdm import tqdm
from os import makedirs
from os.path import join
import numpy as np
import torch
from torch import nn

from sage.guidance import (
    build_trivial_targets,
    sage_validity_and_potentials_batch,
)


def get_curr_lr(n_update, lr_decay, warmup, max_lr, min_lr, total_updates):
    """
    Calculates the current learning rate based on the update step, learning rate decay schedule,
    warmup period, and other parameters.

    Parameters:
    n_update (int): The current update step (1-indexed).
    lr_decay (str): The type of learning rate decay to apply ("linear" or "cosine").
    warmup (float): The fraction of total updates to be used for the learning rate warmup.
    max_lr (float): The maximum learning rate.
    min_lr (float): The minimum learning rate.
    total_updates (int): The total number of updates.

    Returns:
    float: The current learning rate.

    Raises:
    NotImplementedError: If an unsupported lr_decay type is provided.
    """
    # Convert to 0-indexed for internal calculations
    n_update -= 1
    total_updates -= 1

    # Calculate the end of the warmup period
    warmup_period_end = total_updates * warmup

    if warmup_period_end > 0 and n_update <= warmup_period_end:
        lrnow = max_lr * n_update / warmup_period_end
    else:
        if lr_decay == "linear":
            slope = (max_lr - min_lr) / (warmup_period_end - total_updates)
            intercept = max_lr - slope * warmup_period_end
            lrnow = slope * n_update + intercept

        elif lr_decay == "cosine":
            cosine_arg = (
                (n_update - warmup_period_end)
                / (total_updates - warmup_period_end)
                * math.pi
            )
            lrnow = min_lr + (max_lr - min_lr) * (1 + math.cos(cosine_arg)) / 2

        else:
            raise NotImplementedError(
                "Only 'linear' and 'cosine' lr-schedules are available."
            )

    return lrnow


def sage_training_loop(
    envs,
    args,
    device,
    optimizer,
    policy,
    curr_states,
    success_record,
    ACMoves_hist,
    states_processed,
    initial_states,
):
    """
    SAGE training loop implementing:
    - ASC: Active Symbolic Closure (validity masking)
    - TNSC: Topological Neuro-Symbolic Compression (potentials)
    - Topologically Guided Sampling
    - Hybrid Reward (terminal + weak process reward)
    - SAGE-style PPO update with KL penalty
    """
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    valid_transitions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32).to(device)

    global_step = 0
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Track cumulative valid transitions until first violation per environment
    cumulative_valid = np.zeros(args.num_envs, dtype=np.float32)  # cumulative count per env
    violation_occurred = np.zeros(args.num_envs, dtype=bool)  # track if violation happened
    num_updates = args.total_timesteps // args.batch_size
    episodic_return = np.array([0] * args.num_envs)
    episodic_length = np.array([0] * args.num_envs)
    episode = 0
    returns_queue = deque([0], maxlen=100)
    lengths_queue = deque([0], maxlen=100)
    round1_complete = False
    beta = args.beta  # SAGE uses KL penalty by default

    # Precompute trivial targets for depth potential Ψ_H
    max_relator_length = envs.envs[0].max_relator_length
    trivial_targets = build_trivial_targets(max_relator_length=max_relator_length)

    run_name = f"{args.exp_name}_sage-ffn-nodes_{args.nodes_counts}_{uuid.uuid4()}"
    out_dir = f"out/{run_name}"
    makedirs(out_dir, exist_ok=True)
    if args.wandb_log:
        run = wandb.init(
            project=args.wandb_project_name,
            name=run_name,
            config=vars(args),
            save_code=True,
        )

    print(f"total number of timesteps: {args.total_timesteps}, updates: {num_updates}")
    for update in tqdm(
        range(1, num_updates + 1), desc="SAGE Training Progress", total=num_updates
    ):

        random.seed(args.seed + update)
        np.random.seed(args.seed + update)
        torch.manual_seed(args.seed + update)

        if args.anneal_lr:
            lrnow = get_curr_lr(
                n_update=update,
                lr_decay=args.lr_decay,
                warmup=args.warmup_period,
                max_lr=args.learning_rate,
                min_lr=args.learning_rate * args.min_lr_frac,
                total_updates=num_updates,
            )
            optimizer.param_groups[0]["lr"] = lrnow

        # Rollout phase with SAGE guidance
        for step in tqdm(
            range(0, args.num_steps), desc=f"SAGE Rollout - {update}", leave=False
        ):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Compute SAGE validity masks and potentials
            obs_np = next_obs.cpu().numpy()
            valid_masks, psi_totals = sage_validity_and_potentials_batch(
                states=obs_np,
                trivial_targets=trivial_targets,
                lambda_width=args.sage_width_coef,
                lambda_depth=args.sage_depth_coef,
            )

            # Sample actions with SAGE guidance
            with torch.no_grad():
                action, logprob, _, value = policy.get_action_and_value(
                    next_obs,
                    valid_action_mask=valid_masks,
                    psi_total=psi_totals,
                    sage_lambda=args.sage_lambda,
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Track valid transitions and accumulate until first violation
            action_np = action.cpu().numpy()
            for i in range(args.num_envs):
                # Skip if violation already occurred for this trajectory
                if violation_occurred[i]:
                    continue

                # Check if current action is valid
                if valid_masks[i, action_np[i]]:
                    # Action is valid: accumulate
                    cumulative_valid[i] += 1.0
                else:
                    # First invalid action: mark violation occurred
                    violation_occurred[i] = True
                    # Note: cumulative_valid[i] keeps the count up to this point

            # Execute environment step
            next_obs, reward, done, truncated, infos = envs.step(
                action.cpu().numpy()
            )

            # Hybrid reward: R(τ) = 1[τ ∈ T*] + β * Σ_t 1[Valid_S(·) = ⊤]
            # Accumulate valid transitions until first violation, then add reward
            if args.sage_beta_valid > 0:
                # Check for environments that violated or ended
                for i in range(args.num_envs):
                    # Add cumulative reward when violation occurs or episode ends
                    if violation_occurred[i] or done[i] or truncated[i]:
                        # Add cumulative reward for valid transitions until violation/end
                        if cumulative_valid[i] > 0:
                            reward[i] += args.sage_beta_valid * cumulative_valid[i]
                        # Reset for next trajectory
                        cumulative_valid[i] = 0.0
                        violation_occurred[i] = False

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            episodic_return = episodic_return + reward
            episodic_length = episodic_length + 1

            # Track valid transitions for logging
            valid_transitions[step] = torch.tensor([
                1.0 if valid_masks[i, action_np[i]] else 0.0
                for i in range(args.num_envs)
            ]).to(device)

            _record_info = np.array(
                [
                    True if done[i] or truncated[i] else False
                    for i in range(args.num_envs)
                ]
            )
            if _record_info.any():
                for i, el in enumerate(_record_info):

                    if done[i]:
                        if curr_states[i] in success_record["unsolved"]:
                            success_record["unsolved"].remove(curr_states[i])
                            success_record["solved"].add(curr_states[i])

                        if curr_states[i] not in ACMoves_hist:
                            ACMoves_hist[curr_states[i]] = infos["final_info"][i][
                                "actions"
                            ]
                        else:
                            prev_path_length = len(ACMoves_hist[curr_states[i]])
                            new_path_length = len(infos["final_info"][i]["actions"])
                            if new_path_length < prev_path_length:
                                ACMoves_hist[curr_states[i]] = infos["final_info"][i][
                                    "actions"
                                ]

                    if el:
                        returns_queue.append(episodic_return[i])
                        lengths_queue.append(episodic_length[i])
                        episode += 1
                        episodic_return[i], episodic_length[i] = 0, 0

                        # Reset cumulative valid tracking for new trajectory
                        cumulative_valid[i] = 0.0
                        violation_occurred[i] = False

                        prev_state = curr_states[i]
                        round1_complete = (
                            True
                            if round1_complete
                            or (max(states_processed) == len(initial_states) - 1)
                            else False
                        )
                        if not round1_complete:
                            curr_states[i] = max(states_processed) + 1
                        else:
                            if len(success_record["solved"]) == 0 or (
                                success_record["unsolved"]
                                and random.uniform(0, 1) > args.repeat_solved_prob
                            ):
                                curr_states[i] = random.choice(
                                    list(success_record["unsolved"])
                                )
                            else:
                                curr_states[i] = random.choice(
                                    list(success_record["solved"])
                                )
                        states_processed.add(curr_states[i])
                        next_obs[i] = initial_states[curr_states[i]]
                        envs.envs[i].reset(options={"starting_state": next_obs[i]})

            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                done
            ).to(device)

        if not args.norm_rewards:
            rewards /= envs.envs[0].max_reward
            normalized_returns = np.array(returns_queue) / envs.envs[0].max_reward
            normalized_lengths = np.array(lengths_queue) / args.horizon_length
        else:
            normalized_returns = np.array(returns_queue)
            normalized_lengths = np.array(lengths_queue)

        # Compute advantages with optional group-relative normalization
        with torch.no_grad():
            next_value = policy.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # Group-relative advantage normalization (SAGE-style)
        if args.sage_group_adv:
            # Normalize advantages within each trajectory/episode
            episode_ids = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
            episode_counter = 0
            for i in range(args.num_envs):
                for t in range(args.num_steps):
                    if dones[t, i] or (t > 0 and dones[t-1, i]):
                        episode_counter += 1
                    episode_ids[t, i] = episode_counter

            b_episode_ids = episode_ids.reshape(-1)
            b_advantages_flat = advantages.reshape(-1)

            # Normalize within each episode group
            for ep_id in torch.unique(b_episode_ids):
                mask = b_episode_ids == ep_id
                if mask.sum() > 1:
                    b_advantages_flat[mask] = (
                        (b_advantages_flat[mask] - b_advantages_flat[mask].mean())
                        / (b_advantages_flat[mask].std() + 1e-8)
                    )

            advantages = b_advantages_flat.reshape(advantages.shape)

        # Flatten data
        b_obs = obs.reshape(
            (-1,) + envs.single_observation_space.shape
        )
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_valid_transitions = valid_transitions.reshape(-1)

        # SAGE-style PPO update with KL penalty
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Recompute logprobs with current policy (for KL computation)
                _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Approximate KL divergence
                kl_var = (ratio - 1) - logratio
                with torch.no_grad():
                    approx_kl = kl_var.mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv and not args.sage_group_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # SAGE policy loss: PPO clip + KL penalty
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss_clip = torch.max(pg_loss1, pg_loss2).mean()

                # KL penalty term
                pg_loss_kl = beta * kl_var.mean()
                pg_loss = pg_loss_clip - pg_loss_kl

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    policy.parameters(), args.max_grad_norm
                )
                optimizer.step()

            # Adaptive beta for KL penalty
            if args.target_kl is not None:
                if approx_kl > args.target_kl * 1.5:
                    beta = beta * 2
                elif approx_kl < args.target_kl / 1.5:
                    beta = beta / 2

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if args.wandb_log:
            wandb.log(
                {
                    "charts/global_step": global_step,
                    "charts/episode": episode,
                    "charts/normalized_returns_mean": normalized_returns.mean(),
                    "charts/normalized_lengths_mean": normalized_lengths.mean(),
                    "charts/learning_rate": optimizer.param_groups[0]["lr"],
                    "charts/solved": len(success_record["solved"]),
                    "charts/unsolved": len(success_record["unsolved"]),
                    "charts/highest_solved": (
                        max(success_record["solved"])
                        if success_record["solved"]
                        else -1
                    ),
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy_loss": entropy_loss.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/explained_variance": explained_var,
                    "losses/clipfrac": np.mean(clipfracs),
                    "sage/beta": beta,
                    "sage/valid_transitions_mean": b_valid_transitions.mean().item(),
                    "debug/advantages_mean": b_advantages.mean(),
                    "debug/advantages_std": b_advantages.std(),
                }
            )

        if update > 0 and update % 100 == 0:
            checkpoint = {
                "critic": policy.critic.state_dict(),
                "actor": policy.actor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update,
                "episode": episode,
                "config": vars(args),
                "mean_return": normalized_returns.mean(),
                "success_record": success_record,
                "value_loss": v_loss.item(),
                "policy_loss": pg_loss.item(),
                "entropy_loss": entropy_loss.item(),
                "approx_kl": approx_kl.item(),
                "explained_var": explained_var,
                "clipfrac": np.mean(clipfracs),
                "global_step": global_step,
                "round1_complete": round1_complete,
                "curr_states": curr_states,
                "states_processed": states_processed,
                "ACMoves_hist": ACMoves_hist,
                "supermoves": envs.envs[0].supermoves,
                "sage_beta": beta,
            }
            print(f"saving SAGE checkpoint to {out_dir}")
            torch.save(checkpoint, join(out_dir, "ckpt.pt"))

    return
