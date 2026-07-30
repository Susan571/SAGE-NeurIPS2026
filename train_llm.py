"""Auditable public LLM rollout and structural-prior reference pipeline.

The optimization primitives are in ``llm_sage.grpo``. This CLI separates
reference collection from label-free prior fitting so it can be independently
checked that the prior fit consumes no outcome labels, gold rationales, or test
labels.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from llm_sage.config import SAGEConfig
from llm_sage.data import format_prompt, iter_jsonl, normalize_record, write_jsonl
from llm_sage.filtering import (
    deterministic_clusters,
    embedding_clusters,
    retain_by_entropy,
)
from llm_sage.grpo import GRPOBatch, grpo_update
from llm_sage.hf_backend import HuggingFaceBackend
from llm_sage.rewards import (
    exact_answer_reward,
    multiple_choice_reward,
    sage_trajectory_reward,
)
from llm_sage.sampling import choose_candidate, split_reasoning_steps
from llm_sage.structural_prior import LabelFreeStructuralPrior, semantic_entropy


def load_prior(path: Path, config: SAGEConfig) -> LabelFreeStructuralPrior:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    prior_config = payload.get("config", {})
    prior_model = prior_config.get("model_name")
    if prior_model and prior_model != config.model_name:
        raise ValueError(
            f"Prior was fitted with {prior_model}, but config uses {config.model_name}"
        )
    prior = LabelFreeStructuralPrior(**payload["prior"])
    if prior.task_family != config.task_family:
        raise ValueError(
            f"Prior task family is {prior.task_family}, but config uses "
            f"{config.task_family}"
        )
    return prior


def prepare_dataset(config: SAGEConfig, output: Path) -> None:
    """Export the configured public split into the label-separated JSONL schema."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install optional LLM dependencies: pip install -e '.[llm]'"
        ) from exc
    if config.dataset_config:
        dataset = load_dataset(
            config.dataset_name,
            config.dataset_config,
            split=config.train_split,
        )
    else:
        dataset = load_dataset(config.dataset_name, split=config.train_split)

    def records():
        for index, item in enumerate(dataset):
            normalized = normalize_record(dict(item), index)
            record = {
                "prompt_id": normalized["prompt_id"],
                "prompt": normalized["prompt"],
            }
            if "answer" in normalized:
                record["answer"] = normalized["answer"]
            record["metric"] = (
                "multiple_choice"
                if config.task_family == "natural"
                else "exact_answer"
            )
            yield record

    write_jsonl(records(), output)


def collect_reference_rollouts(
    config: SAGEConfig, source: Path, output: Path, family: str
) -> None:
    backend = HuggingFaceBackend(config.model_name)
    def records():
        for item in iter_jsonl(source):
            prompt = format_prompt(item["prompt"], config.system_prompt)
            outputs = backend.generate_completions(
                prompt,
                config.reference_rollouts,
                config.max_steps * config.max_step_tokens,
                config.temperature,
                config.top_p,
                config.max_prompt_tokens,
            )
            if family == "natural":
                embeddings = backend.hidden_batched(
                    [prompt + output for output in outputs],
                    config.encoder_layer,
                    config.encoder_batch_size,
                    config.max_prompt_tokens
                    + config.max_steps * config.max_step_tokens,
                )
                clusters = embedding_clusters(
                    embeddings, config.semantic_similarity_threshold
                )
            else:
                clusters = deterministic_clusters(outputs)
            record = dict(item)
            record["prompt"] = prompt
            record.update(
                reference_outputs=outputs,
                cluster_ids=clusters,
                semantic_entropy=semantic_entropy(clusters),
                retained=retain_by_entropy(
                    clusters, config.entropy_low, config.entropy_high
                ),
            )
            yield record
    write_jsonl(records(), output)


def fit_prior(config: SAGEConfig, source: Path, output: Path, family: str) -> None:
    backend = HuggingFaceBackend(config.model_name)
    prefixes, steps = [], []
    anchor_sums: dict[str, torch.Tensor] = {}
    anchor_counts: dict[str, int] = {}
    anchor_texts: list[str] = []
    anchor_prompts: list[str] = []
    random_generator = random.Random(config.seed)
    states_seen = 0

    def flush_anchor_batch() -> None:
        if not anchor_texts:
            return
        final_hidden = backend.hidden(
            anchor_texts,
            config.encoder_layer,
            config.max_prompt_tokens + config.max_steps * config.max_step_tokens,
        ).cpu()
        for prompt, representation in zip(anchor_prompts, final_hidden):
            anchor_sums[prompt] = anchor_sums.get(
                prompt, torch.zeros_like(representation)
            ) + representation
            anchor_counts[prompt] = anchor_counts.get(prompt, 0) + 1
        anchor_texts.clear()
        anchor_prompts.clear()

    for item in iter_jsonl(source):
        if not item.get("retained", True):
            continue
        for output in item["reference_outputs"]:
            anchor_texts.append(item["prompt"] + output)
            anchor_prompts.append(item["prompt"])
            if len(anchor_texts) >= config.encoder_batch_size:
                flush_anchor_batch()
            prefix = item["prompt"]
            for step in split_reasoning_steps(output):
                states_seen += 1
                if len(steps) < config.max_prior_states:
                    prefixes.append(prefix)
                    steps.append(step)
                else:
                    replacement = random_generator.randrange(states_seen)
                    if replacement < config.max_prior_states:
                        prefixes[replacement] = prefix
                        steps[replacement] = step
                prefix += step
    flush_anchor_batch()
    if not steps:
        raise ValueError("No retained reference rollouts; adjust entropy thresholds")
    hidden = backend.hidden_batched(
        prefixes,
        config.encoder_layer,
        config.encoder_batch_size,
        config.max_prompt_tokens + config.max_steps * config.max_step_tokens,
    )
    prompt_anchor_hidden_states = {
        prompt: total / anchor_counts[prompt]
        for prompt, total in anchor_sums.items()
    }
    prior = LabelFreeStructuralPrior.fit(
        hidden,
        prefixes,
        steps,
        family,
        config.structural_dim,
        config.ridge_xi,
        config.subspace_rank,
        config.curvature,
        config.hyperbolic_kappa,
        config.epsilon,
        prompt_anchor_hidden_states,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config.to_dict(), "prior": prior.state_dict()}, output)


@dataclass
class GuidedRollout:
    completion: str
    steps: list[str]
    states: list[str]
    step_token_ids: list[torch.Tensor]
    step_potentials: torch.Tensor


def guided_rollout(
    backend: HuggingFaceBackend,
    prior: LabelFreeStructuralPrior,
    prompt: str,
    config: SAGEConfig,
    seed: int | None = None,
    encoder_backend: HuggingFaceBackend | None = None,
) -> GuidedRollout:
    state = prompt
    selected_steps = []
    selected_states = []
    selected_step_token_ids = []
    potentials = []
    generator = torch.Generator(device=backend.device).manual_seed(
        config.seed if seed is None else seed
    )
    encoder = encoder_backend or backend
    prompt_anchor = prior.anchor_for(prompt)
    for _ in range(config.max_steps):
        candidates = backend.candidate_steps(
            state,
            config.candidates_per_step,
            config.max_step_tokens,
            config.temperature,
            config.top_p,
            config.encoder_layer,
            encoder,
            config.max_prompt_tokens
            + (config.max_steps - 1) * config.max_step_tokens,
        )
        state_hidden = encoder.hidden(
            [state],
            config.encoder_layer,
            config.max_prompt_tokens
            + (config.max_steps - 1) * config.max_step_tokens,
        )[0].cpu()
        next_hidden = torch.stack([candidate.next_hidden.cpu() for candidate in candidates])
        psi_p, psi_h = prior.scores(
            state_hidden,
            next_hidden,
            [candidate.text for candidate in candidates],
            anchor_embedding=prompt_anchor,
        )
        selected, _ = choose_candidate(
            candidates,
            psi_p,
            psi_h,
            config.lambda_psi,
            config.alpha,
            config.gamma,
            generator,
        )
        selected_index = next(
            index for index, candidate in enumerate(candidates) if candidate is selected
        )
        potentials.append(
            config.alpha * psi_p[selected_index]
            + config.gamma * psi_h[selected_index]
        )
        selected_steps.append(selected.text)
        selected_states.append(state)
        selected_step_token_ids.append(selected.token_ids.cpu())
        state += selected.text
        if (
            selected.terminal
            or "\\boxed{" in selected.text
            or "final answer" in selected.text.lower()
        ):
            break
    return GuidedRollout(
        state[len(prompt) :],
        selected_steps,
        selected_states,
        selected_step_token_ids,
        torch.stack(potentials) if potentials else torch.empty(0),
    )


def _terminal_reward(
    item: dict[str, Any], completion: str, task_family: str
) -> float:
    if "answer" not in item:
        raise ValueError("Training records require an 'answer' field")
    metric = item.get(
        "metric",
        "multiple_choice" if task_family == "natural" else "exact_answer",
    )
    if metric == "exact_answer":
        return exact_answer_reward(completion, str(item["answer"]))
    if metric == "multiple_choice":
        return multiple_choice_reward(completion, str(item["answer"]))
    raise ValueError(f"Unsupported metric: {metric}")


def train_sage(
    config: SAGEConfig, source: Path, prior_path: Path, output: Path
) -> None:
    """Minimal public implementation of the Appendix E training path."""
    policy = HuggingFaceBackend(config.model_name)
    reference = HuggingFaceBackend(config.model_name)
    reference.model.requires_grad_(False)
    reference.model.eval()
    prior = load_prior(prior_path, config)
    records = []
    for item in iter_jsonl(source):
        if item.get("retained", True):
            records.append(
                {
                    key: item[key]
                    for key in ("prompt_id", "prompt", "answer", "metric")
                    if key in item
                }
            )
    if config.max_train_prompts is not None:
        records = records[: config.max_train_prompts]
    if not records:
        raise ValueError("No retained training prompts were found")
    optimizer = torch.optim.AdamW(policy.model.parameters(), lr=config.learning_rate)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(
        json.dumps(config.to_dict(), indent=2) + "\n"
    )
    global_step = 0
    optimizer_steps = 0
    pending_batches: list[tuple[int, GRPOBatch]] = []

    def save_checkpoint(epoch: int) -> None:
        checkpoint = output / f"checkpoint-{optimizer_steps}"
        policy.model.save_pretrained(checkpoint)
        policy.tokenizer.save_pretrained(checkpoint)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "rollout_step": global_step,
                "optimizer_step": optimizer_steps,
                "epoch": epoch,
            },
            checkpoint / "trainer_state.pt",
        )

    def optimize_pending(epoch: int) -> None:
        nonlocal optimizer_steps
        if not pending_batches:
            return
        policy.model.train()
        accumulation_steps = len(pending_batches)
        for update_epoch in range(config.policy_update_epochs):
            metrics_for_groups = []
            for accumulation_index, (rollout_step, cpu_batch) in enumerate(
                pending_batches
            ):
                batch = cpu_batch.to(policy.device)
                metrics = grpo_update(
                    policy.model,
                    optimizer,
                    batch,
                    config.beta_kl,
                    config.clip_epsilon,
                    config.max_grad_norm,
                    accumulation_steps,
                    accumulation_index,
                    accumulation_index + 1 == accumulation_steps,
                )
                metrics_for_groups.append(metrics)
            optimizer_steps += 1
            averaged = {
                key: sum(metrics[key] for metrics in metrics_for_groups)
                / len(metrics_for_groups)
                for key in ("loss", "policy_loss", "kl", "reward_mean")
            }
            averaged["grad_norm"] = metrics_for_groups[-1]["grad_norm"]
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "rollout_step": global_step,
                        "rollout_batch_start": pending_batches[0][0],
                        "policy_update_epoch": update_epoch,
                        "optimizer_updates": optimizer_steps,
                        **averaged,
                    }
                )
            )
            if optimizer_steps % config.save_every_steps == 0:
                save_checkpoint(epoch)
        pending_batches.clear()

    for epoch in range(config.epochs):
        for prompt_index, item in enumerate(records):
            policy.model.eval()
            rollouts = [
                guided_rollout(
                    policy,
                    prior,
                    item["prompt"],
                    config,
                    config.seed
                    + epoch * len(records) * config.group_size
                    + prompt_index * config.group_size
                    + rollout_index,
                    reference,
                )
                for rollout_index in range(config.group_size)
            ]
            rewards = torch.stack(
                [
                    sage_trajectory_reward(
                        _terminal_reward(
                            item, rollout.completion, config.task_family
                        ),
                        rollout.step_potentials,
                        config.eta,
                    )
                    for rollout in rollouts
                ]
            ).float()
            (
                input_ids,
                attention_mask,
                completion_mask,
                step_ids,
                rollout_ids,
            ) = policy.encode_rollout_actions(
                [rollout.states for rollout in rollouts],
                [rollout.step_token_ids for rollout in rollouts],
                config.max_prompt_tokens
                + (config.max_steps - 1) * config.max_step_tokens,
                config.max_step_tokens,
            )
            old_logprobs, action_mask = policy.completion_logprobs(
                input_ids, attention_mask, completion_mask, step_ids
            )
            ref_logprobs, ref_action_mask = reference.completion_logprobs(
                input_ids.to(reference.device),
                attention_mask.to(reference.device),
                completion_mask.to(reference.device),
                step_ids.to(reference.device),
            )
            ref_logprobs = ref_logprobs.to(policy.device)
            ref_action_mask = ref_action_mask.to(policy.device)
            batch = GRPOBatch(
                input_ids,
                attention_mask,
                completion_mask,
                step_ids,
                old_logprobs,
                ref_logprobs,
                action_mask & ref_action_mask,
                rewards.to(policy.device),
                torch.full((config.group_size,), global_step, device=policy.device),
                rollout_ids,
            ).to("cpu")
            global_step += 1
            pending_batches.append((global_step, batch))
            del (
                input_ids,
                attention_mask,
                completion_mask,
                step_ids,
                rollout_ids,
                old_logprobs,
                ref_logprobs,
                action_mask,
                ref_action_mask,
            )
            if (
                len(pending_batches) == config.gradient_accumulation_steps
                or prompt_index + 1 == len(records)
            ):
                optimize_pending(epoch)
    policy.model.save_pretrained(output / "final")
    policy.tokenizer.save_pretrained(output / "final")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "prepare-data",
            "collect",
            "fit-prior",
            "guided-rollout",
            "train",
        ],
    )
    parser.add_argument("--config", default="configs/llm_sage_math_2b.yaml")
    parser.add_argument("--input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--family", choices=["math", "natural"])
    parser.add_argument("--prior")
    parser.add_argument("--prompt")
    args = parser.parse_args()
    config = SAGEConfig.from_yaml(args.config)
    family = args.family or config.task_family
    if args.family is not None and args.family != config.task_family:
        parser.error(
            f"--family {args.family} conflicts with config task_family "
            f"{config.task_family}"
        )
    torch.manual_seed(config.seed)

    if args.command == "prepare-data":
        prepare_dataset(config, Path(args.output))
    elif args.command == "collect":
        if not args.input:
            parser.error("collect requires --input JSONL")
        collect_reference_rollouts(
            config, Path(args.input), Path(args.output), family
        )
    elif args.command == "fit-prior":
        if not args.input:
            parser.error("fit-prior requires --input JSONL")
        fit_prior(config, Path(args.input), Path(args.output), family)
    elif args.command == "guided-rollout":
        if not args.prior or args.prompt is None:
            parser.error("guided-rollout requires --prior and --prompt")
        backend = HuggingFaceBackend(config.model_name)
        prior = load_prior(Path(args.prior), config)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            guided_rollout(
                backend,
                prior,
                format_prompt(args.prompt, config.system_prompt),
                config,
            ).completion
        )
    else:
        if not args.input or not args.prior:
            parser.error("train requires --input filtered JSONL and --prior")
        train_sage(config, Path(args.input), Path(args.prior), Path(args.output))


if __name__ == "__main__":
    main()
