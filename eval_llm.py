"""Label-aware benchmark evaluation; SAGE guidance is intentionally absent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_sage.config import SAGEConfig
from llm_sage.data import format_prompt, load_jsonl, write_jsonl
from llm_sage.hf_backend import HuggingFaceBackend
from llm_sage.rewards import exact_answer_reward, multiple_choice_reward


def score(completion: str, answer: str, metric: str) -> float:
    if metric == "exact_answer":
        return exact_answer_reward(completion, answer)
    if metric == "multiple_choice":
        return multiple_choice_reward(completion, answer)
    raise ValueError(f"Unsupported metric: {metric}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    config = SAGEConfig.from_yaml(args.config)
    backend = HuggingFaceBackend(args.model)
    records = load_jsonl(args.input)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    if not records:
        raise ValueError("No evaluation records were found")
    results = []
    for item in records:
        if "answer" not in item:
            raise ValueError("Evaluation records require answer/target/ground_truth")
        prompt = format_prompt(item["prompt"], config.system_prompt)
        completion = backend.generate_direct(
            prompt,
            config.max_steps * config.max_step_tokens,
            config.max_prompt_tokens,
        )
        metric = item.get(
            "metric",
            "multiple_choice"
            if config.task_family == "natural"
            else "exact_answer",
        )
        result = {
            "prompt_id": item["prompt_id"],
            "completion": completion,
            "metric": metric,
            "correct": score(completion, item["answer"], metric),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))
    write_jsonl(results, args.output)
    accuracy = sum(result["correct"] for result in results) / max(len(results), 1)
    summary = {
        "num_samples": len(results),
        "accuracy": accuracy,
        "accuracy_percent": 100.0 * accuracy,
        "model": args.model,
        "task_family": config.task_family,
        "seed": config.seed,
        "guidance_at_inference": False,
    }
    Path(args.output).with_suffix(".metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
