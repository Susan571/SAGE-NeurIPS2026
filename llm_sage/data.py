from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def _first(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in item and item[key] not in (None, ""):
            return item[key]
    return None


def normalize_record(item: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """Normalize common reasoning JSONL schemas without leaking labels."""
    extra = item.get("extra_info") if isinstance(item.get("extra_info"), dict) else {}
    prompt = _first(item, ("prompt", "problem", "question", "query"))
    if isinstance(prompt, list):
        prompt = "\n".join(
            str(message.get("content", ""))
            for message in prompt
            if isinstance(message, dict)
            and message.get("role", "user") in {"system", "user"}
        )
    if prompt is None:
        raise ValueError(f"Record {index} has no prompt/problem/question/query")
    choices = _first(item, ("options", "choices"))
    if isinstance(choices, dict):
        rendered_choices = [
            f"{label}. {text}" for label, text in choices.items()
        ]
    elif isinstance(choices, list):
        rendered_choices = [
            f"{chr(ord('A') + choice_index)}. {text}"
            for choice_index, text in enumerate(choices)
        ]
    else:
        rendered_choices = []
    if rendered_choices:
        prompt = str(prompt) + "\n" + "\n".join(rendered_choices)
    answer = _first(item, ("answer", "target", "ground_truth", "gt"))
    if answer is None:
        answer = _first(extra, ("answer", "target", "ground_truth", "gt"))
    normalized = dict(item)
    normalized["prompt_id"] = item.get("prompt_id", item.get("id", index))
    normalized["prompt"] = str(prompt)
    if answer is not None:
        normalized["answer"] = str(answer)
    return normalized


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open() as stream:
        for index, line in enumerate(stream):
            if line.strip():
                yield normalize_record(json.loads(line), index)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def format_prompt(prompt: str, system_prompt: str) -> str:
    prompt = prompt.strip()
    system_prompt = system_prompt.strip()
    if not system_prompt or prompt.startswith(system_prompt):
        return prompt
    return system_prompt + "\n\n" + prompt


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
