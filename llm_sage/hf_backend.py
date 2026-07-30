from __future__ import annotations

from typing import Sequence

import torch

from .grpo import step_logprobs
from .sampling import CandidateStep


class HuggingFaceBackend:
    """HF backend for Appendix E.2 finite candidate-step generation."""

    def __init__(self, model_name: str, device: str | None = None):
        try:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install optional LLM dependencies: pip install -e '.[llm]'"
            ) from exc
        model_config = AutoConfig.from_pretrained(model_name)
        if model_config.model_type == "qwen3_5":
            try:
                from transformers import AutoModelForMultimodalLM
            except ImportError as exc:
                raise RuntimeError(
                    "Qwen3.5 requires transformers>=5.12.0; reinstall with "
                    "pip install -e '.[llm]'"
                ) from exc
            model_class = AutoModelForMultimodalLM
        else:
            model_class = AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        load_kwargs = {"dtype": "auto"}
        if device is None:
            load_kwargs["device_map"] = "auto"
        self.model = model_class.from_pretrained(model_name, **load_kwargs)
        if device is not None:
            self.model.to(device)
        if hasattr(self.model.config, "text_config"):
            self.model.config.text_config.use_cache = True
        else:
            self.model.config.use_cache = True
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def generate_completions(
        self,
        prompt: str,
        count: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        max_prompt_tokens: int | None = None,
    ) -> list[str]:
        """Sample complete reference rollouts for fixed-subset filtering."""
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=max_prompt_tokens is not None,
            max_length=max_prompt_tokens,
        ).to(self.device)
        generated = self.model.generate(
            **encoded,
            do_sample=True,
            num_return_sequences=count,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_length = encoded.input_ids.shape[1]
        return [
            self.tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True)
            for sequence in generated
        ]

    @torch.no_grad()
    def generate_direct(
        self, prompt: str, max_new_tokens: int, max_prompt_tokens: int | None = None
    ) -> str:
        """Greedy inference with no SAGE modules, matching Appendix E."""
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=max_prompt_tokens is not None,
            max_length=max_prompt_tokens,
        ).to(self.device)
        generated = self.model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(
            generated[0, encoded.input_ids.shape[1] :], skip_special_tokens=True
        )

    @torch.no_grad()
    def hidden(
        self,
        texts: Sequence[str],
        layer: int = -1,
        max_length: int | None = None,
    ) -> torch.Tensor:
        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)
        output = self.model(**encoded, output_hidden_states=True)
        last_indices = encoded.attention_mask.sum(1) - 1
        return output.hidden_states[layer][
            torch.arange(len(texts), device=self.device), last_indices
        ].float()

    def hidden_batched(
        self,
        texts: Sequence[str],
        layer: int = -1,
        batch_size: int = 8,
        max_length: int | None = None,
    ) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        batches = [
            self.hidden(
                texts[start : start + batch_size], layer, max_length
            ).cpu()
            for start in range(0, len(texts), batch_size)
        ]
        if not batches:
            raise ValueError("texts must not be empty")
        return torch.cat(batches)

    @torch.no_grad()
    def candidate_steps(
        self,
        state: str,
        count: int,
        max_step_tokens: int,
        temperature: float,
        top_p: float,
        layer: int = -1,
        hidden_backend: "HuggingFaceBackend | None" = None,
        max_state_tokens: int | None = None,
    ) -> list[CandidateStep]:
        encoded = self.tokenizer(
            state,
            return_tensors="pt",
            truncation=max_state_tokens is not None,
            max_length=max_state_tokens,
        ).to(self.device)
        generated = self.model.generate(
            **encoded,
            do_sample=True,
            num_return_sequences=count,
            max_new_tokens=max_step_tokens,
            temperature=temperature,
            top_p=top_p,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_len = encoded.input_ids.shape[1]
        transition_scores = self.model.compute_transition_scores(
            generated.sequences, generated.scores, normalize_logits=True
        )
        texts, tokens, logprobs = [], [], []
        for sequence, scores in zip(generated.sequences, transition_scores):
            new_tokens = sequence[prompt_len:]
            eos_positions = (
                (new_tokens == self.tokenizer.eos_token_id).nonzero()
                if self.tokenizer.eos_token_id is not None
                else torch.empty(0, 1, dtype=torch.long, device=new_tokens.device)
            )
            if len(eos_positions):
                generation_length = int(eos_positions[0].item()) + 1
                new_tokens = new_tokens[:generation_length]
                scores = scores[:generation_length]
            full_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            boundary = _step_boundary(full_text)
            retained_count = _token_count_for_character_boundary(
                self.tokenizer, new_tokens, boundary
            )
            retained_count = max(1, min(retained_count, len(scores)))
            retained_tokens = new_tokens[:retained_count]
            text = self.tokenizer.decode(
                retained_tokens, skip_special_tokens=True
            )
            texts.append(text)
            tokens.append(retained_tokens.detach())
            logprobs.append(scores[:retained_count].detach())
        encoder = hidden_backend or self
        next_hidden = encoder.hidden(
            [state + text for text in texts],
            layer,
            None if max_state_tokens is None else max_state_tokens + max_step_tokens,
        )
        eos_id = self.tokenizer.eos_token_id
        return [
            CandidateStep(
                text,
                token_ids,
                scores,
                hidden,
                not text.strip()
                or (
                    eos_id is not None
                    and bool((token_ids == eos_id).any())
                ),
            )
            for text, token_ids, scores, hidden in zip(
                texts, tokens, logprobs, next_hidden
            )
        ]

    def encode_prompt_completions(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
        max_length: int,
        steps: Sequence[Sequence[str]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize pairs and mark completion tokens and reasoning-step ids."""
        if len(prompts) != len(completions):
            raise ValueError("prompts and completions must have equal length")
        if steps is not None and len(steps) != len(prompts):
            raise ValueError("steps and prompts must have equal length")
        use_offsets = bool(getattr(self.tokenizer, "is_fast", False))
        encoded = self.tokenizer(
            [p + c for p, c in zip(prompts, completions)],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
            return_offsets_mapping=use_offsets,
        )
        offsets = encoded.pop("offset_mapping", None)
        encoded = encoded.to(self.device)
        completion_mask = torch.zeros_like(encoded.input_ids)
        step_ids = torch.zeros_like(encoded.input_ids)
        for row, prompt in enumerate(prompts):
            sequence_length = int(encoded.attention_mask[row].sum())
            if offsets is not None:
                step_ends = []
                cumulative_end = len(prompt)
                for step in steps[row] if steps is not None else [completions[row]]:
                    cumulative_end += len(step)
                    step_ends.append(cumulative_end)
                for token_index in range(sequence_length):
                    start, end = map(int, offsets[row, token_index])
                    if end <= len(prompt) or end <= start:
                        continue
                    completion_mask[row, token_index] = 1
                    step_id = next(
                        (
                            index
                            for index, step_end in enumerate(step_ends, start=1)
                            if end <= step_end
                        ),
                        max(len(step_ends), 1),
                    )
                    step_ids[row, token_index] = step_id
                continue
            prompt_length = len(
                self.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=True,
                ).input_ids
            )
            completion_mask[row, min(prompt_length, sequence_length) : sequence_length] = 1
            if steps is None:
                step_ids[row, min(prompt_length, sequence_length) : sequence_length] = 1
                continue
            previous = min(prompt_length, sequence_length)
            cumulative = prompt
            for step_index, step in enumerate(steps[row], start=1):
                cumulative += step
                boundary = len(
                    self.tokenizer(
                        cumulative,
                        truncation=True,
                        max_length=max_length,
                        add_special_tokens=True,
                    ).input_ids
                )
                boundary = min(boundary, sequence_length)
                step_ids[row, previous:boundary] = step_index
                previous = boundary
            if previous < sequence_length:
                fallback_step = max(len(steps[row]), 1)
                step_ids[row, previous:sequence_length] = fallback_step
        return (
            encoded.input_ids,
            encoded.attention_mask,
            completion_mask,
            step_ids,
        )

    def encode_rollout_steps(
        self,
        prompts: Sequence[str],
        rollout_step_token_ids: Sequence[Sequence[torch.Tensor]],
        max_prompt_tokens: int,
        max_completion_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replay the exact tokens sampled for each coarse reasoning action."""
        if len(prompts) != len(rollout_step_token_ids):
            raise ValueError("prompts and rollout steps must have equal length")
        sequences: list[list[int]] = []
        sequence_step_ids: list[list[int]] = []
        for prompt, steps in zip(prompts, rollout_step_token_ids):
            prompt_ids = self.tokenizer(
                prompt,
                truncation=True,
                max_length=max_prompt_tokens,
                add_special_tokens=True,
            ).input_ids
            completion_ids: list[int] = []
            completion_steps: list[int] = []
            for step_index, token_ids in enumerate(steps, start=1):
                remaining = max_completion_tokens - len(completion_ids)
                if remaining <= 0:
                    break
                retained = token_ids.detach().cpu().tolist()[:remaining]
                completion_ids.extend(retained)
                completion_steps.extend([step_index] * len(retained))
            sequences.append(prompt_ids + completion_ids)
            sequence_step_ids.append([0] * len(prompt_ids) + completion_steps)
        max_length = max(map(len, sequences))
        input_ids = torch.full(
            (len(sequences), max_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        step_ids = torch.zeros_like(input_ids)
        for row, (sequence, ids) in enumerate(zip(sequences, sequence_step_ids)):
            length = len(sequence)
            input_ids[row, :length] = torch.tensor(sequence, device=self.device)
            attention_mask[row, :length] = 1
            step_ids[row, :length] = torch.tensor(ids, device=self.device)
        completion_mask = step_ids > 0
        return input_ids, attention_mask, completion_mask, step_ids

    def encode_rollout_actions(
        self,
        rollout_states: Sequence[Sequence[str]],
        rollout_step_token_ids: Sequence[Sequence[torch.Tensor]],
        max_state_tokens: int,
        max_action_tokens: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Encode each coarse action with the exact state used to sample it."""
        if len(rollout_states) != len(rollout_step_token_ids):
            raise ValueError("rollout states and steps must have equal length")
        sequences: list[list[int]] = []
        masks: list[list[int]] = []
        rollout_ids: list[int] = []
        for rollout_index, (states, steps) in enumerate(
            zip(rollout_states, rollout_step_token_ids)
        ):
            if len(states) != len(steps):
                raise ValueError("each sampled step must have one pre-action state")
            for state, token_ids in zip(states, steps):
                state_ids = self.tokenizer(
                    state,
                    truncation=True,
                    max_length=max_state_tokens,
                    add_special_tokens=True,
                ).input_ids
                action_ids = token_ids.detach().cpu().tolist()[:max_action_tokens]
                sequences.append(state_ids + action_ids)
                masks.append([0] * len(state_ids) + [1] * len(action_ids))
                rollout_ids.append(rollout_index)
        if not sequences:
            raise ValueError("rollouts contain no sampled actions")
        max_length = max(map(len, sequences))
        input_ids = torch.full(
            (len(sequences), max_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        completion_mask = torch.zeros_like(input_ids)
        for row, (sequence, mask) in enumerate(zip(sequences, masks)):
            length = len(sequence)
            input_ids[row, :length] = torch.tensor(sequence, device=self.device)
            attention_mask[row, :length] = 1
            completion_mask[row, :length] = torch.tensor(mask, device=self.device)
        step_ids = completion_mask.long()
        return (
            input_ids,
            attention_mask,
            completion_mask,
            step_ids,
            torch.tensor(rollout_ids, device=self.device),
        )

    @torch.no_grad()
    def completion_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
        step_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            use_cache=False,
        )
        return step_logprobs(
            output.logits,
            input_ids.to(self.device),
            completion_mask.to(self.device),
            step_ids.to(self.device),
        )


def _step_boundary(text: str) -> int:
    newline = text.find("\n")
    sentence = text.find(". ")
    endpoints = []
    if newline >= 0:
        endpoints.append(newline + 1)
    if sentence >= 0:
        endpoints.append(sentence + 1)
    boxed = text.find(r"\boxed{")
    if boxed >= 0:
        depth = 0
        for index in range(boxed + len(r"\boxed{"), len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                if depth == 0:
                    endpoints.append(index + 1)
                    break
                depth -= 1
    return min(endpoints) if endpoints else len(text)


def _token_count_for_character_boundary(tokenizer, tokens: torch.Tensor, end: int) -> int:
    """Find the generated-token prefix covering a decoded character boundary."""
    if end >= len(tokenizer.decode(tokens, skip_special_tokens=True)):
        return len(tokens)
    for count in range(1, len(tokens) + 1):
        decoded = tokenizer.decode(tokens[:count], skip_special_tokens=True)
        if len(decoded) >= end:
            return count
    return len(tokens)
