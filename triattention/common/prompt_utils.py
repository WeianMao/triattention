"""Shared prompt utilities to keep calibration and inference aligned."""
from __future__ import annotations

from typing import Iterable, List, Sequence

from transformers import AutoTokenizer

PROMPT_TEMPLATE = (
    "You are given a math problem.\n\nProblem: {question}\n\n "
    "You need to solve the problem step by step. First, you need to provide the chain-of-thought, "
    "then provide the final answer.\n\n Provide the final answer in the format: Final answer:  \\boxed{{}}"
)
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def extract_question_from_record(
    record: dict,
    *,
    fallback_keys: Sequence[str] | None = None,
) -> str:
    """Return a question string, tolerating LazyEviction-style multi-message JSON."""
    keys: List[str] = []
    if fallback_keys:
        keys.extend(list(fallback_keys))
    keys.extend(["question", "problem"])
    for key in keys:
        value = record.get(key)
        if value:
            return str(value)

    messages = record.get("messages")
    if isinstance(messages, Iterable) and not isinstance(messages, (str, bytes)):
        segments = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "system"} and content:
                segments.append(str(content))
        if segments:
            return "\n\n".join(segments)

    raise KeyError("Unable to extract question from record (missing question/problem/messages).")


def build_plain_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


def build_prompt(
    tokenizer: AutoTokenizer,
    question: str,
    *,
    use_chat_template: bool,
    system_prompt: str,
) -> str:
    base_prompt = build_plain_prompt(question)
    if not use_chat_template:
        return base_prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": base_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_prompt_with_response(
    tokenizer: AutoTokenizer,
    question: str,
    response: str,
    *,
    use_chat_template: bool,
    system_prompt: str,
) -> str:
    prompt = build_prompt(
        tokenizer,
        question,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
    )
    return prompt + response
