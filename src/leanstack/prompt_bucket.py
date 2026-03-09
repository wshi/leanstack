from __future__ import annotations

from typing import Any


def build_exact_prompt_text(
    tokenizer: Any,
    prompt_text: str,
    target_tokens: int,
    *,
    separator: str = "\n\n",
    max_expansions: int = 64,
) -> tuple[str, int]:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")

    candidate = prompt_text.strip()
    if not candidate:
        raise ValueError("prompt_text must not be empty")

    token_ids = tokenizer(candidate, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    expansions = 0
    while int(token_ids.shape[0]) < target_tokens:
        expansions += 1
        if expansions > max_expansions:
            raise ValueError(
                f"could not reach target_tokens={target_tokens} from base prompt after {max_expansions} expansions"
            )
        candidate = candidate + separator + prompt_text.strip()
        token_ids = tokenizer(candidate, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

    exact_token_ids = token_ids[:target_tokens]
    exact_prompt_text = tokenizer.decode(exact_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    exact_prompt_tokens = tokenizer(
        exact_prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]

    if int(exact_prompt_tokens.shape[0]) != target_tokens:
        raise ValueError(
            "decode/encode prompt bucket mismatch: "
            f"expected {target_tokens} tokens, got {int(exact_prompt_tokens.shape[0])}"
        )

    return exact_prompt_text, int(exact_prompt_tokens.shape[0])
