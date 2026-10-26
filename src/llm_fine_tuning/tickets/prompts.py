"""
Author: L. Saetta
Date last modified: 2026-10-26
License: MIT
Description: Prompt variants, system-prompt loading, chat-message construction, adapter naming, and prompt metadata.
"""

import hashlib
import json
from pathlib import Path

PROMPT_FILES = {
    "none": None,
    "short": "system_prompt_short.txt",
    "full": "system_prompt_full.txt",
}
PROMPT_METADATA_FILE = "prompt_variant.json"
ADAPTER_BASE_NAME = "demo06-qwen3-1.7b-ticket-classification-lora"


def require_prompt_variant(prompt_variant: str) -> None:
    """Raise an error for an unknown prompt variant.

    Args:
        prompt_variant: Variant name.

    Raises:
        ValueError: If the variant is not one of :data:`PROMPT_FILES`.
    """
    if prompt_variant not in PROMPT_FILES:
        raise ValueError(f"The prompt variant must be one of {sorted(PROMPT_FILES)}, got {prompt_variant!r}.")


def load_system_prompt(prompts_directory: Path, prompt_variant: str) -> str | None:
    """Return the system prompt text of a variant, read and stripped exactly as during training.

    Args:
        prompts_directory: Directory containing the prompt files.
        prompt_variant: ``'none'``, ``'short'``, or ``'full'``.

    Returns:
        The prompt text, or ``None`` for the variant without a system prompt.

    Raises:
        ValueError: If the variant is unknown.
        FileNotFoundError: If the prompt file is missing.
    """
    require_prompt_variant(prompt_variant)
    file_name = PROMPT_FILES[prompt_variant]
    if file_name is None:
        return None
    prompt_file = prompts_directory / file_name
    if not prompt_file.is_file():
        raise FileNotFoundError(f"Missing system prompt file: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8").strip()


def build_prompt_messages(ticket_message: dict, system_prompt: str | None) -> list[dict]:
    """Return the chat prompt for one ticket: the optional system message followed by the ticket message.

    Training, validation, test evaluation, and inference must all build prompts with this function.

    Args:
        ticket_message: The record's ``user`` message.
        system_prompt: System prompt text, or ``None``.

    Returns:
        The list of chat messages.
    """
    system_messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    return [*system_messages, ticket_message]


def adapter_run_name(prompt_variant: str) -> str:
    """Return the training output directory name of a prompt variant.

    Variant ``'none'`` keeps the original name, so its adapter remains the variant A reference.

    Args:
        prompt_variant: Variant name.

    Returns:
        The directory name below the ticket-classification models directory.
    """
    require_prompt_variant(prompt_variant)
    return ADAPTER_BASE_NAME if prompt_variant == "none" else f"{ADAPTER_BASE_NAME}-prompt-{prompt_variant}"


def prompt_metadata(prompt_variant: str, system_prompt: str | None) -> dict:
    """Return the record that identifies the prompt an adapter was trained with.

    Args:
        prompt_variant: Variant name.
        system_prompt: The exact prompt text used, or ``None``.

    Returns:
        The variant, prompt file name, and SHA-256 of the prompt text.
    """
    require_prompt_variant(prompt_variant)
    return {
        "prompt_variant": prompt_variant,
        "prompt_file": PROMPT_FILES[prompt_variant],
        "prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest() if system_prompt else None,
    }


def write_prompt_metadata(adapter_directory: Path, prompt_variant: str, system_prompt: str | None) -> Path:
    """Write ``prompt_variant.json`` next to a saved adapter.

    Args:
        adapter_directory: Directory of the saved adapter.
        prompt_variant: Variant name.
        system_prompt: The exact prompt text used, or ``None``.

    Returns:
        The path of the written file.
    """
    metadata_file = adapter_directory / PROMPT_METADATA_FILE
    metadata_file.write_text(json.dumps(prompt_metadata(prompt_variant, system_prompt), indent=2) + "\n", encoding="utf-8")
    return metadata_file


def verify_adapter_prompt(adapter_directory: Path, prompt_variant: str, system_prompt: str | None) -> None:
    """Raise an error unless an adapter was trained with this prompt variant and exactly this prompt text.

    An adapter without ``prompt_variant.json`` predates specification 021 and is accepted only for ``'none'``.

    Args:
        adapter_directory: Directory of the saved adapter.
        prompt_variant: Variant the evaluation will use.
        system_prompt: Prompt text the evaluation will use, or ``None``.

    Raises:
        FileNotFoundError: If the metadata is missing for a variant other than ``'none'``.
        ValueError: If the recorded variant or prompt hash differs.
    """
    metadata_file = adapter_directory / PROMPT_METADATA_FILE
    if not metadata_file.is_file():
        if prompt_variant != "none":
            raise FileNotFoundError(f"Missing {metadata_file}. Retrain the adapter for prompt variant {prompt_variant!r}.")
        print("No prompt_variant.json found: treating the adapter as a variant A adapter trained before specification 021.")
        return
    recorded = json.loads(metadata_file.read_text(encoding="utf-8"))
    expected = prompt_metadata(prompt_variant, system_prompt)
    if recorded.get("prompt_variant") != expected["prompt_variant"] or recorded.get("prompt_sha256") != expected["prompt_sha256"]:
        raise ValueError(
            f"The adapter was trained with prompt variant {recorded.get('prompt_variant')!r} or a different prompt text; "
            f"the configured variant is {prompt_variant!r}."
        )


def render_generation_prompt(tokenizer, ticket_message: dict, system_prompt: str | None) -> str:
    """Render the text the model receives before generating, with Qwen3 thinking disabled.

    Args:
        tokenizer: Tokenizer with the Qwen3 chat template.
        ticket_message: The record's ``user`` message.
        system_prompt: System prompt text, or ``None``.

    Returns:
        The rendered prompt, ending with the assistant turn opening.
    """
    return tokenizer.apply_chat_template(
        build_prompt_messages(ticket_message, system_prompt),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def longest_sequence_length(tokenizer, splits, system_prompt: str | None) -> int:
    """Return the token count of the longest formatted prompt plus completion across splits.

    Args:
        tokenizer: Tokenizer with the chat template.
        splits: Iterables of raw records.
        system_prompt: System prompt text, or ``None``.

    Returns:
        The largest token count.
    """
    longest = 0
    for records in splits:
        for record in records:
            conversation = [*build_prompt_messages(record["messages"][0], system_prompt), record["messages"][1]]
            text = tokenizer.apply_chat_template(conversation, tokenize=False)
            longest = max(longest, len(tokenizer(text)["input_ids"]))
    return longest


def require_sequences_fit(tokenizer, splits, system_prompt: str | None, max_sequence_length: int) -> int:
    """Fail before training if truncation would cut any formatted example.

    Args:
        tokenizer: Tokenizer with the chat template.
        splits: Iterables of raw records.
        system_prompt: System prompt text, or ``None``.
        max_sequence_length: The trainer's ``max_length``.

    Returns:
        The longest sequence length, which fits.

    Raises:
        ValueError: If the longest sequence exceeds ``max_sequence_length``.
    """
    longest = longest_sequence_length(tokenizer, splits, system_prompt)
    if longest > max_sequence_length:
        raise ValueError(
            f"The longest formatted sequence has {longest} tokens, above MAX_SEQUENCE_LENGTH={max_sequence_length}. "
            "Raise MAX_SEQUENCE_LENGTH or use a shorter prompt variant."
        )
    return longest
