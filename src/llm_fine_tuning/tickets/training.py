"""
Author: L. Saetta
Date last modified: 2026-11-09
License: MIT
Description: Training settings, the learning-rate schedule plan, LoRA and SFT configuration, per-epoch validation
metrics, and adapter saving.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from llm_fine_tuning.display import show_table
from llm_fine_tuning.tickets.evaluation import evaluate_model
from llm_fine_tuning.tickets.prompts import write_prompt_metadata

BEST_CHECKPOINT_METRIC = "eval_joint_accuracy"
EPOCH_TABLE_HEADERS = [
    "Epoch", "Training Loss", "Validation Loss", "Category Accuracy", "Severity Accuracy",
    "Category+Severity Accuracy", "Valid JSON Rate",
]


@dataclass(frozen=True)
class TrainingSettings:
    """Values a learner typically changes for one LoRA training run.

    Attributes:
        prompt_variant: ``'none'`` (A), ``'short'`` (B), or ``'full'`` (C); see ``ticket-classification/prompts``.
        num_train_epochs: Passes over the training split.
        learning_rate: Peak AdamW learning rate, reached at the end of the warmup.
        lr_scheduler_type: Shape of the learning-rate decay after the warmup, for example ``'cosine'``.
        warmup_ratio: Fraction of the total optimizer steps during which the learning rate rises linearly from 0 to
            ``learning_rate``; must be in ``[0, 1)``.
        per_device_train_batch_size: Examples per forward pass (micro-batch).
        gradient_accumulation_steps: Micro-batches per optimizer step.
        per_device_eval_batch_size: Examples per forward pass for validation loss.
        max_sequence_length: Maximum tokens per formatted example; longer examples stop the run.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling numerator; the update is scaled by ``lora_alpha / lora_r``.
        lora_dropout: Dropout on the LoRA input.
        lora_target_modules: Linear layers that receive LoRA adapters.
        generation_max_new_tokens: Maximum generated tokens per validation response.
        validation_generation_batch_size: Validation tickets generated per batch after every epoch.
        seed: Random seed for reproducibility; also the seed of the training-data order.

    Raises:
        ValueError: If ``warmup_ratio`` is outside ``[0, 1)``.
    """

    prompt_variant: str = "none"
    num_train_epochs: int = 5
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.10
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    per_device_eval_batch_size: int = 4
    max_sequence_length: int = 512
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    generation_max_new_tokens: int = 128
    validation_generation_batch_size: int = 4
    seed: int = 42

    def __post_init__(self):
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError(f"warmup_ratio must be in [0, 1), got {self.warmup_ratio!r}.")

    @property
    def effective_batch_size(self) -> int:
        """Examples per optimizer step."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


def learning_rate_schedule(num_train_examples: int, settings: TrainingSettings) -> dict:
    """Return the optimizer-step plan of the learning-rate schedule, computed as the trainer computes it.

    The formulas mirror ``Trainer.set_initial_training_values`` and ``TrainingArguments.get_warmup_steps`` of
    Transformers for one device.

    Args:
        num_train_examples: Records in the training split.
        settings: Training settings.

    Returns:
        Steps per epoch, total optimizer steps, warmup steps, and the schedule settings.

    Raises:
        ValueError: If the training split is empty.
    """
    if num_train_examples < 1:
        raise ValueError("The training split is empty.")
    batches_per_epoch = math.ceil(num_train_examples / settings.per_device_train_batch_size)
    steps_per_epoch = max(math.ceil(batches_per_epoch / settings.gradient_accumulation_steps), 1)
    total_steps = steps_per_epoch * settings.num_train_epochs
    return {
        "Scheduler": settings.lr_scheduler_type,
        "Peak learning rate": settings.learning_rate,
        "Warmup ratio": settings.warmup_ratio,
        "Optimizer steps per epoch": steps_per_epoch,
        "Total optimizer steps": total_steps,
        "Warmup steps": math.ceil(total_steps * settings.warmup_ratio),
    }


def add_lora_adapter(model, settings: TrainingSettings):
    """Disable the generation cache for training and wrap the model with a trainable BF16 LoRA adapter.

    Args:
        model: Base model in its training dtype.
        settings: Training settings.

    Returns:
        The PEFT model.
    """
    model.config.use_cache = False
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=settings.lora_r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        bias="none",
        target_modules=list(settings.lora_target_modules),
    )
    return get_peft_model(model, lora_config, autocast_adapter_dtype=False)


def build_sft_config(output_directory: Path, settings: TrainingSettings) -> SFTConfig:
    """Return the trainer configuration: settings from the notebook plus the fixed project policy.

    Fixed policy: evaluate and save every epoch, keep one checkpoint besides the best, restore the epoch with the
    highest validation Category+Severity Accuracy, compute the loss on the completion only, use gradient
    checkpointing, and run in BF16 on MPS.

    Args:
        output_directory: Directory for checkpoints and logs.
        settings: Training settings.

    Returns:
        The SFT configuration.
    """
    return SFTConfig(
        output_dir=str(output_directory),
        learning_rate=settings.learning_rate,
        lr_scheduler_type=settings.lr_scheduler_type,
        warmup_ratio=settings.warmup_ratio,
        num_train_epochs=settings.num_train_epochs,
        per_device_train_batch_size=settings.per_device_train_batch_size,
        per_device_eval_batch_size=settings.per_device_eval_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        max_length=settings.max_sequence_length,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=5,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=BEST_CHECKPOINT_METRIC,
        greater_is_better=True,
        completion_only_loss=True,
        gradient_checkpointing=True,
        bf16=True,
        bf16_full_eval=True,
        dataloader_pin_memory=False,
        optim="adamw_torch",
        report_to="none",
        seed=settings.seed,
        data_seed=settings.seed,
    )


def epoch_table_row(metrics: dict) -> list[str]:
    """Format one epoch of training and validation metrics as a table row.

    Args:
        metrics: Epoch, losses, and the four classification metrics.

    Returns:
        The formatted row.
    """
    return [
        f"{metrics['epoch']:.0f}", f"{metrics['training_loss']:.6f}", f"{metrics['validation_loss']:.6f}",
        f"{metrics['category_accuracy']:.1%}", f"{metrics['severity_accuracy']:.1%}",
        f"{metrics['joint_accuracy']:.1%}", f"{metrics['valid_json_rate']:.1%}",
    ]


class ValidationMetricsCallback(TrainerCallback):
    """After every validation-loss pass, generate validation responses, score them, and show the epoch row.

    The joint accuracy is added to the evaluation metrics as ``eval_joint_accuracy``, which the trainer uses to
    select the best checkpoint.
    """

    def __init__(self, model, tokenizer, validation_records, system_prompt, device: torch.device, settings: TrainingSettings):
        self.model = model
        self.tokenizer = tokenizer
        self.validation_records = validation_records
        self.system_prompt = system_prompt
        self.device = device
        self.settings = settings
        self.epoch_metrics = []

    def on_evaluate(self, args, state, control, **kwargs):
        evaluation_metrics = kwargs.get("metrics", {})
        generation_metrics, _ = evaluate_model(
            self.model, self.tokenizer, self.validation_records, self.system_prompt, self.device,
            self.settings.validation_generation_batch_size, self.settings.generation_max_new_tokens,
        )
        training_loss = next((entry["loss"] for entry in reversed(state.log_history) if "loss" in entry), float("nan"))
        epoch_metrics = {"epoch": state.epoch, "training_loss": training_loss, "validation_loss": evaluation_metrics["eval_loss"], **generation_metrics}
        evaluation_metrics[BEST_CHECKPOINT_METRIC] = generation_metrics["joint_accuracy"]
        self.epoch_metrics.append(epoch_metrics)
        show_table(EPOCH_TABLE_HEADERS, [epoch_table_row(epoch_metrics)])
        return control


def create_trainer(model, tokenizer, raw_splits, sft_splits, system_prompt, device, settings: TrainingSettings, output_directory: Path):
    """Build the SFT trainer with the per-epoch validation metrics callback.

    Args:
        model: PEFT model returned by :func:`add_lora_adapter`.
        tokenizer: Matching tokenizer.
        raw_splits: Raw ``train`` and ``validation`` records, used for generation metrics.
        sft_splits: Prompt/completion records, used for training and validation loss.
        system_prompt: System prompt text, or ``None``.
        device: Training device.
        settings: Training settings.
        output_directory: Directory for checkpoints and logs.

    Returns:
        A tuple of the trainer and the metrics callback.
    """
    callback = ValidationMetricsCallback(model, tokenizer, raw_splits["validation"], system_prompt, device, settings)
    trainer = SFTTrainer(
        model=model,
        args=build_sft_config(output_directory, settings),
        train_dataset=sft_splits["train"],
        eval_dataset=sft_splits["validation"],
        processing_class=tokenizer,
        callbacks=[callback],
    )
    return trainer, callback


def save_adapter(trainer, tokenizer, adapter_directory: Path, prompt_variant: str, system_prompt: str | None) -> None:
    """Save the restored best adapter, its tokenizer, and the prompt it was trained with.

    Args:
        trainer: Trainer after ``train()`` has restored the best checkpoint.
        tokenizer: Tokenizer to save with the adapter.
        adapter_directory: Destination directory.
        prompt_variant: Variant name.
        system_prompt: Prompt text used for training, or ``None``.
    """
    trainer.save_model(adapter_directory)
    tokenizer.save_pretrained(adapter_directory)
    write_prompt_metadata(adapter_directory, prompt_variant, system_prompt)


def show_epoch_metrics(callback: ValidationMetricsCallback, log_history: list[dict]) -> None:
    """Show every epoch recorded by the callback in one table.

    Args:
        callback: Callback used during training.
        log_history: ``trainer.state.log_history``, used to check that every validation pass has metrics.

    Raises:
        RuntimeError: If an epoch is missing its task-specific metrics.
    """
    validation_passes = [entry for entry in log_history if "eval_loss" in entry and "epoch" in entry]
    if len(callback.epoch_metrics) != len(validation_passes):
        raise RuntimeError("Task-specific validation metrics are unavailable for one or more epochs.")
    show_table(EPOCH_TABLE_HEADERS, [epoch_table_row(metrics) for metrics in callback.epoch_metrics])
