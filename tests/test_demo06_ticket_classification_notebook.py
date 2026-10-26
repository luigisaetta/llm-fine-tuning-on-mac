"""
Author: L. Saetta
Date last modified: 2026-10-26
License: MIT
Description: Structure tests for the Demo 06 ticket-classification LoRA training notebook.
"""

from notebook_structure import assert_didactic_structure, load_notebook


def test_demo06_is_a_didactic_training_notebook() -> None:
    """Demo 06 exposes its settings, uses only training data, and runs every training step through the package."""
    source = assert_didactic_structure(load_notebook("ticket-classification/demo06_ticket_classification_lora_fine_tuning.ipynb"), "settings = TrainingSettings(")
    assert "test_split" not in source and "test_dataset" not in source and "load_test_split" not in source
    for expected in (
        'prompt_variant="none"', "num_train_epochs=8", "per_device_train_batch_size=4", "gradient_accumulation_steps=2",
        "max_sequence_length=512", "require_bfloat16_mps_device()", "load_training_splits(paths)", "to_prompt_completion(raw_splits, system_prompt)",
        "require_sequences_fit(", "add_lora_adapter(", "create_trainer(", "require_parameter_dtypes(", "trainer.train()",
        "save_adapter(trainer, tokenizer, adapter_directory, settings.prompt_variant, system_prompt)", "plot_loss_curves(",
        "show_epoch_metrics(", "severity_confusion_matrix(", "train_samples_per_second",
    ):
        assert expected in source, expected
