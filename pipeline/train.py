"""
train.py — Fine-tune a Whisper model on Hindi audio data.

Reads manifest.json produced by align.py, fine-tunes the base Whisper model,
and saves weights to the output directory specified in the YAML config.

All parameters come from config — no hardcoded values.
"""

import json
import logging
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Callback: mirror only the LATEST checkpoint to Drive
# ─────────────────────────────────────────────
# Defined lazily inside a factory function because it needs to subclass
# transformers.TrainerCallback, which is only imported once transformers
# is actually available (see run() below). This keeps the top of the file
# import-light.

def _make_drive_backup_callback(TrainerCallback):
    class DriveBackupCallback(TrainerCallback):
        """
        After every checkpoint save, copies ONLY the newest checkpoint folder
        to a Drive-backed path, deleting any previously synced checkpoint first.
        This keeps Drive usage capped at ~1 checkpoint's worth of space, even
        though training itself happens on fast local disk.
        """

        def __init__(self, local_output_dir: str, drive_backup_dir: str):
            self.local_output_dir = local_output_dir
            self.drive_backup_dir = drive_backup_dir

        def on_save(self, args, state, control, **kwargs):
            latest = _find_latest_checkpoint(self.local_output_dir)
            if not latest:
                return control

            Path(self.drive_backup_dir).mkdir(parents=True, exist_ok=True)

            # Remove any previously synced checkpoint(s) so Drive never
            # accumulates more than one at a time.
            for existing in Path(self.drive_backup_dir).glob("checkpoint-*"):
                logger.info(f"Removing previous Drive-synced checkpoint: {existing}")
                shutil.rmtree(existing, ignore_errors=True)

            dest = os.path.join(self.drive_backup_dir, os.path.basename(latest))
            logger.info(f"Syncing latest checkpoint to Drive: {latest} -> {dest}")
            shutil.copytree(latest, dest)

            return control

    return DriveBackupCallback


# ─────────────────────────────────────────────
# Entry point called by run_pipeline.py
# ─────────────────────────────────────────────

def run(config: dict) -> None:
    """
    Main entry point. `config` is the parsed YAML dict.
    Expected config keys:
        base_model        : HuggingFace model id, e.g. "openai/whisper-small"
        task              : "transcribe"
        language          : "hi"
        eval_split        : fraction of data held out for evaluation, e.g. 0.1
        output_dir        : where to save fine-tuned weights, e.g. "models/hindi_stt"
        paths.manifest    : path to manifest.json
        kaggle_input_dir  : (optional) if set, audio paths in manifest are remapped
                            to this Kaggle dataset mount point. Leave unset for
                            Colab/local runs where paths are used as-is.
        training (optional):
            num_train_epochs               : default 3
            per_device_train_batch_size    : default 8
            per_device_eval_batch_size     : default 8
            learning_rate                  : default 1e-5
            warmup_steps                   : default 500
            save_steps                     : default 500
            eval_steps                     : default 500
            logging_steps                  : default 25
            fp16                           : default True (set False if no GPU)
            gradient_checkpointing         : default True
            dataloader_num_workers         : default 0
    """

    logger.info("=== train.py starting ===")

    # ── 1. Validate GPU availability ───────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning(
            "No GPU detected — training on CPU. This will be very slow. "
            "Consider Google Colab (free GPU) or Kaggle instead."
        )
    else:
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")

    # ── 2. Pull values from config ─────────────────────────────────────────
    base_model  = config["base_model"]          # e.g. "openai/whisper-small"
    task        = config.get("task", "transcribe")
    language    = config.get("language", "hi")
    eval_split  = float(config.get("eval_split", 0.1))
    output_dir  = config["output_dir"]          # e.g. "models/hindi_stt"

    manifest_path = config["paths"]["manifest"]  # e.g. "data/processed/hi/manifest.json"

    # Optional: if set, audio paths in manifest.json are remapped to this
    # Kaggle dataset mount directory. Audio was uploaded as chunked zips
    # (audio_chunk_00.zip, audio_chunk_01.zip, ...) which Kaggle auto-extracts
    # to subfolders of this directory. Leave unset for Colab/local runs.
    kaggle_input_dir = config.get("kaggle_input_dir", None)
    if kaggle_input_dir:
        logger.info(f"Kaggle mode: audio paths will be remapped to {kaggle_input_dir}")

    # Optional training hyperparameters — all have sensible defaults
    train_cfg = config.get("training", {})
    num_epochs          = int(train_cfg.get("num_train_epochs", 3))
    train_batch_size    = int(train_cfg.get("per_device_train_batch_size", 8))
    eval_batch_size     = int(train_cfg.get("per_device_eval_batch_size", 8))
    learning_rate       = float(train_cfg.get("learning_rate", 1e-5))
    warmup_steps        = int(train_cfg.get("warmup_steps", 500))
    save_steps          = int(train_cfg.get("save_steps", 500))
    eval_steps          = int(train_cfg.get("eval_steps", 500))
    logging_steps       = int(train_cfg.get("logging_steps", 25))
    fp16                = bool(train_cfg.get("fp16", device == "cuda"))
    grad_checkpointing  = bool(train_cfg.get("gradient_checkpointing", True))
    num_workers         = int(train_cfg.get("dataloader_num_workers", 0))

    # Optional: if set, only the LATEST checkpoint is mirrored here after each
    # save (older synced checkpoints are deleted first). Lets you train to fast
    # local disk while still having a resumable copy on persistent storage
    # (e.g. Google Drive) without accumulating multiple checkpoints there.
    drive_backup_dir = train_cfg.get("drive_backup_dir", None)

    logger.info(f"Base model  : {base_model}")
    logger.info(f"Manifest    : {manifest_path}")
    logger.info(f"Output dir  : {output_dir}")
    logger.info(f"Eval split  : {eval_split}")
    logger.info(f"Epochs      : {num_epochs}")

    # ── 3. Load manifest ───────────────────────────────────────────────────
    records = _load_manifest(manifest_path)
    logger.info(f"Loaded {len(records)} records from manifest")

    train_records, eval_records = _train_eval_split(records, eval_split)
    logger.info(f"Train: {len(train_records)}  |  Eval: {len(eval_records)}")

    # ── 4. Load processor + model ──────────────────────────────────────────
    # Import here so the file can be imported without heavy deps installed
    from transformers import (
        WhisperFeatureExtractor,
        WhisperTokenizer,
        WhisperProcessor,
        WhisperForConditionalGeneration,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
        TrainerCallback,
    )
    import evaluate

    logger.info(f"Loading processor from {base_model} …")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model)
    tokenizer = WhisperTokenizer.from_pretrained(
        base_model, language=language, task=task
    )
    processor = WhisperProcessor.from_pretrained(
        base_model, language=language, task=task
    )

    logger.info(f"Loading model from {base_model} …")
    model = WhisperForConditionalGeneration.from_pretrained(base_model)
    model.config.forced_decoder_ids = None       # let the tokenizer handle language forcing
    model.config.suppress_tokens = []

    if grad_checkpointing:
        model.config.use_cache = False           # required when gradient checkpointing is on

    # ── 5. Build HuggingFace datasets ─────────────────────────────────────
    from datasets import Dataset

    logger.info("Preparing datasets …")
    train_dataset = _build_hf_dataset(train_records, processor, kaggle_input_dir)
    eval_dataset  = _build_hf_dataset(eval_records,  processor, kaggle_input_dir)

    # ── 6. Data collator ───────────────────────────────────────────────────
    collator = WhisperDataCollator(processor=processor)

    # ── 7. WER metric ─────────────────────────────────────────────────────
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids      = pred.predictions
        label_ids     = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str  = tokenizer.batch_decode(pred_ids,   skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids,  skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": round(wer, 4)}

    # ── 8. Training arguments ──────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        gradient_checkpointing=grad_checkpointing,
        fp16=fp16,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        logging_steps=logging_steps,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,          # lower WER = better
        save_total_limit=2,               # keep at most 2 checkpoints on local disk at once
        dataloader_num_workers=num_workers,
        report_to=["none"],               # disable W&B / HF Hub reporting by default
    )

    # ── 9. Trainer ─────────────────────────────────────────────────────────
    callbacks = []
    if drive_backup_dir:
        DriveBackupCallback = _make_drive_backup_callback(TrainerCallback)
        callbacks.append(
            DriveBackupCallback(local_output_dir=output_dir, drive_backup_dir=drive_backup_dir)
        )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=callbacks,
    )

    # ── 10. Train ──────────────────────────────────────────────────────────
    # Check for an existing checkpoint — prefer local disk, but fall back to
    # the Drive-backed copy (useful if local disk was wiped by a disconnect
    # but the Drive sync from a previous session is still there).
    resume_checkpoint = _find_latest_checkpoint(output_dir)
    if not resume_checkpoint and drive_backup_dir:
        drive_checkpoint = _find_latest_checkpoint(drive_backup_dir)
        if drive_checkpoint:
            logger.info(f"No local checkpoint found — restoring synced checkpoint from Drive: {drive_checkpoint}")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            restored_path = os.path.join(output_dir, os.path.basename(drive_checkpoint))
            shutil.copytree(drive_checkpoint, restored_path)
            resume_checkpoint = restored_path

    if resume_checkpoint:
        logger.info(f"Found existing checkpoint — resuming from {resume_checkpoint}")
    else:
        logger.info("No existing checkpoint found — starting fine-tuning from scratch")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # ── 11. Save final model + processor ──────────────────────────────────
    logger.info(f"Saving model to {output_dir} …")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    # If we were syncing checkpoints to Drive during training, replace that
    # checkpoint copy with the final model instead (final model has no
    # optimizer state, so it's smaller — no need to keep the mid-training copy).
    if drive_backup_dir:
        final_drive_path = os.path.join(drive_backup_dir, "final_model")
        logger.info(f"Copying final model to Drive: {final_drive_path}")
        Path(drive_backup_dir).mkdir(parents=True, exist_ok=True)

        for existing in Path(drive_backup_dir).glob("checkpoint-*"):
            shutil.rmtree(existing, ignore_errors=True)
        if os.path.exists(final_drive_path):
            shutil.rmtree(final_drive_path)

        shutil.copytree(output_dir, final_drive_path, ignore=shutil.ignore_patterns("checkpoint-*"))
        logger.info(f"Final model safely backed up to: {final_drive_path}")

    logger.info("=== train.py complete ===")
    logger.info(f"Fine-tuned weights saved to: {output_dir}")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Look inside output_dir for existing checkpoint-<step> folders.
    Returns the path to the one with the highest step number, or None
    if no checkpoints exist yet (i.e. this is a fresh training run).
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    checkpoint_dirs = [
        d for d in output_path.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint-")
    ]

    if not checkpoint_dirs:
        return None

    # Sort by the step number embedded in the folder name (checkpoint-200, checkpoint-400, ...)
    def _step_number(d: Path) -> int:
        try:
            return int(d.name.split("-")[-1])
        except ValueError:
            return -1

    latest = max(checkpoint_dirs, key=_step_number)
    return str(latest)


def _load_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    """Read manifest.json — one JSON object per line (JSONL format)."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            "Run the ingest → clean → align stages first."
        )

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"Manifest at {manifest_path} is empty.")

    return records


def _remap_audio_path(audio_path: str, kaggle_input_dir: str) -> str:
    """
    Remap a manifest audio_path to the correct location on Kaggle.

    Audio was uploaded as chunked zips (audio_chunk_00.zip, audio_chunk_01.zip, ...)
    with chunk_size=2000. Kaggle auto-extracts each zip into a subfolder named
    after the zip, so the final path looks like:
        /kaggle/input/indic-voices-hi/audio_chunk_00/sample_000000.wav

    The sample index embedded in the filename determines which chunk it belongs to.
    """
    filename = Path(audio_path).name          # e.g. sample_000006.wav
    try:
        idx = int(filename.replace("sample_", "").replace(".wav", ""))
    except ValueError:
        raise ValueError(f"Cannot parse sample index from filename: {filename}")

    chunk_num = idx // 2000                   # chunk_size=2000, same as upload
    chunk_folder = f"audio_chunk_{chunk_num:02d}"
    return str(Path(kaggle_input_dir) / chunk_folder / filename)


def _train_eval_split(
    records: List[Dict], eval_split: float
) -> tuple[List[Dict], List[Dict]]:
    """Shuffle and split records into train / eval sets."""
    random.seed(42)
    shuffled = records.copy()
    random.shuffle(shuffled)

    n_eval = max(1, int(len(shuffled) * eval_split))
    return shuffled[n_eval:], shuffled[:n_eval]


def _build_hf_dataset(
    records: List[Dict],
    processor,
    kaggle_input_dir: str = None,
) -> "Dataset":
    """
    Turn a list of manifest records into a HuggingFace Dataset.

    Each record looks like:
        {"audio_path": "...", "transcript": "नमस्ते", "duration": 2.4, ...}

    If kaggle_input_dir is set, audio_path values are remapped via
    _remap_audio_path() to point at the correct Kaggle dataset subfolder.
    Otherwise paths are used as-is (Colab / local runs).

    We load the raw audio and run it through WhisperProcessor here so that
    the Dataset stores pre-computed features rather than raw audio arrays.
    This avoids re-reading audio files during every training step.
    """
    import soundfile as sf
    from datasets import Dataset

    def _process(record):
        # Remap path for Kaggle if needed, otherwise use as-is
        if kaggle_input_dir:
            audio_path = _remap_audio_path(record["audio_path"], kaggle_input_dir)
        else:
            audio_path = record["audio_path"]

        transcript = record["transcript"]

        # Load audio — soundfile returns (samples, sample_rate)
        audio_array, sample_rate = sf.read(audio_path, dtype="float32")

        # WhisperProcessor expects 16 kHz mono
        # (clean.py should have already standardised this, but we resample just in case)
        if sample_rate != 16_000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16_000)

        # Extract log-mel features
        inputs = processor(
            audio_array,
            sampling_rate=16_000,
            return_tensors="pt",
        )
        input_features = inputs.input_features[0]   # shape: (80, 3000)

        # Tokenise the transcript
        labels = processor.tokenizer(transcript).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
        }

    processed = [_process(r) for r in records]
    return Dataset.from_list(processed)


# ─────────────────────────────────────────────
# Data collator
# ─────────────────────────────────────────────

@dataclass
class WhisperDataCollator:
    """
    Pads a batch of (input_features, labels) pairs to the same length.

    Whisper's input features are always the same shape (80 mel bins × 3000 frames),
    so only the label sequences need padding.
    """
    processor: Any

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Input features are fixed-size — just stack them
        input_features = torch.stack(
            [torch.tensor(f["input_features"]) for f in features]
        )

        # Labels need padding — use the tokenizer's pad method
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace padding token id with -100 so the loss ignores pad positions
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        return {
            "input_features": input_features,
            "labels": labels,
        }