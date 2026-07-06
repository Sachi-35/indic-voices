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


def _make_kaggle_backup_callback(TrainerCallback):
    class KaggleBackupCallback(TrainerCallback):
        def __init__(self, local_output_dir: str, kaggle_backup_dataset_id: str):
            self.local_output_dir = local_output_dir
            self.kaggle_backup_dataset_id = kaggle_backup_dataset_id

        def on_save(self, args, state, control, **kwargs):
            latest = _find_latest_checkpoint(self.local_output_dir)
            if not latest:
                return control

            import json
            import subprocess

            stage_dir = Path(self.local_output_dir) / "_kaggle_backup_stage"
            if stage_dir.exists():
                shutil.rmtree(stage_dir)
            stage_dir.mkdir(parents=True)

            checkpoint_name = os.path.basename(latest)

            # Zip the checkpoint folder ourselves, preserving its name as the
            # top-level entry inside the archive. This avoids Kaggle's
            # --dir-mode zip silently flattening the folder contents.
            archive_base = str(stage_dir / checkpoint_name)
            shutil.make_archive(
                archive_base, "zip",
                root_dir=os.path.dirname(latest),
                base_dir=checkpoint_name,
            )
            logger.info(f"Zipped checkpoint for Kaggle backup: {latest} -> {archive_base}.zip")

            metadata = {
                "title": self.kaggle_backup_dataset_id.split("/")[-1],
                "id": self.kaggle_backup_dataset_id,
                "licenses": [{"name": "CC0-1.0"}],
            }
            (stage_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))

            step = state.global_step
            result = subprocess.run(
                [
                    "kaggle", "datasets", "version",
                    "-p", str(stage_dir),
                    "-m", f"checkpoint at step {step}",
                    "--dir-mode", "skip",
                ],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                logger.warning(f"Kaggle backup FAILED at step {step}: {result.stderr}")
            else:
                logger.info(f"Kaggle backup succeeded at step {step}")

            shutil.rmtree(stage_dir, ignore_errors=True)
            return control

    return KaggleBackupCallback


def run(config: dict) -> None:
    logger.info("=== train.py starting ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning(
            "No GPU detected — training on CPU. This will be very slow. "
            "Consider Google Colab (free GPU) or Kaggle instead."
        )
    else:
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")

    base_model  = config["base_model"]
    task        = config.get("task", "transcribe")
    language    = config.get("language", "hi")
    eval_split  = float(config.get("eval_split", 0.1))
    output_dir  = config["output_dir"]

    manifest_path = config["paths"]["manifest"]

    kaggle_input_dir = config.get("kaggle_input_dir", None)
    if kaggle_input_dir:
        logger.info(f"Kaggle mode: audio paths will be remapped to {kaggle_input_dir}")

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

    kaggle_backup_dataset_id = train_cfg.get("kaggle_backup_dataset_id", None)

    logger.info(f"Base model  : {base_model}")
    logger.info(f"Manifest    : {manifest_path}")
    logger.info(f"Output dir  : {output_dir}")
    logger.info(f"Eval split  : {eval_split}")
    logger.info(f"Epochs      : {num_epochs}")

    records = _load_manifest(manifest_path)
    logger.info(f"Loaded {len(records)} records from manifest")

    train_records, eval_records = _train_eval_split(records, eval_split)
    logger.info(f"Train: {len(train_records)}  |  Eval: {len(eval_records)}")

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
    tokenizer = WhisperTokenizer.from_pretrained(base_model, language=language, task=task)
    processor = WhisperProcessor.from_pretrained(base_model, language=language, task=task)

    logger.info(f"Loading model from {base_model} …")
    model = WhisperForConditionalGeneration.from_pretrained(base_model)
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    if grad_checkpointing:
        model.config.use_cache = False

    from datasets import Dataset

    logger.info("Preparing datasets …")
    train_dataset = _build_hf_dataset(train_records, processor, kaggle_input_dir)
    eval_dataset  = _build_hf_dataset(eval_records,  processor, kaggle_input_dir)

    collator = WhisperDataCollator(processor=processor)

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids      = pred.predictions
        label_ids     = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str  = tokenizer.batch_decode(pred_ids,   skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids,  skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": round(wer, 4)}

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
        greater_is_better=False,
        save_total_limit=2,
        dataloader_num_workers=num_workers,
        report_to=["none"],
    )

    callbacks = []
    if kaggle_backup_dataset_id:
        KaggleBackupCallback = _make_kaggle_backup_callback(TrainerCallback)
        callbacks.append(
            KaggleBackupCallback(local_output_dir=output_dir, kaggle_backup_dataset_id=kaggle_backup_dataset_id)
        )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
        callbacks=callbacks,
    )

    resume_checkpoint = _find_latest_checkpoint(output_dir)
    if not resume_checkpoint and kaggle_backup_dataset_id:
        import subprocess
        logger.info(f"No local checkpoint found — attempting to restore from Kaggle backup dataset: {kaggle_backup_dataset_id}")
        restore_dir = Path(output_dir) / "_kaggle_restore_stage"
        restore_dir.mkdir(parents=True, exist_ok=True)
        dl_result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", kaggle_backup_dataset_id, "-p", str(restore_dir), "--unzip"],
            capture_output=True, text=True,
        )
        if dl_result.returncode != 0:
            logger.warning(f"Kaggle backup restore failed (may not exist yet): {dl_result.stderr}")
        else:
            restored_checkpoint = _find_latest_checkpoint(str(restore_dir))
            if restored_checkpoint:
                dest = os.path.join(output_dir, os.path.basename(restored_checkpoint))
                shutil.move(restored_checkpoint, dest)
                resume_checkpoint = dest
                logger.info(f"Restored checkpoint from Kaggle backup: {dest}")
        shutil.rmtree(restore_dir, ignore_errors=True)

    if resume_checkpoint:
        logger.info(f"Found existing checkpoint — resuming from {resume_checkpoint}")
    else:
        logger.info("No existing checkpoint found — starting fine-tuning from scratch")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    logger.info(f"Saving model to {output_dir} …")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    if kaggle_backup_dataset_id:
        import json, subprocess
        logger.info(f"Pushing final model to Kaggle backup dataset: {kaggle_backup_dataset_id}")
        stage_dir = Path(output_dir) / "_kaggle_final_stage"
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True)
        shutil.copytree(output_dir, stage_dir / "final_model", ignore=shutil.ignore_patterns("checkpoint-*", "_kaggle_*"))
        metadata = {
            "title": kaggle_backup_dataset_id.split("/")[-1],
            "id": kaggle_backup_dataset_id,
            "licenses": [{"name": "CC0-1.0"}],
        }
        (stage_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))
        result = subprocess.run(
            ["kaggle", "datasets", "version", "-p", str(stage_dir), "-m", "final model", "--dir-mode", "zip"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.warning(f"Final model Kaggle backup FAILED: {result.stderr}")
        else:
            logger.info("Final model safely backed up to Kaggle dataset")
        shutil.rmtree(stage_dir, ignore_errors=True)

    logger.info("=== train.py complete ===")
    logger.info(f"Fine-tuned weights saved to: {output_dir}")


def _find_latest_checkpoint(output_dir: str) -> Optional[str]:
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    checkpoint_dirs = [
        d for d in output_path.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint-")
    ]
    if not checkpoint_dirs:
        return None

    def _step_number(d: Path) -> int:
        try:
            return int(d.name.split("-")[-1])
        except ValueError:
            return -1

    return str(max(checkpoint_dirs, key=_step_number))


def _load_manifest(manifest_path: str) -> List[Dict[str, Any]]:
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


def _build_audio_path_index(kaggle_input_dir: str) -> Dict[str, str]:
    index = {}
    base = Path(kaggle_input_dir)
    folders = sorted(base.glob("data_raw_hi_*/hi"))
    for folder in folders:
        for wav_file in folder.glob("*.wav"):
            index[wav_file.name] = str(wav_file)
    logger.info(f"Built audio path index: {len(index)} files across {len(folders)} folders")
    return index


def _remap_audio_path(audio_path: str, audio_index: Dict[str, str]) -> str:
    filename = Path(audio_path).name
    if filename not in audio_index:
        raise FileNotFoundError(f"{filename} not found in any data_raw_hi_* folder")
    return audio_index[filename]


def _train_eval_split(records: List[Dict], eval_split: float) -> tuple[List[Dict], List[Dict]]:
    random.seed(42)
    shuffled = records.copy()
    random.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * eval_split))
    return shuffled[n_eval:], shuffled[:n_eval]


def _build_hf_dataset(records: List[Dict], processor, kaggle_input_dir: str = None) -> "Dataset":
    import soundfile as sf
    from datasets import Dataset

    audio_index = _build_audio_path_index(kaggle_input_dir) if kaggle_input_dir else None

    def _process(record):
        if audio_index is not None:
            audio_path = _remap_audio_path(record["audio_path"], audio_index)
        else:
            audio_path = record["audio_path"]

        transcript = record["transcript"]
        audio_array, sample_rate = sf.read(audio_path, dtype="float32")

        if sample_rate != 16_000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16_000)

        inputs = processor(audio_array, sampling_rate=16_000, return_tensors="pt")
        input_features = inputs.input_features[0]
        labels = processor.tokenizer(transcript).input_ids

        return {"input_features": input_features, "labels": labels}

    def _generator():
        for r in records:
            yield _process(r)

    return Dataset.from_generator(_generator)


@dataclass
class WhisperDataCollator:
    processor: Any

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = torch.stack([torch.tensor(f["input_features"]) for f in features])
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        return {"input_features": input_features, "labels": labels}


def restore_latest_checkpoint_from_kaggle(output_dir: str, kaggle_backup_dataset_id: str):
    """Downloads and restores the latest checkpoint zip from the Kaggle backup dataset."""
    import zipfile
    import glob
    import subprocess

    inspect_dir = "/kaggle/working/_checkpoint_restore"
    os.makedirs(inspect_dir, exist_ok=True)

    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", kaggle_backup_dataset_id,
        "-p", inspect_dir,
        "--unzip",
    ], check=True)

    zips = glob.glob(os.path.join(inspect_dir, "checkpoint-*.zip"))
    if not zips:
        logger.info("No checkpoint zip found in backup dataset — nothing to restore.")
        return None

    latest_zip = max(zips, key=lambda p: int(os.path.basename(p).split("-")[1].split(".")[0]))
    with zipfile.ZipFile(latest_zip) as zf:
        zf.extractall(output_dir)

    step_name = os.path.basename(latest_zip).replace(".zip", "")
    logger.info(f"Restored {step_name} into {output_dir}")
    return os.path.join(output_dir, step_name)
