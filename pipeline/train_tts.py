import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["WANDB_DISABLED"] = "true"

import yaml
import json
import torch
import soundfile as sf
from dataclasses import dataclass
from typing import List, Dict
from transformers import Trainer, TrainingArguments, AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from pipeline.kaggle_utils_tts import KaggleBackupCallback, find_latest_checkpoint, build_streaming_dataset

CONFIG_PATH = "configs/hindi_tts.yaml"


@dataclass
class TTSDataCollator:
    prompt_tokenizer: AutoTokenizer
    description_tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        prompt_ids = [f["prompt_input_ids"] for f in features]
        description_ids = [f["description_input_ids"] for f in features]

        prompt_batch = self.prompt_tokenizer.pad(
            {"input_ids": prompt_ids}, return_tensors="pt", padding=True
        )
        description_batch = self.description_tokenizer.pad(
            {"input_ids": description_ids}, return_tensors="pt", padding=True
        )

        batch = {
            "prompt_input_ids": prompt_batch["input_ids"],
            "prompt_attention_mask": prompt_batch["attention_mask"],
            "input_ids": description_batch["input_ids"],
            "attention_mask": description_batch["attention_mask"],
        }

        # TODO — Problem 2: pad/stack "labels" (audio codec tokens) here too.
        # Needs the audio-encoding step verified against your installed
        # parler_tts version's exact API before this is safe to fill in.
        labels = [f["labels"] for f in features]
        max_len = max(l.shape[-1] for l in labels)
        padded = torch.stack([
            torch.nn.functional.pad(l, (0, max_len - l.shape[-1]), value=-100)
            for l in labels
        ])
        batch["labels"] = padded

        return batch


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    model = ParlerTTSForConditionalGeneration.from_pretrained(cfg["base_model"])
    prompt_tokenizer = AutoTokenizer.from_pretrained(cfg["prompt_tokenizer"])
    description_tokenizer = AutoTokenizer.from_pretrained(cfg["description_tokenizer"])

    with open(cfg["paths"]["manifest"]) as f:
        records = json.load(f)

    def process(r):
        prompt_ids = prompt_tokenizer(r["text"], return_tensors="pt").input_ids[0]
        description_ids = description_tokenizer(r["description"], return_tensors="pt").input_ids[0]

        # TODO — Problem 2: load r["audio_path"], resample to cfg["sample_rate"],
        # encode via model.audio_encoder, apply build_delay_pattern_mask.
        # Placeholder below WILL produce a working batch shape but meaningless
        # labels — do not start a real training run until this is filled in.
        audio, sr = sf.read(r["audio_path"])
        labels = torch.zeros(model.decoder.config.num_codebooks, 50, dtype=torch.long)  # PLACEHOLDER

        return {
            "prompt_input_ids": prompt_ids,
            "description_input_ids": description_ids,
            "labels": labels,
        }

    dataset = build_streaming_dataset(records, process)
    split = dataset.train_test_split(test_size=cfg["eval_split"], seed=42)

    checkpoint = find_latest_checkpoint(
        cfg["output_dir"], cfg["backup"]["kaggle_dataset"], "downloaded_checkpoints",
    )

    t = cfg["training"]
    args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        num_train_epochs=t["num_train_epochs"],
        save_steps=t["save_steps"],
        eval_steps=t["eval_steps"],
        eval_strategy="steps",
        logging_steps=t["logging_steps"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=TTSDataCollator(prompt_tokenizer, description_tokenizer),
        callbacks=[KaggleBackupCallback(cfg["backup"]["kaggle_dataset"], cfg["output_dir"])],
    )

    trainer.train(resume_from_checkpoint=checkpoint)


if __name__ == "__main__":
    main()