import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # batch size is 1 — DataParallel across 2 GPUs adds only gradient-sync overhead, no real parallelism


import yaml
import json
import torch
import librosa
from dataclasses import dataclass
from typing import List, Dict
from transformers import Trainer, TrainingArguments, AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration, build_delay_pattern_mask
from pipeline.kaggle_utils_tts import KaggleBackupCallback, find_latest_checkpoint, build_streaming_dataset

CONFIG_PATH = "configs/hindi_tts.yaml"


@dataclass
class TTSDataCollator:
    prompt_tokenizer: AutoTokenizer
    description_tokenizer: AutoTokenizer
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        prompt_batch = self.prompt_tokenizer.pad(
            {"input_ids": [f["prompt_input_ids"] for f in features]},
            return_tensors="pt", padding=True,
        )
        description_batch = self.description_tokenizer.pad(
            {"input_ids": [f["description_input_ids"] for f in features]},
            return_tensors="pt", padding=True,
        )

        labels = [f["labels"] for f in features]
        labels = [l if torch.is_tensor(l) else torch.as_tensor(l, dtype=torch.long) for l in labels]
        max_len = max(l.shape[0] for l in labels)  # seq_len is now dim 0
        padded_labels = torch.stack([
            torch.nn.functional.pad(l, (0, 0, 0, max_len - l.shape[0]), value=self.label_pad_token_id)
            for l in labels
        ])

        return {
            "prompt_input_ids": prompt_batch["input_ids"],
            "prompt_attention_mask": prompt_batch["attention_mask"],
            "input_ids": description_batch["input_ids"],
            "attention_mask": description_batch["attention_mask"],
            "labels": padded_labels,
        }


def precompute_audio_labels(records, model, sample_rate, num_codebooks, bos_token_id, pad_token_id, eos_token_id):
    model.audio_encoder.eval()
    labels_by_index = []
    with torch.no_grad():
        for i, r in enumerate(records):
            audio, _ = librosa.load(r["audio_path"], sr=sample_rate, mono=True)
            input_values = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            encoder_out = model.audio_encoder.encode(input_values)
            codes = encoder_out.audio_codes.squeeze(0)

            seq_len = codes.shape[-1]
            _, delay_pattern_mask = build_delay_pattern_mask(
                codes,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                max_length=seq_len + num_codebooks - 1,
                num_codebooks=num_codebooks,
            )
            labels = torch.where(delay_pattern_mask == -1, eos_token_id, delay_pattern_mask)
            labels = labels.transpose(0, 1)  # (num_codebooks, seq_len) -> (seq_len, num_codebooks), required by model
            labels_by_index.append(labels)

            if i == 0:
                print(f"[precompute] example 0 — codes: {codes.shape}, labels: {labels.shape}")
            if (i + 1) % 20 == 0:
                print(f"[precompute] encoded {i + 1}/{len(records)}")

    return labels_by_index


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    model = ParlerTTSForConditionalGeneration.from_pretrained(cfg["base_model"])

    # Freeze text encoder + audio encoder — only fine-tune the decoder.
    # This is standard Parler-TTS practice and cuts Adam optimizer-state memory substantially.
    for p in model.text_encoder.parameters():
        p.requires_grad = False
    for p in model.audio_encoder.parameters():
        p.requires_grad = False
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[main] trainable params: {n_trainable:,} / {n_total:,}")

    prompt_tokenizer = AutoTokenizer.from_pretrained(cfg["prompt_tokenizer"])
    description_tokenizer = AutoTokenizer.from_pretrained(cfg["description_tokenizer"])

    with open(cfg["paths"]["manifest"]) as f:
        records = json.load(f)

    dcfg = model.decoder.config
    print(f"[main] num_codebooks={dcfg.num_codebooks}, bos={dcfg.bos_token_id}, "
          f"eos={dcfg.eos_token_id}, pad={dcfg.pad_token_id}")

    audio_labels = precompute_audio_labels(
        records, model, cfg["sample_rate"],
        dcfg.num_codebooks, dcfg.bos_token_id, dcfg.pad_token_id, dcfg.eos_token_id,
    )

    def process(indexed_r):
        idx, r = indexed_r
        prompt_ids = prompt_tokenizer(r["text"], return_tensors="pt").input_ids[0]
        description_ids = description_tokenizer(r["description"], return_tensors="pt").input_ids[0]
        return {
            "prompt_input_ids": prompt_ids,
            "description_input_ids": description_ids,
            "labels": audio_labels[idx],
        }

    dataset = build_streaming_dataset(list(enumerate(records)), process)
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
        remove_unused_columns=False,
        fp16=True,
        gradient_checkpointing=True,
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