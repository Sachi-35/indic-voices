import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import yaml
import json
from transformers import Trainer, TrainingArguments, AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from pipeline.kaggle_utils_tts import KaggleBackupCallback, find_latest_checkpoint, build_streaming_dataset


CONFIG_PATH = "configs/hindi_tts.yaml"


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
        return {
            "prompt_input_ids": prompt_ids,
            "description_input_ids": description_ids,
            "audio_path": r["audio_path"],
        }

    dataset = build_streaming_dataset(records, process)
    split = dataset.train_test_split(test_size=cfg["eval_split"], seed=42)

    checkpoint = find_latest_checkpoint(
        cfg["output_dir"],
        cfg["backup"]["kaggle_dataset"],
        "downloaded_checkpoints",
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
        callbacks=[KaggleBackupCallback(cfg["backup"]["kaggle_dataset"], cfg["output_dir"])],
    )

    trainer.train(resume_from_checkpoint=checkpoint)


if __name__ == "__main__":
    main()