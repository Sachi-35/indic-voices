import yaml
import json
from transformers import Trainer, TrainingArguments, AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from pipeline.kaggle_utils_tts import KaggleBackupCallback, find_latest_checkpoint, build_streaming_dataset

CONFIG_PATH = "configs/hindi_tts.yaml"


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    model = ParlerTTSForConditionalGeneration.from_pretrained(cfg["model"]["base_model"])
    prompt_tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["prompt_tokenizer"])
    description_tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["description_tokenizer"])

    with open(cfg["data"]["manifest_path"]) as f:
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
    split = dataset.train_test_split(test_size=0.1, seed=42)

    checkpoint = find_latest_checkpoint(
        cfg["training"]["output_dir"],
        cfg["backup"]["kaggle_dataset"],
        "downloaded_checkpoints",
    )

    args = TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        save_steps=cfg["training"]["save_steps"],
        eval_steps=cfg["training"]["eval_steps"],
        eval_strategy="steps",
        logging_steps=cfg["training"]["logging_steps"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        callbacks=[KaggleBackupCallback(cfg["backup"]["kaggle_dataset"], cfg["training"]["output_dir"])],
    )

    trainer.train(resume_from_checkpoint=checkpoint)


if __name__ == "__main__":
    main()