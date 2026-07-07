import os
import glob
import subprocess
from transformers import TrainerCallback


class KaggleBackupCallback(TrainerCallback):
    def __init__(self, kaggle_dataset: str, local_checkpoint_dir: str):
        self.kaggle_dataset = kaggle_dataset
        self.local_checkpoint_dir = local_checkpoint_dir

    def on_save(self, args, state, control, **kwargs):
        try:
            subprocess.run(
                ["kaggle", "datasets", "version",
                 "-p", self.local_checkpoint_dir,
                 "-m", f"checkpoint at step {state.global_step}",
                 "-r", "zip"],
                check=True,
            )
            print(f"[KaggleBackupCallback] backed up step {state.global_step}")
        except subprocess.CalledProcessError as e:
            print(f"[KaggleBackupCallback] backup failed: {e}")


def find_latest_checkpoint(local_dir, kaggle_dataset, kaggle_download_dir):
    local_checkpoints = sorted(
        glob.glob(os.path.join(local_dir, "checkpoint-*")),
        key=lambda p: int(p.split("-")[-1]),
    )
    if local_checkpoints:
        return local_checkpoints[-1]
    os.makedirs(kaggle_download_dir, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", kaggle_dataset,
         "-p", kaggle_download_dir, "--unzip"],
        check=True,
    )
    downloaded = sorted(
        glob.glob(os.path.join(kaggle_download_dir, "checkpoint-*")),
        key=lambda p: int(p.split("-")[-1]),
    )
    return downloaded[-1] if downloaded else None


def build_streaming_dataset(records, process_fn):
    from datasets import Dataset
    def gen():
        for r in records:
            yield process_fn(r)
    return Dataset.from_generator(gen)