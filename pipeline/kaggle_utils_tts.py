import os
import glob
import shutil
import subprocess
from transformers import TrainerCallback

STAGING_DIR = "kaggle_backup_staging"


class KaggleBackupCallback(TrainerCallback):
    def __init__(self, kaggle_dataset: str, local_checkpoint_dir: str):
        self.kaggle_dataset = kaggle_dataset
        self.local_checkpoint_dir = local_checkpoint_dir
        os.makedirs(STAGING_DIR, exist_ok=True)

    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        ckpt_dir = os.path.join(self.local_checkpoint_dir, f"checkpoint-{step}")
        if not os.path.isdir(ckpt_dir):
            print(f"[KaggleBackupCallback] {ckpt_dir} not found, skipping backup")
            return

        # Zip this checkpoint ourselves, preserving folder structure inside the zip
        archive_base = os.path.join(STAGING_DIR, f"checkpoint-{step}")
        shutil.make_archive(archive_base, "zip", root_dir=self.local_checkpoint_dir, base_dir=f"checkpoint-{step}")

        try:
            subprocess.run(
                ["kaggle", "datasets", "version",
                 "-p", STAGING_DIR,
                 "-m", f"checkpoint at step {step}",
                 "-r", "skip"],   # skip = upload files as-is, don't let Kaggle re-zip
                check=True,
            )
            print(f"[KaggleBackupCallback] backed up step {step}")
        except subprocess.CalledProcessError as e:
            print(f"[KaggleBackupCallback] backup failed: {e}")


def find_latest_checkpoint(local_dir, kaggle_dataset, kaggle_download_dir):
    local_checkpoints = sorted(
        glob.glob(os.path.join(local_dir, "checkpoint-*")),
        key=lambda p: int(p.split("-")[-1]),
    )
    if local_checkpoints:
        print(f"[find_latest_checkpoint] found local checkpoint: {local_checkpoints[-1]}")
        return local_checkpoints[-1]

    print(f"[find_latest_checkpoint] no local checkpoint, checking {kaggle_dataset}")
    os.makedirs(kaggle_download_dir, exist_ok=True)
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", kaggle_dataset, "-p", kaggle_download_dir],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[find_latest_checkpoint] no backup found or download failed: {result.stderr}")
        return None

    zips = sorted(
        glob.glob(os.path.join(kaggle_download_dir, "checkpoint-*.zip")),
        key=lambda p: int(os.path.basename(p).replace("checkpoint-", "").replace(".zip", "")),
    )
    if not zips:
        print("[find_latest_checkpoint] no checkpoint zips in backup dataset")
        return None

    latest_zip = zips[-1]
    step = os.path.basename(latest_zip).replace("checkpoint-", "").replace(".zip", "")
    restore_path = os.path.join(local_dir, f"checkpoint-{step}")
    os.makedirs(local_dir, exist_ok=True)
    shutil.unpack_archive(latest_zip, local_dir)
    print(f"[find_latest_checkpoint] restored {restore_path} from Kaggle backup")
    return restore_path


def build_streaming_dataset(records, process_fn):
    from datasets import Dataset
    def gen():
        for r in records:
            yield process_fn(r)
    return Dataset.from_generator(gen)