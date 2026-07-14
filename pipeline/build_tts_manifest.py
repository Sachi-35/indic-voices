import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

import io
import json
import yaml
import soundfile as sf
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, HfApi

CONFIG_PATH = "configs/hindi_tts.yaml"
REPO_LANGUAGE_DIR = "Hindi"  # actual folder name in the repo (not the ISO code used for load_dataset config)


def get_shard_files(dataset_repo, split):
    api = HfApi()
    all_files = api.list_repo_files(dataset_repo, repo_type="dataset")
    shard_files = [f for f in all_files if f.startswith(f"{REPO_LANGUAGE_DIR}/{split}-")]
    shard_files.sort()
    return shard_files


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    speaker_filter = cfg["speaker_filter"]
    shard_files = get_shard_files(cfg["dataset"], cfg["dataset_split"])
    print(f"Found {len(shard_files)} shard(s) for split='{cfg['dataset_split']}'")

    if speaker_filter is None:
        print("No speaker_filter set — scanning first shard only for sample speaker IDs:")
        local_path = hf_hub_download(repo_id=cfg["dataset"], repo_type="dataset", filename=shard_files[0])
        pf = pq.ParquetFile(local_path)
        seen = set()
        for batch in pf.iter_batches(batch_size=1000, columns=["speaker_id"]):
            seen.update(batch.column("speaker_id").to_pylist())
            if len(seen) >= 20:
                break
        os.remove(local_path)
        print(sorted(seen))
        print("\n>>> Pick one, set speaker_filter in the YAML, push, then rerun this script.")
        return

    description = cfg["description_template"]
    min_dur, max_dur = cfg["min_duration"], cfg["max_duration"]
    raw_dir = cfg["paths"]["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)

    manifest = []
    seen_count = 0
    saved_count = 0

    for shard_idx, shard_file in enumerate(shard_files):
        print(f"[shard {shard_idx + 1}/{len(shard_files)}] downloading {shard_file} ...")
        local_path = hf_hub_download(repo_id=cfg["dataset"], repo_type="dataset", filename=shard_file)

        pf = pq.ParquetFile(local_path)
        for batch in pf.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                seen_count += 1
                if seen_count % 5000 == 0:
                    print(f"scanned {seen_count} rows, matched {len(manifest)} so far")

                if row["speaker_id"] != speaker_filter:
                    continue

                audio_field = row["audio"]  # {'bytes': ..., 'path': ...} per HF Audio feature parquet storage
                audio_array, sampling_rate = sf.read(io.BytesIO(audio_field["bytes"]))
                dur = len(audio_array) / sampling_rate
                if not (min_dur <= dur <= max_dur):
                    continue
                if cfg["remove_empty_transcripts"] and not row["text"].strip():
                    continue

                out_filename = f"{speaker_filter}_{saved_count:04d}.wav"
                out_path = os.path.join(raw_dir, out_filename)
                sf.write(out_path, audio_array, sampling_rate)
                saved_count += 1

                manifest.append({
                    "audio_path": out_path,
                    "text": row["text"],
                    "description": description,
                    "duration_sec": dur,
                    "speaker_id": row["speaker_id"],
                })

        # Free disk space before pulling the next ~480MB shard
        os.remove(local_path)
        print(f"[shard {shard_idx + 1}/{len(shard_files)}] done — {saved_count} clips matched so far, shard deleted")

    manifest_path = cfg["paths"]["manifest"]
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(manifest)} rows to {manifest_path} (scanned {seen_count} total rows)")
    print(f"Saved {saved_count} audio files to {raw_dir}")


if __name__ == "__main__":
    main()