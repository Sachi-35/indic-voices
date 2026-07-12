import json
import os
import yaml
import soundfile as sf
from datasets import load_dataset

CONFIG_PATH = "configs/hindi_tts.yaml"


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    ds = load_dataset(
        cfg["dataset"],
        cfg["dataset_language_key"],
        split=cfg["dataset_split"],
        streaming=True,
    )

    speaker_filter = cfg["speaker_filter"]

    if speaker_filter is None:
        print("No speaker_filter set — listing first 20 unique speaker IDs:")
        seen = set()
        for row in ds:
            seen.add(row["speaker_id"])
            if len(seen) >= 20:
                break
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

    for row in ds:
        seen_count += 1
        if seen_count % 5000 == 0:
            print(f"scanned {seen_count} rows, matched {len(manifest)} so far")

        if row["speaker_id"] != speaker_filter:
            continue

        audio_array = row["audio"]["array"]
        sampling_rate = row["audio"]["sampling_rate"]
        dur = len(audio_array) / sampling_rate
        if not (min_dur <= dur <= max_dur):
            continue
        if cfg["remove_empty_transcripts"] and not row["text"].strip():
            continue

        # Actually write the audio to disk — this is what was missing.
        out_filename = f"{speaker_filter}_{saved_count:04d}.wav"
        out_path = os.path.join(raw_dir, out_filename)
        sf.write(out_path, audio_array, sampling_rate)
        saved_count += 1

        manifest.append({
            "audio_path": out_path,   # real, resolvable path now
            "text": row["text"],
            "description": description,
            "duration_sec": dur,
            "speaker_id": row["speaker_id"],
        })

    manifest_path = cfg["paths"]["manifest"]
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(manifest)} rows to {manifest_path} (scanned {seen_count} total rows)")
    print(f"Saved {saved_count} audio files to {raw_dir}")


if __name__ == "__main__":
    main()