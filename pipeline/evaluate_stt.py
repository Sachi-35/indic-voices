
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["WANDB_DISABLED"] = "true"

import sys
import json
import time
import yaml
import jiwer
import subprocess
import soundfile as sf
import librosa
from datetime import datetime, timezone
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizerFast,
)

sys.path.insert(0, "pipeline")
from train import _build_audio_path_index, _remap_audio_path, _train_eval_split, _load_manifest

CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "configs/hindi_stt.yaml"
_lang = CONFIG_PATH.split("/")[-1].replace("_stt.yaml", "")
PROGRESS_PATH = f"outputs/eval_stt_progress_{_lang}.jsonl"


def backup_progress(paths=None, label="progress"):
    github_token = os.environ.get("GITHUB_TOKEN", "")
    if not github_token:
        print("[eval] GITHUB_TOKEN not set, skipping backup", flush=True)
        return
    if paths is None:
        paths = [PROGRESS_PATH]
    try:
        subprocess.run(
            ["git", "remote", "set-url", "origin",
             f"https://Sachi-35:{github_token}@github.com/Sachi-35/indic-voices.git"],
            check=True,
        )
        for p in paths:
            if os.path.exists(p):
                subprocess.run(["git", "add", "-f", p], check=True)
        result = subprocess.run(["git", "commit", "-m", f"{label} checkpoint"], capture_output=True, text=True)
        if result.returncode != 0 and "nothing to commit" not in result.stdout:
            print(f"[eval] git commit issue: {result.stdout}", flush=True)
        subprocess.run(["git", "push"], check=True)
        print(f"[eval] {label} backed up to git", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"[eval] {label} backup failed (continuing anyway): {e}", flush=True)


def load_progress():
    completed = {}
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                completed[row["index"]] = row
    return completed


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    os.makedirs("outputs", exist_ok=True)

    output_dir = cfg["output_dir"]
    kaggle_input_dir = cfg.get("kaggle_input_dir")
    eval_split = float(cfg.get("eval_split", 0.1))
    max_eval_samples = cfg.get("evaluation", {}).get("num_eval_samples")  # None = full split

    print(f"[eval] loading model from {output_dir}", flush=True)
    model = WhisperForConditionalGeneration.from_pretrained(output_dir)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(output_dir)
    tokenizer = WhisperTokenizerFast.from_pretrained(output_dir)

    if kaggle_input_dir:
        audio_index = _build_audio_path_index(kaggle_input_dir)
        print(f"[eval] indexed {len(audio_index)} audio files", flush=True)
    else:
        audio_index = None
        print("[eval] no kaggle_input_dir set — using manifest audio_path values directly", flush=True)

    records = _load_manifest(cfg["paths"]["manifest"])
    _, eval_records = _train_eval_split(records, eval_split)
    if max_eval_samples:
        eval_records = eval_records[:max_eval_samples]
    n = len(eval_records)
    print(f"[eval] evaluating on {n} held-out records", flush=True)

    completed = load_progress()
    print(f"[eval] resuming - {len(completed)}/{n} samples already done", flush=True)

    with open(PROGRESS_PATH, "a") as progress_f:
        for i, r in enumerate(eval_records):
            if i in completed:
                if (i + 1) % 200 == 0:
                    print(f"[eval] sample {i+1}/{n} already done, skipping", flush=True)
                continue

            audio_path = _remap_audio_path(r["audio_path"], audio_index) if audio_index is not None else r["audio_path"]
            audio_arr, sr = sf.read(audio_path, dtype="float32")
            if sr != 16000:
                audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=16000)

            inputs = feature_extractor(audio_arr, sampling_rate=16000, return_tensors="pt")
            pred_ids = model.generate(inputs["input_features"], max_new_tokens=225, language="hi", task="transcribe")
            hyp = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]

            row = {"index": i, "ref": r["transcript"], "hyp": hyp}
            progress_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            progress_f.flush()
            completed[i] = row

            if (i + 1) % 200 == 0:
                print(f"[eval] sample {i+1}/{n} done", flush=True)
                backup_progress()

    refs = [completed[i]["ref"] for i in range(n)]
    hyps = [completed[i]["hyp"] for i in range(n)]
    wer = jiwer.wer(refs, hyps)
    cer = jiwer.cer(refs, hyps)

    # RTF on one sample
    rtf_record = eval_records[0]
    audio_path = _remap_audio_path(rtf_record["audio_path"], audio_index) if audio_index is not None else rtf_record["audio_path"]
    audio_arr, sr = sf.read(audio_path, dtype="float32")
    if sr != 16000:
        audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=16000)
    inputs = feature_extractor(audio_arr, sampling_rate=16000, return_tensors="pt")
    t0 = time.time()
    _ = model.generate(inputs["input_features"], max_new_tokens=225, language="hi", task="transcribe")
    processing_time = time.time() - t0
    audio_duration = len(audio_arr) / 16000
    rtf = processing_time / audio_duration

    results = {
        "task": "stt",
        "language": "hi",
        "model": cfg["base_model"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "wer": wer,
        "cer": cer,
        "num_eval_samples": n,
        "rtf": rtf,
        "rtf_processing_time_sec": processing_time,
        "rtf_audio_duration_sec": audio_duration,
    }

    metrics_path = cfg.get("evaluation", {}).get("metrics_output", "outputs/hindi_stt_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    backup_progress(paths=[metrics_path], label="stt_final_metrics")

    print(results)


if __name__ == "__main__":
    main()
