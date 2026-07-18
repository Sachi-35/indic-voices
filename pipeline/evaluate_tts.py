import librosa
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["WANDB_DISABLED"] = "true"

import json
import yaml
import jiwer
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

CONFIG_PATH = "configs/hindi_tts.yaml"
PROGRESS_PATH = "outputs/eval_intelligibility_progress.jsonl"


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
    os.makedirs("outputs/mos_clips", exist_ok=True)

    tts_model = ParlerTTSForConditionalGeneration.from_pretrained(cfg["output_dir"])
    prompt_tok = AutoTokenizer.from_pretrained(cfg["prompt_tokenizer"])
    desc_tok = AutoTokenizer.from_pretrained(cfg["description_tokenizer"])

    stt_checkpoint = cfg["evaluation"]["stt_reference_model"]
    stt_processor = WhisperProcessor.from_pretrained(stt_checkpoint)
    stt_model = WhisperForConditionalGeneration.from_pretrained(stt_checkpoint)

    # Known issue with vasista22/whisper-* checkpoints: generation_config.suppress_tokens is empty,
    # which crashes newer transformers' generate() at suppress_tokens[-2]. Setting prev_sot_token_id
    # directly avoids that broken fallback path.
    stt_model.generation_config.prev_sot_token_id = stt_model.generation_config.decoder_start_token_id

    with open(cfg["paths"]["manifest"]) as f:
        records = json.load(f)

    n = cfg["evaluation"]["num_intelligibility_samples"]

    completed = load_progress()
    print(f"[eval] resuming — {len(completed)}/{n} intelligibility samples already done", flush=True)

    with open(PROGRESS_PATH, "a") as progress_f:
        for i, r in enumerate(records[:n]):
            if i in completed:
                print(f"[eval] sample {i+1}/{n} already done, skipping", flush=True)
                continue

            print(f"[eval] generating sample {i+1}/{n}...", flush=True)
            desc_ids = desc_tok(r["description"], return_tensors="pt").input_ids
            prompt_ids = prompt_tok(r["text"], return_tensors="pt").input_ids
            audio_arr = tts_model.generate(
                input_ids=desc_ids,
                prompt_input_ids=prompt_ids,
                max_new_tokens=1000,
                use_cache=True,
            ).cpu().numpy().squeeze()
            print(f"[eval] sample {i+1}/{n} done, shape={audio_arr.shape}", flush=True)

            # Whisper requires 16kHz input; our TTS output is generated at cfg["sample_rate"] (44100 for the DAC codec)
            audio_arr_16k = librosa.resample(audio_arr, orig_sr=cfg["sample_rate"], target_sr=16000)
            inputs = stt_processor(audio_arr_16k, sampling_rate=16000, return_tensors="pt")
            pred_ids = stt_model.generate(inputs["input_features"])
            transcription = stt_processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

            row = {"index": i, "ref": r["text"], "hyp": transcription}
            progress_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            progress_f.flush()
            completed[i] = row

    refs = [completed[i]["ref"] for i in range(n)]
    hyps = [completed[i]["hyp"] for i in range(n)]
    intelligibility_wer = jiwer.wer(refs, hyps)

    mos_n = cfg["evaluation"]["num_mos_samples"]
    for i, r in enumerate(records[:mos_n]):
        out_path = f"outputs/mos_clips/clip_{i}.wav"
        if os.path.exists(out_path):
            print(f"[eval] MOS clip {i+1}/{mos_n} already exists, skipping", flush=True)
            continue
        print(f"[eval] generating MOS clip {i+1}/{mos_n}...", flush=True)
        desc_ids = desc_tok(r["description"], return_tensors="pt").input_ids
        prompt_ids = prompt_tok(r["text"], return_tensors="pt").input_ids
        audio_arr = tts_model.generate(
            input_ids=desc_ids,
            prompt_input_ids=prompt_ids,
            max_new_tokens=1000,
            use_cache=True,
        ).cpu().numpy().squeeze()
        sf.write(out_path, audio_arr, cfg["sample_rate"])

    results = {
        "intelligibility_wer": intelligibility_wer,
        "intelligibility_reference_model": stt_checkpoint,
        "num_intelligibility_samples": n,
        "mos_clips_dir": "outputs/mos_clips",
        "mos_score": None,
    }
    with open(cfg["evaluation"]["metrics_output"], "w") as f:
        json.dump(results, f, indent=2)

    print(results)


if __name__ == "__main__":
    main()