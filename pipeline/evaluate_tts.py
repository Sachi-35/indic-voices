import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import json
import yaml
import jiwer
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

CONFIG_PATH = "configs/hindi_tts.yaml"


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    tts_model = ParlerTTSForConditionalGeneration.from_pretrained(cfg["output_dir"])
    prompt_tok = AutoTokenizer.from_pretrained(cfg["prompt_tokenizer"])
    desc_tok = AutoTokenizer.from_pretrained(cfg["description_tokenizer"])

    stt_checkpoint = cfg["evaluation"]["stt_reference_model"]
    stt_processor = WhisperProcessor.from_pretrained(stt_checkpoint)
    stt_model = WhisperForConditionalGeneration.from_pretrained(stt_checkpoint)

    with open(cfg["paths"]["manifest"]) as f:
        records = json.load(f)

    n = cfg["evaluation"]["num_intelligibility_samples"]
    refs, hyps = [], []

    for r in records[:n]:
        desc_ids = desc_tok(r["description"], return_tensors="pt").input_ids
        prompt_ids = prompt_tok(r["text"], return_tensors="pt").input_ids
        audio_arr = tts_model.generate(input_ids=desc_ids, prompt_input_ids=prompt_ids).cpu().numpy().squeeze()

        inputs = stt_processor(audio_arr, sampling_rate=cfg["sample_rate"], return_tensors="pt")
        pred_ids = stt_model.generate(inputs["input_features"])
        transcription = stt_processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

        refs.append(r["text"])
        hyps.append(transcription)

    intelligibility_wer = jiwer.wer(refs, hyps)

    mos_n = cfg["evaluation"]["num_mos_samples"]
    for i, r in enumerate(records[:mos_n]):
        desc_ids = desc_tok(r["description"], return_tensors="pt").input_ids
        prompt_ids = prompt_tok(r["text"], return_tensors="pt").input_ids
        audio_arr = tts_model.generate(input_ids=desc_ids, prompt_input_ids=prompt_ids).cpu().numpy().squeeze()
        sf.write(f"outputs/mos_clips/clip_{i}.wav", audio_arr, cfg["sample_rate"])

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