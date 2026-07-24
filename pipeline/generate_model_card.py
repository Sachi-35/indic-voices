import yaml
import json
import os

def generate_model_card(config_path, metrics_path, output_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    with open(metrics_path) as f:
        metrics = json.load(f)

    wer = metrics.get("wer")
    cer = metrics.get("cer")
    rtf = metrics.get("rtf")
    mos_score = metrics.get("mos_score")
    mos_display = "not yet scored (clips generated, human panel pending)" if mos_score is None else str(mos_score)
    rtf_flag = " ⚠️ above real-time target of 1.0" if (rtf is not None and rtf > 1.0) else ""

    card = f"""# Hindi TTS Model Card

## Model
- **Base model:** {cfg["base_model"]}
- **Language:** {metrics.get("language", "hi")}
- **Sample rate:** {cfg["sample_rate"]} Hz
- **Dataset:** {cfg["dataset"]}
- **Speaker filter:** {cfg.get("speaker_filter", "N/A")}

## Evaluation
- **Evaluated at:** {metrics.get("evaluated_at", "N/A")}
- **Eval split:** {cfg.get("eval_split", "N/A")}
- **Intelligibility WER:** {wer:.4f}
- **Intelligibility CER:** {cer:.4f}
- **Reference STT model:** {metrics.get("intelligibility_reference_model")}
- **Num intelligibility samples:** {metrics.get("num_intelligibility_samples")}
- **MOS clips:** {metrics.get("mos_clips_dir")} ({mos_display})
- **RTF (Real Time Factor):** {rtf:.2f}{rtf_flag}
  - Processing time: {metrics.get("rtf_processing_time_sec"):.2f}s
  - Audio duration: {metrics.get("rtf_audio_duration_sec"):.2f}s

## Known limitations
- RTF of {rtf:.2f} means generation currently runs slower than real-time on a
  Kaggle T4 GPU (fp16 disabled, no inference-specific optimization applied yet).
  This does not yet meet the on-premise real-time deployment target and is an
  open item for further work (batching, fp16 inference pass, or ONNX/other
  runtime acceleration, if export proves feasible for this architecture).
- MOS score requires a human listening panel; clips are generated and ready,
  but scoring has not yet been conducted.

## Notes
Config-driven pipeline — adding a new language requires only a new YAML config,
no code changes to this evaluation or model card generation logic.
"""
    with open(output_path, "w") as f:
        f.write(card)
    print(f"Model card written to {output_path}")

if __name__ == "__main__":
    generate_model_card(
        "configs/hindi_tts.yaml",
        "outputs/hindi_tts_metrics.json",
        "outputs/hindi_tts_model_card.md",
    )
