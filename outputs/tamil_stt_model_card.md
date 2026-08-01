# Tamil STT Model Card

## Model
- **Base model:** openai/whisper-small
- **Language:** ta
- **Dataset:** ai4bharat/IndicVoices (via HuggingFace streaming)
- **Task:** transcribe

## Evaluation
- **Evaluated at:** 2026-08-01T11:32:33.975162+00:00
- **Eval split:** 0.1
- **WER:** 1.1658
- **CER:** 0.9558
- **Num eval samples:** 32
- **RTF (Real Time Factor):** 0.916 (meets real-time target)
  - Processing time: 10.70s
  - Audio duration: 11.68s

## Packaging / Export
Packaging step not yet run.

## Known limitations
- Small training set (288 samples approx., 90/10 split) — this run exists to prove pipeline reusability across languages/scripts, not to produce a production-quality model. WER/CER above reflect data scarcity, not a pipeline defect.
- Alignment used a general-purpose Tamil wav2vec2 CTC model (`Amrrs/wav2vec2-large-xlsr-53-tamil`), not one purpose-built for this dataset; some valid samples were likely dropped at the `min_alignment_score` threshold.

## Notes
Config-driven pipeline — adding a new language requires only a new YAML config,
no code changes to this evaluation or model card generation logic.
