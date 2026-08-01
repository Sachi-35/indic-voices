# Hindi STT Model Card

## Model
- **Base model:** openai/whisper-small
- **Language:** hi
- **Dataset:** ai4bharat/indicvoices_r (via Kaggle: /kaggle/input/datasets/sachi35/indicvoices-hindi-stage1)
- **Task:** transcribe

## Evaluation
- **Evaluated at:** 2026-07-28T17:00:41.109628+00:00
- **Eval split:** 0.1
- **WER:** 0.1717
- **CER:** 0.0699
- **Num eval samples:** 2562
- **RTF (Real Time Factor):** 0.654 (meets real-time target)
  - Processing time: 6.13s
  - Audio duration: 9.37s

## Packaging / Export
Packaging step not yet run.

## Known limitations
- Base model spec originally referenced `ai4bharat/indicwhisper-hi`; actual training
  used `openai/whisper-small` due to a mismatch. Mentor clarification pending.
- Config `dataset` field previously referenced IndicVoices-R by mistake; corrected —
  STT trains on base IndicVoices (Kaggle-hosted), distinct from TTS's IndicVoices-R.

## Notes
Config-driven pipeline — adding a new language requires only a new YAML config,
no code changes to this evaluation or model card generation logic.
