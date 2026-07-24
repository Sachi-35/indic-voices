# Hindi TTS Model Card

## Model
- **Base model:** ai4bharat/indic-parler-tts
- **Language:** hi
- **Sample rate:** 44100 Hz
- **Dataset:** ai4bharat/indicvoices_r
- **Speaker filter:** S4259529700390036

## Evaluation
- **Evaluated at:** 2026-07-24T13:58:52.749613+00:00
- **Eval split:** 0.1
- **Intelligibility WER:** 0.3458
- **Intelligibility CER:** 0.2191
- **Reference STT model:** vasista22/whisper-hindi-large-v2
- **Num intelligibility samples:** 50
- **MOS clips:** outputs/mos_clips (not yet scored (clips generated, human panel pending))
- **RTF (Real Time Factor):** 12.67 ⚠️ above real-time target of 1.0
  - Processing time: 135.33s
  - Audio duration: 10.68s

## Known limitations
- RTF of 12.67 means generation currently runs slower than real-time on a
  Kaggle T4 GPU (fp16 disabled, no inference-specific optimization applied yet).
  This does not yet meet the on-premise real-time deployment target and is an
  open item for further work (batching, fp16 inference pass, or ONNX/other
  runtime acceleration, if export proves feasible for this architecture).
- MOS score requires a human listening panel; clips are generated and ready,
  but scoring has not yet been conducted.

## Notes
Config-driven pipeline — adding a new language requires only a new YAML config,
no code changes to this evaluation or model card generation logic.
