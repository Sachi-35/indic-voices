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

## Packaging / Export
ONNX export was attempted and failed:

> The checkpoint you are trying to load has model type `parler_tts` but Transformers does not recognize this architecture. This could be because of an issue with the checkpoint, or because your version of Transformers is out of date.

This occurs because Parler-TTS is a custom architecture not registered in optimum's ONNX exporter config registry (not native to `transformers`), so no export config exists for it out of the box. Combined with the decoder's autoregressive generation loop, ONNX export is not expected to work without writing a custom optimum export config for this architecture. Fallback: model saved in standard HF format at `outputs/packaged_hf/` — usable directly with `transformers`/`parler-tts`, just not ONNX Runtime.

## Notes
Config-driven pipeline — adding a new language requires only a new YAML config,
no code changes to this evaluation or model card generation logic.
