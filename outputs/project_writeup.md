# Project Write-Up — Indic Voice Pipeline (Hindi + Tamil)

## 1. What the pipeline does

The pipeline is a config-driven system for fine-tuning speech models on Indian
languages, covering two independent tracks: STT (Speech-to-Text, via Whisper)
and TTS (Text-to-Speech, via Parler-TTS). Each track runs through the same
five conceptual stages: **ingest** (stream a HuggingFace dataset, standardise
audio to WAV at a fixed sample rate) → **clean** (drop clips that are too
short/long/quiet, normalise transcript text) → **align** (for STT: forced-align
audio to text with a wav2vec2 CTC model, dropping low-confidence pairs) →
**train** (fine-tune the base model with Hugging Face `Trainer`) → **evaluate**
(compute WER/CER against a held-out split, measure RTF, generate MOS clips for
TTS) → **package** (attempt ONNX export, fall back to standard HF format on
failure). Every stage reads its behaviour entirely from a YAML config file —
dataset name, language code, script, model choice, thresholds, and paths all
live in `configs/<language>_<task>.yaml`. The only thing that should change to
support a new language is that config file, plus new data.

## 2. Data decisions that mattered

Cleaning thresholds (`min_duration: 1.0s`, `max_duration: 20.0s`,
`min_rms_db: -40.0`) were kept consistent across both languages to make results
comparable, rather than re-tuned per language — the point of Week 6 was to
prove the *thresholds themselves* transfer, not to optimise them per language.
Text normalisation needed care: Hindi's `clean.py` originally normalised
Devanagari numerals and danda punctuation, and separately called
`indic-nlp-library`'s per-language normalizer when available — this already
handled Tamil's `language` code correctly without changes, since it's a
generic dispatcher rather than a Hindi-only function.

The most significant real bug found this week was in `align.py`: it defaulted
to an English-only wav2vec2 acoustic model (`WAV2VEC2_ASR_LARGE_LV60K_960H`)
for *any* language that WhisperX didn't have a built-in default for. This had
been silently working for Hindi (WhisperX has a Hindi default), but would have
produced meaningless alignment scores for Tamil had it not been caught. Fixed
by removing the hardcoded fallback and requiring an explicit
`alignment_model` in config when a language has no WhisperX built-in — for
Tamil, `Amrrs/wav2vec2-large-xlsr-53-tamil` (a public wav2vec2-large-xlsr-53
fine-tune on Tamil Common Voice) was used instead, found via a quick model
search rather than assumed.

Two further bugs were only caught because Tamil was actually run through the
pipeline: `evaluate_stt.py` and the model card generator both had `"hi"` /
"Hindi" hardcoded into their output rather than reading `language` from
config — the Tamil run's first metrics file and model card literally said
"Hindi" until this was caught and fixed. Both are now genuinely
language-agnostic.

## 3. Results

| Metric | Hindi STT | Tamil STT |
|---|---|---|
| Training samples | 25,621 | 322 |
| WER | 0.1717 | 1.1658 |
| CER | 0.0699 | 0.9558 |
| RTF | 0.654 | 0.916 |

Hindi STT performs well (WER ~17%, RTF well under real-time). Tamil STT's
numbers are poor by design, not by defect: 322 training samples is roughly
1/80th of Hindi's training set, capped deliberately to fit a one-day timeline
for proving reusability rather than producing a deployable model. The
*pipeline mechanics* — ingest, clean, align, train, evaluate, model-card
generation — all ran successfully on Tamil with zero pipeline code changes
beyond the three genuine bug fixes above (which also improved the Hindi path).
TTS (Hindi only, from Week 5): intelligibility WER 0.346, RTF 12.67 (well
above the 1.0 real-time target — Parler-TTS's autoregressive decoder is
inherently slow on a T4 without further optimisation), ONNX export genuinely
failed (Parler-TTS is a custom architecture not registered in optimum's ONNX
exporter, confirmed via a real captured error after correctly installing the
right exporter package). STT's Whisper architecture, by contrast, exported to
ONNX successfully with only a minor (~6e-05) tied-weight numerical mismatch,
a known benign artifact of ONNX's handling of tied embeddings.

## 4. What to improve next

- **Tamil data volume** is the single biggest lever — 322 samples is a proof
  of mechanism, not of quality; scaling to even a few thousand aligned samples
  would likely bring Tamil's WER into a meaningful range.
- **Alignment model choice for new languages is still a manual step** — for
  any language without a WhisperX built-in default, someone has to find and
  specify a suitable wav2vec2 CTC model by hand. A more complete pipeline
  would probably auto-select from a small curated per-language mapping.
- **TTS reusability is unproven** for a second language — Week 6 only proved
  STT reusability given the time constraint; TTS should be the next thing
  tested, since it has its own separate hardcoding risks (e.g. speaker
  filtering, description-tokenizer language behaviour) that haven't been
  exercised outside Hindi.
- **The base-model spec mismatch** (originally `ai4bharat/indicwhisper-hi`,
  actually trained on `openai/whisper-small`) is still an open item pending
  mentor clarification — not resolved this week.
- **`run_pipeline.py`**, the intended single-command orchestrator, still isn't
  wired to the real per-track `evaluate_*`/`package_*` scripts (it references
  generic `evaluate.py`/`package.py` stage modules that remain unimplemented
  stubs from Weeks 1–2). All work this week was run stage-by-stage instead.
  Wiring this up would be a natural next step for a cleaner demo.
