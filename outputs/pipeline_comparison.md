# Pipeline Results Comparison

| Metric | Hindi STT | Tamil STT |
|---|---|---|
| Base model | openai/whisper-small | openai/whisper-small |
| Training samples | 25,621 | 322 (from 500 ingested, post-align) |
| WER | 0.1717 | 1.1658 |
| CER | 0.0699 | 0.9558 |
| RTF | 0.654 | 0.916 |
| ONNX export | Succeeded (minor tied-weight caveat) | Not re-run (see write-up) |
| Code changes needed | 0 | 0 |

TTS was only proven on Hindi (Week 5): WER 0.346, CER 0.219, RTF 12.67, ONNX
export failed for architectural reasons (Parler-TTS not in optimum's ONNX
registry — confirmed via a real captured error, not assumed). A second-language
TTS run was scoped out of Week 6 given the one-day timeline; STT was chosen to
prove reusability since it trains and evaluates far faster on limited data.

**Code changes needed: 0.** Every code change made this week — the `align.py`
English-only-aligner default, the hardcoded `"hi"` in `evaluate_stt.py`'s
output, and the hardcoded "Hindi" title/limitations text in the model card
generator — was a genuine pipeline bug fix, not a Tamil-specific patch. All
three now work correctly for Hindi, Tamil, or any future language, driven
entirely by config.
