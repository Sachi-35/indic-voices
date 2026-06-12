# Indic Voice Pipeline — vSense AI

A config-driven pipeline for fine-tuning STT and TTS models for Indian languages.

**The core rule: adding a new language = new config file + new data. Zero code changes.**

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full data prep pipeline for Hindi STT (Stages 1-3)
python run_pipeline.py --config configs/hindi_stt.yaml --stages ingest clean align

# Run a single stage
python run_pipeline.py --config configs/hindi_stt.yaml --stages clean
```

## Pipeline Stages

| Stage | File | What it does |
|---|---|---|
| 1 ingest | `pipeline/ingest.py` | Download dataset, standardise audio, write catalogue.csv |
| 2 clean | `pipeline/clean.py` | Filter bad audio/text, normalise Hindi text |
| 3 align | `pipeline/align.py` | Word-level timestamp alignment via WhisperX → manifest.json |
| 4 train | `pipeline/train.py` | Fine-tune base model (Week 3) |
| 5 evaluate | `pipeline/evaluate.py` | WER/CER/MOS metrics + model card (Week 4-5) |
| 6 package | `pipeline/package.py` | Export to ONNX, latency test (Week 5-6) |

## Adding a New Language

1. Copy `configs/hindi_stt.yaml` → `configs/bengali_stt.yaml`
2. Change `language`, `script`, `dataset_language_key`, `base_model`, and `paths`
3. Run: `python run_pipeline.py --config configs/bengali_stt.yaml`

## Output Files

After Stages 1-3, you will have:

```
data/processed/hi/
├── catalogue.csv          # raw index from Stage 1
├── cleaned_catalogue.csv  # filtered index from Stage 2
├── clean_report.txt       # what was dropped and why
└── manifest.json          # final aligned data — input to Stage 4
```

Each line of `manifest.json`:
```json
{"audio_path": "data/raw/hi/sample_000001.wav", "transcript": "नमस्ते", "duration": 2.4, "language": "hi", "align_score": 0.91}
```