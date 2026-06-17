"""
pipeline/align.py  —  Stage 3: Align
======================================
Takes the cleaned catalogue and uses WhisperX to precisely align
each audio file with its transcript at the word level.

What this file does, in plain English:
  1. Load the cleaned catalogue (audio path + transcript per row)
  2. For each sample, run WhisperX forced-alignment
     → this pins each word to an exact start/end timestamp in the audio
  3. Compute a per-sample confidence score (average of word scores)
  4. Drop samples where confidence is below the threshold in config
  5. Write manifest.json — the final output of Stages 1-3, ready for training

Why alignment matters:
  The STT model learns by seeing "this audio chunk = this text".
  If the audio and text are even slightly out of sync the model
  learns noise, not language. WhisperX fixes that.

Output
------
  data/processed/<lang>/manifest.json
    Each line: {"audio_path": "...", "transcript": "...",
                "duration": 4.2, "language": "hi", "align_score": 0.87}
"""

import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(config: dict) -> Path:
    """
    Run the align stage.

    Parameters
    ----------
    config : dict   Parsed YAML config.

    Returns
    -------
    Path   Path to manifest.json.
    """
    processed_dir = Path(config["paths"]["processed_dir"])
    cleaned_catalogue = processed_dir / "cleaned_catalogue.csv"
    manifest_path = Path(config["paths"]["manifest"])

    logger.info("=== STAGE 3: ALIGN ===")
    logger.info(f"Input  : {cleaned_catalogue}")
    logger.info(f"Output : {manifest_path}")

    rows = _read_catalogue(cleaned_catalogue)
    logger.info(f"Samples to align: {len(rows)}")

    # Load WhisperX alignment model once (expensive to reload per sample)
    align_model, align_metadata, device = _load_align_model(config)

    min_score = float(config.get("min_alignment_score", 0.5))
    manifest_rows = []
    dropped = 0

    for idx, row in enumerate(rows):
        try:
            score = _align_sample(
                audio_path=row["audio_path"],
                transcript=row["transcript"],
                align_model=align_model,
                align_metadata=align_metadata,
                device=device,
                config=config,
            )
        except Exception as exc:
            logger.warning(f"Sample {idx} alignment failed: {exc}")
            dropped += 1
            continue

        if score < min_score:
            logger.debug(f"Sample {idx} dropped: align_score {score:.3f} < {min_score}")
            dropped += 1
            continue

        manifest_rows.append({
            "audio_path": row["audio_path"],
            "transcript": row["transcript"],
            "duration": float(row["duration"]),
            "language": row["language"],
            "align_score": round(score, 4),
        })

        if (idx + 1) % 200 == 0:
            logger.info(f"  Aligned {idx + 1}/{len(rows)}…")

    _write_manifest(manifest_rows, manifest_path)
    logger.info(f"Manifest written : {manifest_path}  ({len(manifest_rows)} entries)")
    logger.info(f"Dropped          : {dropped}")
    return manifest_path


# ---------------------------------------------------------------------------
# WhisperX helpers
# ---------------------------------------------------------------------------

def _load_align_model(config: dict):
    """
    Load the WhisperX phoneme-alignment model.
    Returns (model, metadata, device).
    """
    try:
        import whisperx
        import torch
    except ImportError as exc:
        raise ImportError(
            "WhisperX is not installed. Run: pip install whisperx"
        ) from exc

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    language = config.get("language", "hi")
    model_name = config.get(
        "alignment_model",
        "WAV2VEC2_ASR_LARGE_LV60K_960H"
    )

    logger.info(f"Loading alignment model '{model_name}' on {device}…")
    model, metadata = whisperx.load_align_model(
        language_code=language,
        device=device,
        model_name=model_name,
    )
    return model, metadata, device


def _align_sample(
    audio_path: str,
    transcript: str,
    align_model,
    align_metadata,
    device: str,
    config: dict,
) -> float:
    """
    Run forced alignment on one sample.

    Returns
    -------
    float   Mean word-level confidence score (0.0 – 1.0).
            Returns 0.0 if alignment produces no scored words.
    """
    import whisperx

    # WhisperX expects audio as a float32 numpy array at the model's sample rate
    audio_array, sr = sf.read(audio_path, dtype="float32")
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    # WhisperX alignment input: list of segment dicts
    segments = [{"text": transcript, "start": 0.0, "end": len(audio_array) / sr}]

    result = whisperx.align(
        segments,
        align_model,
        align_metadata,
        audio_array,
        device,
        return_char_alignments=False,
    )

    return _mean_word_score(result)


def _mean_word_score(align_result: dict) -> float:
    """
    Extract the average word-level score from a WhisperX alignment result.
    WhisperX returns:  {"segments": [{"words": [{"word": "...", "score": 0.9, ...}]}]}
    """
    scores = []
    for segment in align_result.get("segments", []):
        for word in segment.get("words", []):
            s = word.get("score")
            if s is not None:
                scores.append(float(s))

    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_catalogue(path: Path) -> list[dict]:
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_manifest(rows: list[dict], path: Path):
    """
    Write manifest as newline-delimited JSON (one JSON object per line).
    This format is standard for audio training pipelines (NeMo, ESPnet, etc.)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")