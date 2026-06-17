"""
pipeline/ingest.py  —  Stage 1: Ingest
=======================================
Downloads the HuggingFace dataset specified in the config,
standardises every audio file (sample rate, mono, WAV),
and writes a catalogue CSV listing every (audio_path, transcript) pair.

What this file does, in plain English:
  1. Read config to know which dataset / language to pull
  2. Stream the dataset from HuggingFace (avoids downloading everything at once)
  3. For each sample: save the audio as a standardised WAV, record the transcript
  4. Write catalogue.csv  →  one row per sample, columns: audio_path, transcript, duration

Output
------
  data/processed/<lang>/catalogue.csv
"""

import os
import csv
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point — called by run_pipeline.py
# ---------------------------------------------------------------------------

def run(config: dict) -> Path:
    """
    Run the ingest stage.

    Parameters
    ----------
    config : dict
        Parsed YAML config (see configs/hindi_stt.yaml for all keys).

    Returns
    -------
    Path
        Path to the written catalogue CSV.
    """
    raw_dir = Path(config["paths"]["raw_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    catalogue_path = Path(config["paths"]["catalogue"])

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== STAGE 1: INGEST ===")
    logger.info(f"Dataset  : {config['dataset']}")
    logger.info(f"Language : {config['language']}")
    logger.info(f"Raw dir  : {raw_dir}")

    # ------------------------------------------------------------------
    # 1. Load dataset from HuggingFace
    #    streaming=True means it downloads one sample at a time — safe
    #    for large datasets on limited disk space.
    # ------------------------------------------------------------------
    logger.info("Loading dataset from HuggingFace (streaming)…")
    try:
        dataset = load_dataset(
            config["dataset"],
            config.get("dataset_language_key", config["language"]),
            split=config.get("dataset_split", "train"),
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as exc:
        # Some datasets don't use a language sub-config — fall back
        logger.warning(f"Language key failed ({exc}), retrying without sub-config…")
        dataset = load_dataset(
            config["dataset"],
            split=config.get("dataset_split", "train"),
            streaming=True,
            trust_remote_code=True,
        )

    target_sr = config["sample_rate"]
    rows = []

    logger.info("Processing samples…")
    for idx, sample in enumerate(dataset):
        try:
            audio_path, duration = _save_audio(sample, idx, raw_dir, target_sr)
            transcript = _extract_transcript(sample)

            if transcript is None:
                logger.debug(f"Sample {idx}: no transcript found, skipping")
                continue

            rows.append({
                "audio_path": str(audio_path),
                "transcript": transcript,
                "duration": round(duration, 3),
                "language": config["language"],
            })

            if (idx + 1) % 500 == 0:
                logger.info(f"  Processed {idx + 1} samples…")

        except Exception as exc:
            logger.warning(f"Sample {idx} failed: {exc}")
            continue

    # ------------------------------------------------------------------
    # 2. Write catalogue CSV
    # ------------------------------------------------------------------
    _write_catalogue(rows, catalogue_path)
    logger.info(f"Catalogue written → {catalogue_path}  ({len(rows)} rows)")
    return catalogue_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_audio(sample: dict, idx: int, raw_dir: Path, target_sr: int):
    """
    Extract audio from a HuggingFace sample, resample to target_sr,
    convert to mono, save as WAV.

    Returns (path, duration_seconds).
    """
    audio_data = sample.get("audio", {})
    array = np.array(audio_data["array"], dtype=np.float32)
    original_sr = audio_data["sampling_rate"]

    # Resample if needed
    if original_sr != target_sr:
        array = _resample(array, original_sr, target_sr)

    # Force mono: if stereo (2D array), average channels
    if array.ndim > 1:
        array = array.mean(axis=1)

    duration = len(array) / target_sr
    out_path = raw_dir / f"sample_{idx:06d}.wav"
    sf.write(str(out_path), array, target_sr, subtype="PCM_16")
    return out_path, duration


def _resample(array: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """
    Simple resample using numpy (avoids adding librosa as a hard dependency here).
    For production, swap this with librosa.resample for better quality.
    """
    if from_sr == to_sr:
        return array
    # librosa is imported lazily so the file still runs even without it
    try:
        import librosa
        return librosa.resample(array, orig_sr=from_sr, target_sr=to_sr)
    except ImportError:
        # Fallback: crude linear interpolation
        ratio = to_sr / from_sr
        new_length = int(len(array) * ratio)
        indices = np.linspace(0, len(array) - 1, new_length)
        return np.interp(indices, np.arange(len(array)), array).astype(np.float32)


def _extract_transcript(sample: dict) -> str | None:
    """
    IndicVoices stores transcripts under various keys depending on the version.
    Try each known key in order.
    """
    for key in ("text", "transcription", "transcript", "sentence"):
        val = sample.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _write_catalogue(rows: list[dict], path: Path):
    """Write list of dicts to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "transcript", "duration", "language"])
        writer.writeheader()
        writer.writerows(rows)