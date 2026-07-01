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

Resume behaviour
-----------------
  catalogue.csv is written INCREMENTALLY — one row per sample, flushed to
  disk immediately. If the process is interrupted, re-running this stage
  will:
    - read whatever rows already exist in catalogue.csv
    - confirm the corresponding .wav file for each row still exists on disk
      (rows whose file is missing, e.g. deleted by accident, are treated
      as NOT done and will be re-processed)
    - skip that many samples at the START of the streaming iterator
      (streaming datasets can't be randomly seeked, so already-done
      samples are still streamed past, just not re-saved/re-processed)
    - append only new, valid rows from where it left off
  This means it is safe to stop and resume later without losing progress,
  and safe even if some already-downloaded .wav files were deleted —
  those specific samples will simply be redone.
"""

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
    # 0. Resume support: check what's already validly done
    # ------------------------------------------------------------------
    existing_rows = _read_existing_catalogue(catalogue_path)
    valid_existing_rows, n_missing_files = _filter_rows_with_valid_audio(existing_rows)

    n_resume_skip = len(valid_existing_rows)

    if existing_rows:
        logger.info(
            f"Found existing catalogue with {len(existing_rows)} rows "
            f"({n_resume_skip} have valid audio on disk, "
            f"{n_missing_files} are missing their .wav file and will be redone)."
        )
        logger.info(f"Will skip the first {n_resume_skip} samples in the stream.")

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
            trust_remote_code=False,
        )
    except Exception as exc:
        # Some datasets don't use a language sub-config — fall back
        logger.warning(f"Language key failed ({exc}), retrying without sub-config…")
        dataset = load_dataset(
            config["dataset"],
            split=config.get("dataset_split", "train"),
            streaming=True,
            trust_remote_code=False,
        )

    target_sr = config["sample_rate"]

    logger.info("Processing samples…")

    # Open catalogue in append mode. If we have valid existing rows, the
    # file already contains them (header + N rows) — we just keep adding.
    # If the catalogue doesn't exist yet, or had zero valid rows, we
    # (re)write it fresh with a header first.
    write_header = (n_resume_skip == 0)

    n_new_rows = 0
    n_skipped_in_stream = 0
    next_index = n_resume_skip  # used to keep filenames unique/sequential

    with open(catalogue_path, "a" if not write_header else "w",
              newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["audio_path", "transcript", "duration", "language"]
        )
        if write_header:
            writer.writeheader()

        for idx, sample in enumerate(dataset):
            # Skip samples already done in a previous run. We still have
            # to iterate past them (streaming can't seek), but we don't
            # redo any of the expensive save/resample work.
            if idx < n_resume_skip:
                n_skipped_in_stream += 1
                continue

            try:
                audio_path, duration = _save_audio(sample, idx, raw_dir, target_sr)
                transcript = _extract_transcript(sample)

                if transcript is None:
                    logger.debug(f"Sample {idx}: no transcript found, skipping")
                    continue

                row = {
                    "audio_path": str(audio_path),
                    "transcript": transcript,
                    "duration": round(duration, 3),
                    "language": config["language"],
                }

                writer.writerow(row)
                f.flush()
                n_new_rows += 1
                next_index = idx + 1

                if (idx + 1) % 500 == 0:
                    logger.info(f"  Processed {idx + 1} samples so far "
                                f"({n_new_rows} new this run)…")

            except Exception as exc:
                logger.warning(f"Sample {idx} failed: {exc}")
                continue

    total_rows = n_resume_skip + n_new_rows
    logger.info(f"Catalogue written → {catalogue_path}  ({total_rows} total rows)")
    logger.info(f"  - already done (resumed)  : {n_resume_skip}")
    logger.info(f"  - new this run            : {n_new_rows}")
    return catalogue_path


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _read_existing_catalogue(path: Path) -> list[dict]:
    """
    Read an existing (possibly partial) catalogue.csv, if present.
    Returns a list of row dicts, in order. Tolerant of a missing file
    (returns empty list).
    """
    if not path.exists():
        return []

    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _filter_rows_with_valid_audio(rows: list[dict]) -> tuple[list[dict], int]:
    """
    Given existing catalogue rows, check that each row's audio_path
    still actually exists on disk (non-zero size). This handles the
    case where .wav files were deleted (accidentally or otherwise)
    after being logged — those rows should NOT be trusted as "done".

    IMPORTANT: this function assumes existing rows are a contiguous
    prefix of the dataset stream (sample_000000.wav, sample_000001.wav, …
    with no gaps). If any row in the middle has a missing file, we treat
    everything from that point onward as needing to be redone — this
    keeps the "skip first N samples in the stream" resume logic simple
    and correct. Trailing/scattered partial deletions are the expected
    case (e.g. accidentally deleting just the tail of a folder); deletions
    of files in the middle are less common but handled safely the same way.

    Returns
    -------
    (valid_rows, n_missing) — valid_rows is the contiguous prefix that's
    safe to resume from; n_missing is how many rows from that point on
    were dropped because their file was missing.
    """
    valid_rows = []
    for row in rows:
        audio_path = Path(row["audio_path"])
        if audio_path.exists() and audio_path.stat().st_size > 0:
            valid_rows.append(row)
        else:
            # First missing file found — stop trusting the rest of the
            # catalogue as "done", even if later files happen to still
            # exist, to keep the resume index simple and correct.
            break

    n_missing = len(rows) - len(valid_rows)
    return valid_rows, n_missing


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