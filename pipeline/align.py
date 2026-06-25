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

Resume behaviour
-----------------
  manifest.json is written INCREMENTALLY — one line per sample, flushed
  to disk immediately. If the process is interrupted (Colab disconnect,
  manual stop, crash), re-running this stage will:
    - read whatever is already in manifest.json
    - skip any audio_path already present there
    - append only the remaining samples
  This means it is always safe to stop and resume later without losing
  progress or duplicating work.
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
    logger.info(f"Samples in catalogue: {len(rows)}")

    # --- Resume support: find out what's already been done -----------------
    already_done = _read_existing_manifest_paths(manifest_path)
    if already_done:
        logger.info(
            f"Found existing manifest with {len(already_done)} entries — "
            f"these will be skipped."
        )

    rows_remaining = [r for r in rows if r["audio_path"] not in already_done]
    skipped_already_done = len(rows) - len(rows_remaining)

    logger.info(f"Already aligned   : {skipped_already_done}")
    logger.info(f"Remaining to align: {len(rows_remaining)}")

    if not rows_remaining:
        logger.info("Nothing left to align — manifest is already complete.")
        return manifest_path

    # Load WhisperX alignment model once (expensive to reload per sample)
    align_model, align_metadata, device = _load_align_model(config)

    min_score = float(config.get("min_alignment_score", 0.5))
    dropped = 0
    written = 0

    # Open manifest in APPEND mode — existing entries (if any) are preserved.
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        for idx, row in enumerate(rows_remaining):
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

            entry = {
                "audio_path": row["audio_path"],
                "transcript": row["transcript"],
                "duration": float(row["duration"]),
                "language": row["language"],
                "align_score": round(score, 4),
            }

            # Write immediately and flush to disk — this is the key change.
            # Even if the process is killed right after this line, the
            # entry is already safely on disk in Drive.
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            written += 1

            if (idx + 1) % 200 == 0:
                logger.info(f"  Aligned {idx + 1}/{len(rows_remaining)} this run "
                            f"({written} written, {dropped} dropped)…")

    total_in_manifest = skipped_already_done + written
    logger.info(f"Manifest written : {manifest_path}  ({total_in_manifest} total entries)")
    logger.info(f"  - from this run     : {written}")
    logger.info(f"  - already done      : {skipped_already_done}")
    logger.info(f"  - dropped this run  : {dropped}")
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
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


def _read_existing_manifest_paths(path: Path) -> set[str]:
    """
    Read an existing (possibly partial) manifest.json, if present,
    and return the set of audio_path values already processed.
    Used to support resuming an interrupted align run.

    Tolerant of a trailing incomplete/corrupted last line (e.g. if the
    process was killed mid-write) — that one line is simply skipped.
    """
    if not path.exists():
        return set()

    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                done.add(entry["audio_path"])
            except (json.JSONDecodeError, KeyError):
                # Likely the last line, cut off mid-write. Skip it —
                # that sample will simply be re-aligned this run.
                continue
    return done