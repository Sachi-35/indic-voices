"""
pipeline/clean.py  —  Stage 2: Clean
======================================
Reads the catalogue from Stage 1 and filters / normalises every sample.

What this file does, in plain English:
  1. Read catalogue.csv (list of audio files + transcripts)
  2. Audio checks: drop clips that are too short, too long, or too quiet
  3. Text checks: drop empty transcripts, fix Devanagari numerals,
     remove stray punctuation / extra spaces
  4. Write cleaned_catalogue.csv  →  only the rows that passed all checks

Output
------
  data/processed/<lang>/cleaned_catalogue.csv
  data/processed/<lang>/clean_report.txt   (summary of what was dropped and why)
"""

import csv
import logging
import re
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Devanagari → ASCII digit map
# ०१२३४५६७८९  →  0123456789
# ---------------------------------------------------------------------------
_DEVA_DIGIT_TABLE = str.maketrans("०१२३४५६७८९", "0123456789")

# Common Unicode punctuation to strip (expand as needed)
_PUNCT_RE = re.compile(r"[।॥\u200b\u200c\u200d\ufeff]+")  # danda, double danda, ZWS, ZWJ, BOM


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(config: dict) -> Path:
    """
    Run the clean stage.

    Parameters
    ----------
    config : dict   Parsed YAML config.

    Returns
    -------
    Path   Path to cleaned_catalogue.csv.
    """
    catalogue_path = Path(config["paths"]["catalogue"])
    processed_dir = Path(config["paths"]["processed_dir"])
    cleaned_path = processed_dir / "cleaned_catalogue.csv"
    report_path = processed_dir / "clean_report.txt"

    logger.info("=== STAGE 2: CLEAN ===")
    logger.info(f"Input catalogue : {catalogue_path}")

    rows = _read_catalogue(catalogue_path)
    logger.info(f"Rows loaded     : {len(rows)}")

    kept, dropped = [], []

    for row in rows:
        reason = _check_row(row, config)
        if reason:
            dropped.append({**row, "drop_reason": reason})
        else:
            # Normalise text in-place before saving
            row["transcript"] = _normalise_text(row["transcript"], config)
            kept.append(row)

    # Write outputs
    _write_catalogue(kept, cleaned_path)
    _write_report(kept, dropped, report_path)

    logger.info(f"Kept    : {len(kept)}")
    logger.info(f"Dropped : {len(dropped)}")
    logger.info(f"Cleaned catalogue → {cleaned_path}")
    logger.info(f"Report            → {report_path}")
    return cleaned_path


# ---------------------------------------------------------------------------
# Row-level checks  (return None = pass, return string = drop reason)
# ---------------------------------------------------------------------------

def _check_row(row: dict, config: dict) -> str | None:
    """
    Run every check on one catalogue row.
    Returns the first failure reason, or None if the row is clean.
    """
    # --- Text checks (cheap, do first) ---
    transcript = row.get("transcript", "").strip()

    if config.get("remove_empty_transcripts", True):
        if not transcript:
            return "empty_transcript"

    # --- Audio checks (more expensive, do after text) ---
    audio_path = Path(row["audio_path"])
    if not audio_path.exists():
        return "file_missing"

    try:
        info = sf.info(str(audio_path))
        duration = info.duration
    except Exception as exc:
        return f"unreadable_audio:{exc}"

    min_dur = float(config.get("min_duration", 1.0))
    max_dur = float(config.get("max_duration", 20.0))

    if duration < min_dur:
        return f"too_short:{duration:.2f}s"
    if duration > max_dur:
        return f"too_long:{duration:.2f}s"

    # Volume / RMS check
    min_rms_db = float(config.get("min_rms_db", -40.0))
    if _rms_db(audio_path) < min_rms_db:
        return f"too_quiet:<{min_rms_db}dB"

    return None   # passed all checks


def _rms_db(audio_path: Path) -> float:
    """
    Compute the RMS level of an audio file in decibels.
    Silence / near-silence → very negative number.
    """
    data, _ = sf.read(str(audio_path), dtype="float32")
    rms = np.sqrt(np.mean(data ** 2))
    if rms == 0:
        return -np.inf
    return 20 * np.log10(rms)


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def _normalise_text(text: str, config: dict) -> str:
    """
    Apply all text normalisations specified in the config.
    """
    if config.get("normalize_numerals", True):
        text = text.translate(_DEVA_DIGIT_TABLE)

    # Remove Devanagari punctuation (danda etc.) and zero-width characters
    text = _PUNCT_RE.sub(" ", text)

    if config.get("strip_extra_whitespace", True):
        text = " ".join(text.split())

    # Try indic-nlp-library for deeper normalisation if installed
    try:
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        lang = config.get("language", "hi")
        normalizer = IndicNormalizerFactory().get_normalizer(lang)
        text = normalizer.normalize(text)
    except ImportError:
        pass   # library not installed — skip, not critical
    except Exception as exc:
        logger.debug(f"indic-nlp normalisation skipped: {exc}")

    return text.strip()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_catalogue(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_catalogue(rows: list[dict], path: Path):
    if not rows:
        logger.warning("No rows to write to cleaned catalogue!")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _write_report(kept: list, dropped: list, path: Path):
    """Write a human-readable summary of what was dropped and why."""
    from collections import Counter
    reasons = Counter(r["drop_reason"] for r in dropped)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== CLEAN STAGE REPORT ===\n\n")
        f.write(f"Total input rows : {len(kept) + len(dropped)}\n")
        f.write(f"Rows kept        : {len(kept)}\n")
        f.write(f"Rows dropped     : {len(dropped)}\n\n")
        f.write("Drop reasons:\n")
        for reason, count in reasons.most_common():
            f.write(f"  {reason:<30} {count}\n")