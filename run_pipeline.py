"""
run_pipeline.py  —  Pipeline Orchestrator
==========================================
The single entry point for the entire pipeline.

Usage
-----
  python run_pipeline.py --config configs/hindi_stt.yaml
  python run_pipeline.py --config configs/hindi_stt.yaml --stages ingest clean
  python run_pipeline.py --config configs/hindi_tts.yaml --stages ingest clean align

Arguments
---------
  --config   Path to a YAML config file. Required.
  --stages   Space-separated list of stages to run.
             Default: all stages in order.
             Options: ingest  clean  align  train  evaluate  package

How it works
------------
  1. Load the YAML config into a plain Python dict
  2. Run each requested stage by calling its run(config) function
  3. Stop immediately if any stage fails (fail-fast)
  4. Print a summary at the end

Adding a new language
---------------------
  1. Create configs/new_language_stt.yaml  (copy hindi_stt.yaml, change values)
  2. Add data to data/raw/<lang>/
  3. Run: python run_pipeline.py --config configs/new_language_stt.yaml
  Zero code changes needed.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

# Stage modules — one import per pipeline stage
from pipeline import ingest, clean, align, train, evaluate, package

# ---------------------------------------------------------------------------
# Logging setup  (INFO to console, DEBUG to file)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/pipeline.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("run_pipeline")


# ---------------------------------------------------------------------------
# Stage registry  (order matters)
# ---------------------------------------------------------------------------
STAGES = {
    "ingest":   ingest.run,
    "clean":    clean.run,
    "align":    align.run,
    "train":    train.run,
    "evaluate": evaluate.run,
    "package":  package.run,
}
STAGE_ORDER = list(STAGES.keys())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Indic Voice Pipeline — config-driven STT/TTS fine-tuning"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file, e.g. configs/hindi_stt.yaml",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_ORDER,
        default=STAGE_ORDER,
        help="Which stages to run (default: all). Example: --stages ingest clean align",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Ensure outputs directory exists for the log file
    Path("outputs").mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Pipeline start")
    logger.info(f"Config   : {config_path}")
    logger.info(f"Language : {config.get('language_name', config.get('language', '?'))}")
    logger.info(f"Stages   : {' → '.join(args.stages)}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 2. Run each requested stage
    # ------------------------------------------------------------------
    results = {}
    pipeline_start = time.time()

    for stage_name in args.stages:
        stage_fn = STAGES[stage_name]
        logger.info(f"\n▶  Running stage: {stage_name.upper()}")
        stage_start = time.time()

        try:
            output = stage_fn(config)
            elapsed = time.time() - stage_start
            results[stage_name] = {"status": "✓ OK", "output": output, "time": elapsed}
            logger.info(f"✓  {stage_name} completed in {elapsed:.1f}s  →  {output}")

        except NotImplementedError:
            logger.warning(f"⚠  {stage_name} is not yet implemented — skipping")
            results[stage_name] = {"status": "⚠ SKIPPED (not implemented)", "output": None}

        except Exception as exc:
            elapsed = time.time() - stage_start
            results[stage_name] = {"status": f"✗ FAILED: {exc}", "output": None, "time": elapsed}
            logger.exception(f"✗  {stage_name} failed after {elapsed:.1f}s")
            logger.error("Pipeline stopped (fail-fast). Fix the error above and re-run.")
            _print_summary(results, time.time() - pipeline_start)
            sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Print summary
    # ------------------------------------------------------------------
    _print_summary(results, time.time() - pipeline_start)


def _print_summary(results: dict, total_time: float):
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    for stage, info in results.items():
        logger.info(f"  {stage:<12}  {info['status']}")
    logger.info(f"\n  Total time: {total_time:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()