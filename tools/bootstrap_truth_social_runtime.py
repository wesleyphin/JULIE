#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINBERT_DIR = ROOT / "models" / "finbert"
REMOTE_FINBERT_ID = "ProsusAI/finbert"
REQUIRED_MODULES = ("truthbrush", "transformers", "torch", "accelerate")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def ensure_finbert_assets(model_dir: Path, *, force: bool = False) -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    config_path = model_dir / "config.json"
    tokenizer_path = model_dir / "tokenizer_config.json"
    if not force and config_path.exists() and tokenizer_path.exists():
        print(f"FinBERT assets already available at {model_dir}")
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading FinBERT to {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(REMOTE_FINBERT_ID)
    model = AutoModelForSequenceClassification.from_pretrained(REMOTE_FINBERT_ID)
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)
    print("FinBERT assets downloaded.")


def smoke_test(model_dir: Path) -> dict:
    from services.sentiment_service import build_truth_social_sentiment_service

    service = build_truth_social_sentiment_service(
        {
            "enabled": True,
            "finbert_local_path": str(model_dir),
            "target_handle": "realDonaldTrump",
        }
    )
    analysis = service._classify_text("Markets are rallying after upbeat economic news.")  # noqa: SLF001
    snapshot = service.snapshot()
    metadata = dict(snapshot.get("metadata") or {})
    return {
        "sentiment_label": analysis.get("sentiment_label"),
        "sentiment_score": analysis.get("sentiment_score"),
        "finbert_confidence": analysis.get("finbert_confidence"),
        "quantized_8bit": bool(snapshot.get("quantized_8bit")),
        "quantization_mode": metadata.get("quantization_mode"),
        "model_source": metadata.get("model_source"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify/install the local Truth Social + FinBERT runtime assets."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_FINBERT_DIR,
        help="Directory where the local FinBERT snapshot should live.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the local FinBERT snapshot even if it already exists.",
    )
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Skip the local FinBERT load/classification smoke test.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    missing = [name for name in REQUIRED_MODULES if not module_available(name)]
    if missing:
        raise SystemExit(
            "Missing sentiment runtime dependencies: "
            + ", ".join(missing)
            + ". Install requirements first, then rerun this bootstrap."
        )

    ensure_finbert_assets(args.model_dir, force=args.force_download)

    if args.skip_smoke_test:
        smoke = None
    else:
        print("Running FinBERT smoke test ...")
        smoke = smoke_test(args.model_dir)
        print(json.dumps(smoke, indent=2))

    if platform.system() != "Linux":
        print("bitsandbytes is optional on this platform; Julie will use the best available local fallback (dynamic int8 or fp32).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
