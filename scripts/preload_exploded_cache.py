from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.exploded_view_engine import ExplodedViewEngine


IPHONE_MODELS = [
    "iPhone 14",
    "iPhone 14 Plus",
    "iPhone 14 Pro",
    "iPhone 14 Pro Max",
    "iPhone 15",
    "iPhone 15 Plus",
    "iPhone 15 Pro",
    "iPhone 15 Pro Max",
    "iPhone 16",
    "iPhone 16 Plus",
    "iPhone 16 Pro",
    "iPhone 16 Pro Max",
    "iPhone 16e",
]

SAMSUNG_MODELS = [
    "Samsung Galaxy S20",
    "Samsung Galaxy S20+",
    "Samsung Galaxy S20 Ultra",
    "Samsung Galaxy S20 FE",
    "Samsung Galaxy S21",
    "Samsung Galaxy S21+",
    "Samsung Galaxy S21 Ultra",
    "Samsung Galaxy S21 FE",
    "Samsung Galaxy S22",
    "Samsung Galaxy S22+",
    "Samsung Galaxy S22 Ultra",
    "Samsung Galaxy S23",
    "Samsung Galaxy S23+",
    "Samsung Galaxy S23 Ultra",
    "Samsung Galaxy S23 FE",
    "Samsung Galaxy S24",
    "Samsung Galaxy S24+",
    "Samsung Galaxy S24 Ultra",
    "Samsung Galaxy S24 FE",
    "Samsung Galaxy S25",
    "Samsung Galaxy S25+",
    "Samsung Galaxy S25 Ultra",
    "Samsung Galaxy S25 Edge",
    "Samsung Galaxy S25 FE",
    "Samsung Galaxy S26",
    "Samsung Galaxy S26+",
    "Samsung Galaxy S26 Ultra",
]


def _default_models() -> list[str]:
    return IPHONE_MODELS + SAMSUNG_MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preload exploded-view cache entries for supported phone models."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Delete existing cache folders for a model before re-fetching.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Fetch only the given model. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = args.models or _default_models()
    engine = ExplodedViewEngine()

    if engine.client is None:
        print("Missing GEMINI_API_KEY. Set it in .env before preloading.")
        return 1

    print(f"Preloading exploded cache for {len(models)} models...")
    print(f"Cache directory: {engine.cache_dir}")

    success_count = 0
    results: list[tuple[str, bool, str]] = []
    for index, model_name in enumerate(models, start=1):
        print(f"[{index}/{len(models)}] {model_name}")
        ok, message = engine.load_for_model(
            model_name,
            category="smartphone",
            refresh=args.refresh,
        )
        if ok:
            success_count += 1
            cache_root = engine.cache_root_for_model(model_name)
            print(f"  OK   {message}")
            print(f"  PATH {cache_root}")
        else:
            print(f"  FAIL {message}")
        results.append((model_name, ok, message))

    print("")
    print(f"Completed: {success_count}/{len(models)} models cached successfully.")
    failed = [row for row in results if not row[1]]
    if failed:
        print("Models without cached images:")
        for model_name, _, message in failed:
            print(f"  - {model_name}: {message}")

    return 0 if success_count == len(models) else 2


if __name__ == "__main__":
    raise SystemExit(main())
