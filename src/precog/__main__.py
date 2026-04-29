from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

from .runtime import PreCog


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="precog")
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("doctor", help="print environment and package diagnostics")
    subcommands.add_parser("metrics", help="print empty runtime metrics in Prometheus format")

    inspect_state = subcommands.add_parser(
        "inspect-state",
        help="summarize a PreCog state JSON file",
    )
    inspect_state.add_argument("path", type=Path)

    args = parser.parse_args(argv)
    if args.command == "doctor":
        print(json.dumps(_doctor(), indent=2, sort_keys=True))
        return 0
    if args.command == "metrics":
        print(PreCog().metrics_text(), end="")
        return 0
    if args.command == "inspect-state":
        print(json.dumps(_inspect_state(args.path), indent=2, sort_keys=True))
        return 0
    return 1


def _doctor() -> dict[str, str]:
    try:
        package_version = version("precog")
    except PackageNotFoundError:
        package_version = "editable"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "precog": package_version,
        "executable": sys.executable,
    }


def _inspect_state(path: Path) -> dict[str, int]:
    state = json.loads(path.read_text(encoding="utf-8"))
    predictor = state.get("predictor", {})
    cache = state.get("cache", {})
    return {
        "args_memory_tools": len(predictor.get("args_memory", {})),
        "bigram_sources": len(predictor.get("bigrams", {})),
        "cache_entries": len(cache.get("entries", [])),
    }


if __name__ == "__main__":
    raise SystemExit(main())
