"""Console entry point for static PHASE configuration validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from phase.config_validation import discover_config_paths, validate_config_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=["configs"])
    parser.add_argument(
        "--check-paths",
        action="store_true",
        help="require expanded data, feature, statistics, and checkpoint paths",
    )
    args = parser.parse_args()
    paths = []
    for raw in args.paths:
        path = Path(raw)
        paths.extend(discover_config_paths(path) if path.is_dir() else [path])
    paths = sorted(set(paths))
    if not paths:
        parser.error("no YAML configurations found")
    failed = 0
    for path in paths:
        errors = validate_config_file(path, check_paths=args.check_paths)
        if errors:
            failed += 1
            print(f"FAIL {path}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"OK   {path}")
    if failed:
        raise SystemExit(f"{failed} configuration(s) failed validation")
    print(f"Validated {len(paths)} configuration(s).")


if __name__ == "__main__":
    main()
