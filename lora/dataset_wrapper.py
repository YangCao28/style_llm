from __future__ import annotations

import argparse
import json
from pathlib import Path


def ensure_uniform_fields(path: Path, output_path: Path | None = None) -> Path:
    path = path.resolve()
    if output_path is None:
        output_path = path

    with path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
        for line in infile:
            if not line.strip():
                continue
            obj = json.loads(line)
            merged_from = obj.get("merged_from")
            if not isinstance(merged_from, list):
                merged_from = []
            obj["merged_from"] = merged_from
            obj["merged_count"] = int(obj.get("merged_count", len(merged_from)))
            obj["author"] = obj.get("author") or ""
            outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize combined dataset fields")
    parser.add_argument("--input", type=Path, required=True, help="Source JSONL (combined_dataset.jsonl)")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination JSONL (defaults to overwriting input).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_uniform_fields(args.input, args.output)


if __name__ == "__main__":
    main()
