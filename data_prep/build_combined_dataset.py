"""Build a combined dataset with selected sources.

Sources:
- data/dataset/disha72_chunks.jsonl
- data/dataset/zhang_henshui_chunks.jsonl
- data/dataset/wuxia_chunks_cleaned1.jsonl (only Liang and Ni authors)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, List, Dict, Any

try:
    from opencc import OpenCC
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: install 'opencc-python-reimplemented'.") from exc

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
DATASET_DIR = DATA_ROOT / "dataset"

BAD_SUBSTRINGS = [
    "",
]

LEADING_STRIP_CHARS = ")）】]}」』'\""

CONVERTER = OpenCC("t2s")

SOURCES: List[Dict[str, Any]] = [
    {
        "name": "disha72",
        "path": DATASET_DIR / "disha72_chunks.jsonl",
        "merge_every": 2,
    },
    {
        "name": "zhang_henshui",
        "path": DATASET_DIR / "zhang_henshui_chunks.jsonl",
        "merge_every": 2,
    },
    {
        "name": "wuxia_liang_ni",
        "path": DATASET_DIR / "wuxia_chunks_cleaned1.jsonl",
        "author_filter": {"Liang", "Ni"},
    },
]


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {path} line {line_number}: {exc}") from exc


def load_source_records(source: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    author_filter = source.get("author_filter")
    for record in iter_jsonl(source["path"]):
        if author_filter and record.get("author") not in author_filter:
            continue
        enriched = dict(record)
        enriched["dataset"] = source["name"]
        text = enriched.get("text")
        if isinstance(text, str):
            enriched["text"] = sanitize_text(text)
        if "author" not in enriched or enriched["author"] is None:
            enriched["author"] = ""
        yield enriched


def iter_processed_records(source: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    records = load_source_records(source)
    merge_every = source.get("merge_every", 0)
    if merge_every and merge_every > 1:
        return merge_every_records(records, merge_every, source["name"])
    return records


def merge_every_records(
    records: Iterator[Dict[str, Any]], group_size: int, dataset_name: str
) -> Iterator[Dict[str, Any]]:
    buffer: List[Dict[str, Any]] = []
    counter = 0
    for record in records:
        buffer.append(record)
        if len(buffer) == group_size:
            counter += 1
            yield build_merged_record(buffer, dataset_name, counter)
            buffer = []

    if buffer:
        counter += 1
        yield build_merged_record(buffer, dataset_name, counter)


def build_merged_record(
    records: List[Dict[str, Any]], dataset_name: str, index: int
) -> Dict[str, Any]:
    merged_text = "\n\n".join(item.get("text", "") for item in records if item.get("text"))
    base = dict(records[0])
    base["id"] = f"{dataset_name}_pair_{index:05d}"
    base["text"] = merged_text
    base["merged_from"] = [item.get("id") for item in records]
    base["merged_count"] = len(records)
    return base


def sanitize_text(text: str) -> str:
    cleaned = text
    for bad in BAD_SUBSTRINGS:
        cleaned = cleaned.replace(bad, "")
    cleaned = CONVERTER.convert(cleaned)
    cleaned = cleaned.lstrip(LEADING_STRIP_CHARS)
    return cleaned


def build_dataset(output_path: Path, sources: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, int] = {}
    with output_path.open("w", encoding="utf-8") as outfile:
        for source in sources:
            count = 0
            for record in iter_processed_records(source):
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            stats[source["name"]] = count
    stats["total"] = sum(stats.values())
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine multiple JSONL datasets.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DATASET_DIR / "combined_dataset.jsonl",
        help="Path for the combined JSONL file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [src["path"] for src in SOURCES if not src["path"].exists()]
    if missing:
        formatted = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing source files:\n{formatted}")

    stats = build_dataset(args.output, SOURCES)
    print("Wrote", stats["total"], "records to", args.output)
    for name, count in stats.items():
        if name == "total":
            continue
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
