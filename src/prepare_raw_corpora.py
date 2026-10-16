"""Audit local raw corpora and explicitly export selected HF datasets to JSONL.

No network access or corpus writes occur in audit/dry-run mode. Downloads are
streamed and capped by default; an unbounded export requires an explicit flag.
Datasets without a configured Hugging Face ID (laws/dialogues) must be supplied
from an approved local source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:
    from src.pilot_experiment_config import (
        CONFIG_FILE_PATH,
        audit_corpus_paths,
        load_pilot_config,
    )
except ImportError:  # direct invocation: python src/prepare_raw_corpora.py
    from pilot_experiment_config import CONFIG_FILE_PATH, audit_corpus_paths, load_pilot_config


def resolve_corpus_path(config: Dict[str, Any], corpus_id: str, root_override: Optional[str]) -> Path:
    status = audit_corpus_paths(config, root_override=root_override)
    if corpus_id not in status:
        raise KeyError(f"Unknown corpus {corpus_id!r}; configured: {', '.join(status)}")
    return Path(status[corpus_id]["effective_path"])


def print_inventory(config: Dict[str, Any], root_override: Optional[str]) -> bool:
    status = audit_corpus_paths(config, root_override=root_override)
    print("CORPUS INVENTORY")
    print(f"{'Corpus':<20} {'Status':<10} {'Files':>7} {'Size MiB':>12}  Path")
    print("-" * 104)
    ready = True
    for corpus_id, entry in status.items():
        found = bool(entry["exists"] and entry["file_count"] > 0)
        ready &= found
        state = "READY" if found else "MISSING"
        print(
            f"{corpus_id:<20} {state:<10} {entry['file_count']:>7} "
            f"{entry['size_mb']:>12.2f}  {entry['effective_path']}"
        )
    print("\nHF datasets can be exported with the download subcommand. Sources without an HF ID require approved local input.")
    return ready


def normalize_row(row: Dict[str, Any], corpus_id: str, info: Dict[str, Any], index: int) -> Optional[Dict[str, str]]:
    text_column = info.get("text_column", "text")
    text = row.get(text_column)
    if not isinstance(text, str) or not text.strip():
        return None
    doc_id = row.get("id") or row.get("url") or row.get("_id")
    if not doc_id:
        doc_id = hashlib.sha256((corpus_id + "\0" + text).encode("utf-8")).hexdigest()
    return {
        "id": str(doc_id),
        "source": corpus_id,
        "type": str(info.get("doc_type", "general_web")),
        "license": str(info.get("license", "Unknown")),
        "text": text,
    }


def export_hf_dataset(
    config: Dict[str, Any],
    corpus_id: str,
    root_override: Optional[str],
    split: str,
    max_docs: int,
    overwrite: bool,
) -> Dict[str, Any]:
    info = config["corpora_sources"][corpus_id]
    dataset_id = info.get("hf_dataset_id")
    if not dataset_id:
        raise ValueError(f"{corpus_id} has no configured hf_dataset_id; provide approved local files instead")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("HF export requires the 'datasets' package in this Python environment") from exc

    target_dir = resolve_corpus_path(config, corpus_id, root_override)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / "part-00000.jsonl"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing export: {output_path} (pass --overwrite)")

    dataset_config = info.get("hf_config")
    kwargs: Dict[str, Any] = {"split": split, "streaming": True}
    if dataset_config:
        kwargs["name"] = dataset_config

    # Write to a temporary file first so an interrupted export is not mistaken for a complete corpus.
    temp_path = output_path.with_suffix(output_path.suffix + ".partial")
    rows_written = 0
    rows_skipped = 0
    try:
        stream = load_dataset(dataset_id, **kwargs)
        with temp_path.open("w", encoding="utf-8", newline="\n") as f:
            for row_index, row in enumerate(stream):
                normalized = normalize_row(row, corpus_id, info, row_index)
                if normalized is None:
                    rows_skipped += 1
                    continue
                f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                rows_written += 1
                if max_docs and rows_written >= max_docs:
                    break
        os.replace(temp_path, output_path)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    return {
        "corpus_id": corpus_id,
        "dataset_id": dataset_id,
        "split": split,
        "rows_written": rows_written,
        "rows_skipped": rows_skipped,
        "bounded": bool(max_docs),
        "output": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(CONFIG_FILE_PATH))
    parser.add_argument("--data-root", default=None, help="Corpus root containing one directory per corpus ID; defaults to HETMAN_DATA_ROOT or project data/raw")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit", help="List local corpus readiness without accessing the network")
    download = sub.add_parser("download", help="Stream one configured HF dataset into normalized JSONL")
    download.add_argument("--corpus", action="append", required=True, help="Configured corpus ID; repeat to export more than one")
    download.add_argument("--split", default="train")
    download.add_argument("--max-docs", type=int, default=10_000, help="Safety cap per corpus; use 0 only with --confirm-unbounded")
    download.add_argument("--confirm-download", action="store_true", help="Acknowledge network access and writing corpus files")
    download.add_argument("--confirm-unbounded", action="store_true", help="Acknowledge potentially very large, unbounded exports")
    download.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_pilot_config(str(args.config))
    if args.command == "audit":
        print_inventory(config, args.data_root)
        return 0

    if not args.confirm_download:
        raise SystemExit("Download not started. Re-run with --confirm-download after reviewing the configured dataset and output paths.")
    if args.max_docs < 0:
        raise SystemExit("--max-docs must be >= 0")
    if args.max_docs == 0 and not args.confirm_unbounded:
        raise SystemExit("Unbounded export requires --confirm-unbounded; otherwise choose a positive --max-docs cap.")
    unknown = set(args.corpus) - set(config.get("corpora_sources", {}))
    if unknown:
        raise SystemExit(f"Unknown corpus IDs: {', '.join(sorted(unknown))}")

    for corpus_id in args.corpus:
        result = export_hf_dataset(
            config, corpus_id, args.data_root, args.split, args.max_docs, args.overwrite
        )
        print(json.dumps(result, ensure_ascii=False))
    print("\nRun `python src/prepare_raw_corpora.py audit` to review local availability.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, KeyError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
