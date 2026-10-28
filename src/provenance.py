"""Content-based provenance helpers for reproducible datasets and tokenizers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SUPPORTED_CORPUS_SUFFIXES = (".jsonl", ".jsonl.gz", ".parquet", ".txt")


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _corpus_files(root: Path) -> List[Path]:
    if root.is_file():
        files = [root] if str(root).endswith(SUPPORTED_CORPUS_SUFFIXES) else []
    elif root.is_dir():
        files = []
        for current, dirs, names in os.walk(root):
            dirs.sort()
            for name in sorted(names):
                path = Path(current) / name
                if str(path).endswith(SUPPORTED_CORPUS_SUFFIXES):
                    files.append(path)
    else:
        files = []
    return sorted(files, key=lambda p: p.relative_to(root).as_posix() if root.is_dir() else p.name)


def corpus_content_manifest(
    config: Dict[str, Any],
    data_root: Optional[str],
    source_ids: Iterable[str],
) -> List[Dict[str, Any]]:
    """Return sorted corpus file hashes; fail if any required source is missing or changes mid-read."""
    env_root = os.getenv(config.get("data_root_env_var", "HETMAN_DATA_ROOT"))
    base_root = data_root or env_root
    manifest: List[Dict[str, Any]] = []
    sources = config.get("corpora_sources", {})
    for source_id in sorted(set(source_ids)):
        if source_id not in sources:
            raise KeyError(f"Unknown corpus source in provenance request: {source_id}")
        info = sources[source_id]
        local_path = info.get("local_path", "")
        if os.path.isabs(local_path):
            root = Path(local_path)
        elif base_root:
            root = Path(base_root) / source_id
        else:
            root = Path(__file__).resolve().parents[1] / local_path
        root = root.resolve()
        files = _corpus_files(root)
        if not files:
            raise FileNotFoundError(f"No supported corpus files found for {source_id}: {root}")
        for path in files:
            before = path.stat()
            digest = sha256_file(path)
            after = path.stat()
            if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
                raise RuntimeError(f"Corpus file changed while provenance was being computed: {path}")
            relative = path.relative_to(root).as_posix() if root.is_dir() else path.name
            manifest.append({
                "source_id": source_id,
                "relative_path": relative,
                "size_bytes": before.st_size,
                "sha256": digest,
            })
    return manifest
