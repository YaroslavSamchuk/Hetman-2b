"""Train and atomically publish the pilot's vocabulary from training-split corpus text."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tokenizer"))

try:
    from train_tokenizer import SPECIAL_TOKENS, build_hetman_bpe_tokenizer, save_tokenizer_artifacts
    from src.dataset_builder import DatasetRecipeBuilder, DocumentSplitter
    from src.pilot_experiment_config import load_pilot_config, validate_experiment_config
    from src.provenance import canonical_sha256, corpus_content_manifest
except ImportError:  # direct invocation: python src/train_pilot_tokenizer.py
    from train_tokenizer import SPECIAL_TOKENS, build_hetman_bpe_tokenizer, save_tokenizer_artifacts
    from dataset_builder import DatasetRecipeBuilder, DocumentSplitter
    from pilot_experiment_config import load_pilot_config, validate_experiment_config
    from provenance import canonical_sha256, corpus_content_manifest
from tokenizers import pre_tokenizers, trainers


def source_signature(config: Dict[str, Any], data_root: Path | None) -> str:
    source_ids = sorted({source_id for recipe in config["recipes"].values() for source_id in recipe["mixture"]})
    manifest = corpus_content_manifest(config, str(data_root) if data_root else None, source_ids)
    return canonical_sha256({"corpora": manifest, "validation_split": config.get("validation_split", {})})


def _load_valid_existing(destination: Path, target_vocab: int, signature: str) -> Dict[str, Any] | None:
    manifest_path = destination / "tokenizer_manifest.json"
    tokenizer_path = destination / "tokenizer.json"
    if not destination.exists():
        return None
    if not manifest_path.is_file() or not tokenizer_path.is_file():
        raise FileExistsError(f"Incomplete tokenizer output exists at {destination}; preserve it and select a new output path")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    special_ids_match = all(tokenizer.token_to_id(token) == index for index, token in enumerate(SPECIAL_TOKENS))
    if (
        manifest.get("source_signature") != signature
        or manifest.get("vocab_size") != target_vocab
        or manifest.get("tokenizer_sha256") != digest
        or tokenizer.get_vocab_size(with_added_tokens=True) != target_vocab
        or not special_ids_match
    ):
        raise FileExistsError(
            f"Existing tokenizer at {destination} does not match current corpora/config. "
            "It was not overwritten; select a new output directory to rebuild."
        )
    return manifest


def train(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_pilot_config(str(args.config))
    validate_experiment_config(config)
    target_vocab = int(config.get("tokenizer_vocab_size", 65536))
    signature = source_signature(config, args.data_root)
    destination = args.output_dir.resolve()
    existing = _load_valid_existing(destination, target_vocab, signature)
    if existing:
        print(f"Tokenizer validated at {destination} (vocabulary={target_vocab:,}); skipping retraining")
        return existing

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.name}.building-{uuid.uuid4().hex}"
    try:
        tokenizer = build_hetman_bpe_tokenizer(vocab_size=target_vocab)
        trainer = trainers.BpeTrainer(
            vocab_size=target_vocab,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        builder = DatasetRecipeBuilder(str(args.config))
        split_cfg = config.get("validation_split", {})
        splitter = DocumentSplitter(
            fraction_val=float(split_cfg.get("fraction_per_source", 0.01)),
            seed=int(split_cfg.get("seed", 42)),
        )
        source_ids = sorted({source_id for recipe in config["recipes"].values() for source_id in recipe["mixture"]})
        counts = {"documents_seen": 0, "training_documents_used": 0}

        def training_texts() -> Iterator[str]:
            for source_id in source_ids:
                iterator = builder.get_corpus_iterator(source_id, data_root=str(args.data_root) if args.data_root else None)
                try:
                    for row in iterator:
                        counts["documents_seen"] += 1
                        if splitter.assign_split(source_id, str(row["id"])) != "train":
                            continue
                        text = row["text"].strip()
                        if text:
                            counts["training_documents_used"] += 1
                            yield text
                finally:
                    if hasattr(iterator, "close"):
                        iterator.close()

        tokenizer.train_from_iterator(training_texts(), trainer=trainer)
        actual_vocab = tokenizer.get_vocab_size(with_added_tokens=True)
        if actual_vocab != target_vocab:
            raise ValueError(
                f"Tokenizer training produced {actual_vocab:,} entries; configured target is {target_vocab:,}. "
                "Check corpus volume and vocabulary settings before starting training."
            )
        for expected_id, token in enumerate(SPECIAL_TOKENS):
            actual_id = tokenizer.token_to_id(token)
            if actual_id != expected_id:
                raise ValueError(f"Special-token ID mismatch for {token}: expected {expected_id}, got {actual_id}")

        save_tokenizer_artifacts(tokenizer, str(stage), target_vocab)
        tokenizer_path = stage / "tokenizer.json"
        digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        manifest = {
            "vocab_size": actual_vocab,
            "source_signature": signature,
            "training_split_only": True,
            "split_seed": int(split_cfg.get("seed", 42)),
            "validation_fraction": float(split_cfg.get("fraction_per_source", 0.01)),
            "source_ids": source_ids,
            "documents_seen": counts["documents_seen"],
            "training_documents_used": counts["training_documents_used"],
            "tokenizer_sha256": digest,
        }
        (stage / "tokenizer_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(destination)
        print(f"Tokenizer trained and validated: {destination} (vocabulary={actual_vocab:,})")
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "pilot_dataset_recipes.json")
    parser.add_argument("--data-root", type=Path, default=None, help="Corpus root, or set HETMAN_DATA_ROOT")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "tokenizer")
    args = parser.parse_args()
    try:
        train(args)
        return 0
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
