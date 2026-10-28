"""Build validated, restart-safe A/B/C pilot shards from local licensed corpora."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

try:
    from src.dataset_builder import DatasetRecipeBuilder, MMapIndexedDataset
    from src.pilot_experiment_config import audit_corpus_paths, load_pilot_config, validate_experiment_config
    from src.provenance import canonical_sha256, corpus_content_manifest, sha256_file
except ImportError:  # direct invocation: python src/build_pilot_datasets.py
    from dataset_builder import DatasetRecipeBuilder, MMapIndexedDataset
    from pilot_experiment_config import audit_corpus_paths, load_pilot_config, validate_experiment_config
    from provenance import canonical_sha256, corpus_content_manifest, sha256_file


def _validate_completed(
    recipe_dir: Path,
    recipe_id: str,
    recipe: Dict[str, Any],
    seq_len: int,
    expected_provenance: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    stats_path = recipe_dir / f"{recipe_id}_stats.json"
    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    if expected_provenance is not None and stats.get("provenance") != expected_provenance:
        raise ValueError(
            f"{recipe_id}: existing shards were built from different corpora, tokenizer, or config. "
            "Preserve this output and choose a new output directory to rebuild."
        )
    train = MMapIndexedDataset(str(recipe_dir / f"{recipe_id}_train"))
    val = MMapIndexedDataset(str(recipe_dir / f"{recipe_id}_val"))
    if len(train) == 0 or len(val) == 0:
        raise ValueError(f"{recipe_id}: train or validation shard is empty")
    for name, shard, expected_split in (("train", train, 0), ("validation", val, 1)):
        base = recipe_dir / f"{recipe_id}_{'train' if name == 'train' else 'val'}"
        bin_size = Path(f"{base}.bin").stat().st_size
        if bin_size != shard.total_tokens * 4:
            raise ValueError(f"{recipe_id}: {name} binary size does not match indexed token total")
        offset = 0
        for row in shard.index_mmap:
            count = int(row["count"])
            if int(row["offset"]) != offset or count <= 0 or count > seq_len:
                raise ValueError(f"{recipe_id}: {name} shard has a malformed index record")
            if int(row["split"]) != expected_split:
                raise ValueError(f"{recipe_id}: {name} shard contains a record with the wrong split tag")
            offset += count * 4
        if offset != bin_size:
            raise ValueError(f"{recipe_id}: {name} index does not cover the complete token file")
    if train.total_tokens != int(recipe["target_tokens"]):
        raise ValueError(f"{recipe_id}: completed train shard has {train.total_tokens:,} tokens; expected {int(recipe['target_tokens']):,}")
    if train.sequence_length != seq_len or val.sequence_length != seq_len:
        raise ValueError(f"{recipe_id}: shard sequence length does not match config ({seq_len})")
    if len(val) == 0 or not stats.get("train_val_isolated"):
        raise ValueError(f"{recipe_id}: validation shard is empty or train/validation isolation was not verified")
    for name, shard in (("train", train), ("validation", val)):
        recorded = stats[f"{name if name == 'train' else 'val'}_shards"]["total_tokens"]
        if int(recorded) != shard.total_tokens:
            raise ValueError(f"{recipe_id}: stats token count disagrees with the {name} shard header")
    return stats


def build(args: argparse.Namespace) -> int:
    config = load_pilot_config(str(args.config))
    validate_experiment_config(config)
    selected = list(config["recipes"]) if args.recipe == "all" else [args.recipe]
    unknown = set(selected) - set(config["recipes"])
    if unknown:
        raise ValueError(f"Unknown recipe IDs: {', '.join(sorted(unknown))}")

    required_sources = sorted({
        source_id
        for recipe_id in selected
        for source_id in config["recipes"][recipe_id]["mixture"]
    })
    unreviewed = []
    for source_id in required_sources:
        source = config.get("corpora_sources", {}).get(source_id, {})
        review = source.get("license_review") or {}
        if (
            review.get("status") != "approved"
            or not isinstance(review.get("evidence"), str)
            or not review["evidence"].strip()
            or not isinstance(review.get("reviewer"), str)
            or not review["reviewer"].strip()
            or not isinstance(review.get("review_date"), str)
            or not review["review_date"].strip()
        ):
            unreviewed.append(source_id)
    if unreviewed:
        raise ValueError(
            "Dataset build blocked: source licenses have not been reviewed with evidence, reviewer, and date: "
            + ", ".join(unreviewed)
        )

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("Building real shards requires the `tokenizers` package") from exc
    if not args.tokenizer.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {args.tokenizer}")
    tokenizer_vocab = Tokenizer.from_file(str(args.tokenizer)).get_vocab_size(with_added_tokens=True)
    expected_vocab = config.get("tokenizer_vocab_size")
    if expected_vocab is not None and tokenizer_vocab != int(expected_vocab):
        raise ValueError(
            f"Tokenizer has {tokenizer_vocab:,} entries; experiment config requires {int(expected_vocab):,}. "
            "Train/validate the pilot tokenizer before building shards."
        )

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    paths = audit_corpus_paths(config, root_override=args.data_root)

    for recipe_id in selected:
        recipe = config["recipes"][recipe_id]
        provenance = {
            "config_sha256": canonical_sha256(config),
            "tokenizer_sha256": sha256_file(args.tokenizer),
            "corpora": corpus_content_manifest(config, str(args.data_root) if args.data_root else None, recipe["mixture"]),
        }
        destination = output_root / recipe_id
        if destination.exists():
            stats = _validate_completed(
                destination, recipe_id, recipe, int(config["sequence_length"]), provenance
            )
            print(f"[{recipe_id}] validated existing shards ({stats['train_shards']['total_tokens']:,} train tokens); skipping rebuild")
            continue

        missing: List[str] = []
        for source_id in recipe["mixture"]:
            entry = paths[source_id]
            if not entry["exists"] or entry["file_count"] == 0:
                missing.append(f"{source_id}: {entry['effective_path']}")
        if missing:
            raise FileNotFoundError("Required raw corpus data is missing:\n  " + "\n  ".join(missing))

        stage = output_root / f".{recipe_id}.building-{uuid.uuid4().hex}"
        try:
            print(f"[{recipe_id}] building from local corpora into temporary staging directory")
            builder = DatasetRecipeBuilder(str(args.config))
            stats = builder.build_recipe(
                recipe_id=recipe_id,
                output_dir=str(stage),
                data_root=args.data_root,
                tokenizer_path=str(args.tokenizer),
            )
            stats["provenance"] = provenance
            stats["manifest_file"] = str(destination / f"{recipe_id}_manifest.jsonl")
            for split in ("train_shards", "val_shards"):
                for key in ("bin_path", "idx_path"):
                    stats[split][key] = str(destination / Path(stats[split][key]).name)
            with (stage / f"{recipe_id}_stats.json").open("w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            _validate_completed(stage, recipe_id, recipe, int(config["sequence_length"]), provenance)
            stage.rename(destination)
            print(f"[{recipe_id}] verified and published shards: {destination}")
            print(f"[{recipe_id}] train tokens={stats['train_shards']['total_tokens']:,}; sequences={stats['train_shards']['total_sequences']:,}")
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", default="all", choices=("all", "recipe_a_baseline", "recipe_b_cleaned", "recipe_c_balanced"))
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "pilot_dataset_recipes.json")
    parser.add_argument("--data-root", type=Path, default=None, help="Corpus root, or set HETMAN_DATA_ROOT")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "processed")
    parser.add_argument("--tokenizer", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "processed" / "tokenizer" / "tokenizer.json")
    args = parser.parse_args()
    try:
        return build(args)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
