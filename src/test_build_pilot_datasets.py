"""Local integration test for staged, exact-token pilot shard building."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from build_pilot_datasets import build


def test_staged_builder_and_idempotent_validation() -> None:
    project = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="hetman_build_pilot_") as temp:
        root = Path(temp)
        data_root = root / "raw"
        source_dir = data_root / "testsource"
        source_dir.mkdir(parents=True)
        raw_path = source_dir / "docs.jsonl"
        with raw_path.open("w", encoding="utf-8") as f:
            for doc_id in range(500):
                words = " ".join(f"word{doc_id}_{index}" for index in range(80))
                f.write(json.dumps({"id": f"doc-{doc_id}", "text": words}) + "\n")

        config = {
            "total_tokens_per_recipe": 66,
            "sequence_length": 16,
            "total_sequences_per_recipe": 4,
            "data_root_env_var": "HETMAN_DATA_ROOT",
            "validation_split": {"fraction_per_source": 0.01, "seed": 42},
            "corpora_sources": {
                "testsource": {
                    "name": "test corpus",
                    "local_path": "data/raw/testsource",
                    "doc_type": "general_web",
                    "license": "test fixture",
                    "license_review": {
                        "status": "approved",
                        "evidence": "fixture://synthetic-test-data",
                        "reviewer": "automated test",
                        "review_date": "2026-01-01",
                    },
                    "text_column": "text",
                    "hf_dataset_id": None,
                }
            },
            "recipes": {
                "recipe_test": {
                    "target_tokens": 66,
                    "curation_applied": False,
                    "mixture": {"testsource": {"percentage": 100.0, "tokens": 66, "sequences_4k": 4}},
                }
            },
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        output = root / "processed"
        args = argparse.Namespace(
            recipe="recipe_test",
            config=config_path,
            data_root=data_root,
            output_dir=output,
            tokenizer=project / "tokenizer" / "tokenizer.json",
        )

        assert build(args) == 0

        recipe_dir = output / "recipe_test"
        stats = json.loads((recipe_dir / "recipe_test_stats.json").read_text(encoding="utf-8"))
        assert stats["train_shards"]["total_tokens"] == 66
        assert Path(stats["train_shards"]["bin_path"]).is_file()
        assert Path(stats["manifest_file"]).is_file()
        assert not list(output.glob(".recipe_test.building-*"))

        # A second call validates the complete bundle and skips rebuilding it.
        assert build(args) == 0

        # Same-size source edits must invalidate cached shards based on content.
        original = raw_path.read_bytes()
        changed = original.replace(b"word0_0", b"w0rd0_0", 1)
        assert len(changed) == len(original)
        raw_path.write_bytes(changed)
        try:
            build(args)
        except ValueError as exc:
            assert "different corpora" in str(exc)
        else:
            raise AssertionError("Cached shards were incorrectly reused after corpus content changed")


def test_real_sources_fail_closed_without_license_review() -> None:
    project = Path(__file__).resolve().parents[1]
    args = argparse.Namespace(
        recipe="recipe_a_baseline",
        config=project / "configs" / "pilot_dataset_recipes.json",
        data_root=None,
        output_dir=project / "data" / "must_not_be_created",
        tokenizer=project / "tokenizer" / "tokenizer.json",
    )
    try:
        build(args)
    except ValueError as exc:
        assert "license" in str(exc).lower() and "unverified" not in str(exc).lower()
    else:
        raise AssertionError("Real dataset sources must be blocked until their license reviews are documented")


if __name__ == "__main__":
    test_real_sources_fail_closed_without_license_review()
    test_staged_builder_and_idempotent_validation()
    print("[PASS] Staged dataset build, validation, and idempotent rerun verified.")
