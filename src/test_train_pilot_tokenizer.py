"""Integration test for tokenizer vocabulary size, special IDs, and safe reuse."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from tokenizers import Tokenizer

from train_pilot_tokenizer import train


def test_train_and_reuse_tokenizer() -> None:
    with tempfile.TemporaryDirectory(prefix="hetman_pilot_tokenizer_") as temp:
        root = Path(temp)
        corpus_dir = root / "raw" / "testsource"
        corpus_dir.mkdir(parents=True)
        with (corpus_dir / "docs.jsonl").open("w", encoding="utf-8") as f:
            for doc_id in range(500):
                text = " ".join(f"word{doc_id}_{index}" for index in range(80))
                f.write(json.dumps({"id": f"doc-{doc_id}", "text": text}) + "\n")

        config = {
            "total_tokens_per_recipe": 65,
            "sequence_length": 16,
            "total_sequences_per_recipe": 4,
            "tokenizer_vocab_size": 512,
            "data_root_env_var": "HETMAN_DATA_ROOT",
            "validation_split": {"fraction_per_source": 0.01, "seed": 42},
            "corpora_sources": {
                "testsource": {
                    "name": "test corpus",
                    "local_path": "data/raw/testsource",
                    "doc_type": "general_web",
                    "license": "test fixture",
                    "text_column": "text",
                    "hf_dataset_id": None,
                }
            },
            "recipes": {
                "recipe_test": {
                    "target_tokens": 65,
                    "curation_applied": False,
                    "mixture": {"testsource": {"percentage": 100.0, "tokens": 65, "sequences_4k": 4}},
                }
            },
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        output = root / "processed" / "tokenizer"
        args = argparse.Namespace(config=config_path, data_root=root / "raw", output_dir=output)

        manifest = train(args)
        tok = Tokenizer.from_file(str(output / "tokenizer.json"))
        assert tok.get_vocab_size(with_added_tokens=True) == 512
        assert manifest["training_split_only"] is True
        assert manifest["training_documents_used"] > 0
        assert tok.token_to_id("<unk>") == 0
        assert tok.token_to_id("<s>") == 1
        assert tok.token_to_id("<|im_end|>") == 5

        # Re-running against the same corpus reuses the immutable verified artifact.
        assert train(args)["tokenizer_sha256"] == manifest["tokenizer_sha256"]


if __name__ == "__main__":
    test_train_and_reuse_tokenizer()
    print("[PASS] Pilot tokenizer training, vocabulary checks, and safe reuse verified.")
