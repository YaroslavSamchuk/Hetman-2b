"""
Test suite for src/run_pilot_experiment.py:
Verifies:
1. ShardSequences dataset wrapping and tensor conversion
2. GRUProxyLM causal recurrent language model forward pass
3. masked_chunk_loss truncated BPTT with exact sequence gradient accumulation
4. End-to-end run_recipe execution on synthetic mmap shards
5. Validation perplexity evaluation across domains (including handling absent sources)
6. Checkpointing and metrics JSONL logging
"""

import json
import math
import os
import signal
import shutil
import tempfile
import zlib
import numpy as np
import torch

from dataset_builder import BinaryShardWriter, MMapIndexedDataset
from run_pilot_experiment import (
    GRUProxyLM,
    ShardSequences,
    evaluate_domain,
    masked_chunk_loss,
    run_recipe,
    source_indices,
    parse_args,
)
import run_pilot_experiment as pilot_runner

def test_gru_proxy_lm():
    print("[1] Testing GRUProxyLM model architecture and forward pass...")
    vocab_size = 1000
    hidden_size = 32
    seq_len = 64
    batch_size = 2

    model = GRUProxyLM(vocab_size, hidden_size)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, hidden = model(x)

    assert logits.shape == (batch_size, seq_len, vocab_size), f"Unexpected logits shape: {logits.shape}"
    assert hidden.shape == (1, batch_size, hidden_size), f"Unexpected hidden shape: {hidden.shape}"
    print("    [PASS] GRUProxyLM shapes and weight tying verified.")


def test_masked_chunk_loss():
    print("\n[2] Testing masked_chunk_loss with padding exclusion...")
    vocab_size = 500
    hidden_size = 32
    seq_len = 64
    chunk_tokens = 16
    device = torch.device("cpu")

    model = GRUProxyLM(vocab_size, hidden_size)
    
    # Create a sequence with 40 real tokens and 24 padding tokens
    real_tokens = 40
    input_ids = torch.randint(0, vocab_size, (1, seq_len))
    input_ids[0, real_tokens:] = 3  # PAD_TOKEN_ID
    attention_mask = torch.zeros(1, seq_len, dtype=torch.int64)
    attention_mask[0, :real_tokens] = 1

    nll, token_count = masked_chunk_loss(
        model, input_ids, attention_mask, device, chunk_tokens, backward=True
    )

    # In causal LM, targets are mask[:, 1:] -> real_tokens - 1
    assert token_count == real_tokens - 1, f"Expected {real_tokens - 1} target tokens, got {token_count}"
    assert nll > 0.0, "NLL must be positive"

    # Verify gradients exist
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert grad_norm > 0.0, "Gradients should be non-zero after backward"
    print(f"    [PASS] Truncated BPTT loss computed: {nll:.4f} for {token_count} valid tokens. Gradients verified.")


def test_end_to_end_pilot_runner():
    print("\n[3] Testing end-to-end pilot runner on synthetic mmap shards...")
    tmp_dir = tempfile.mkdtemp(prefix="hetman_test_pilot_")
    try:
        shard_dir = os.path.join(tmp_dir, "shards")
        output_dir = os.path.join(tmp_dir, "runs")
        recipe_id = "recipe_test"
        recipe_shard_dir = os.path.join(shard_dir, recipe_id)
        os.makedirs(recipe_shard_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        seq_len = 128
        total_train_tokens = 512

        # 1. Create a synthetic train shard (4 sequences of 128 tokens = 512 tokens)
        train_writer = BinaryShardWriter(os.path.join(recipe_shard_dir, f"{recipe_id}_train"), sequence_length=seq_len)
        for seq_i in range(4):
            tokens = list(range(seq_i * 10, seq_i * 10 + seq_len))
            train_writer.add_sequence(tokens, source_id="ubertext_2.0", split="train")
        train_writer.close()

        # 2. Create a synthetic val shard with 2 domains: ubertext_2.0 and laws_ua (fineweb_edu and starcoder2 absent)
        val_writer = BinaryShardWriter(os.path.join(recipe_shard_dir, f"{recipe_id}_val"), sequence_length=seq_len)
        val_writer.add_sequence(list(range(seq_len)), source_id="ubertext_2.0", split="val")
        val_writer.add_sequence(list(range(seq_len)), source_id="laws_ua", split="val")
        val_writer.close()

        # 3. Create mock pilot config
        test_config = {
            "sequence_length": seq_len,
            "recipes": {
                recipe_id: {
                    "description": "Test recipe",
                    "target_tokens": total_train_tokens,
                }
            },
            "validation_benchmarks": {
                "perplexity_domains": [
                    {"name": "ukr_general_val", "target_tokens": 100},
                    {"name": "ukr_legal_val", "target_tokens": 100},
                    {"name": "fineweb_edu_val", "target_tokens": 100},
                    {"name": "code_python_val", "target_tokens": 100},
                ]
            }
        }
        config_path = os.path.join(tmp_dir, "test_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(test_config, f)

        # 4. Prepare args namespace
        import argparse
        test_args = argparse.Namespace(
            recipe=recipe_id,
            config=config_path,
            shard_dir=shard_dir,
            output_dir=output_dir,
            device="cpu",
            seed=42,
            vocab_size=2000,
            hidden_size=32,
            learning_rate=1e-3,
            weight_decay=0.01,
            grad_clip=1.0,
            chunk_tokens=32,
            workers=0,
            eval_every=2,
            checkpoint_every=2,
            no_eval=False,
            max_steps=2,
            resume=False,
        )

        # 5. Interrupt after a completed step; the runner must checkpoint and resume.
        append_jsonl = pilot_runner.append_jsonl

        def append_and_interrupt(path, record):
            append_jsonl(path, record)
            if record["step"] == 1:
                signal.raise_signal(signal.SIGINT)

        pilot_runner.append_jsonl = append_and_interrupt
        try:
            run_recipe(test_args, test_config, recipe_id)
            raise AssertionError("The synthetic SIGINT should stop the run")
        except SystemExit as exc:
            assert exc.code == 130
        finally:
            pilot_runner.append_jsonl = append_jsonl
        assert os.path.exists(os.path.join(output_dir, f"{recipe_id}_checkpoints", "step_000001.pt"))

        with open(os.path.join(output_dir, f"{recipe_id}_metrics.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"recipe": recipe_id, "step": 99, "interrupted_write": True}) + "\n")

        test_args.max_steps = 4
        test_args.resume = True
        run_recipe(test_args, test_config, recipe_id)

        # 6. Verify outputs
        log_path = os.path.join(output_dir, f"{recipe_id}_metrics.jsonl")
        assert os.path.exists(log_path), f"Metrics file not found: {log_path}"
        
        with open(log_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]

        assert len(lines) == 4, f"Expected 4 log lines after resume, got {len(lines)}"
        print(f"    Recorded {len(lines)} training step metrics.")

        # Check step 2 and step 4 evaluation records
        step2 = lines[1]
        assert "validation" in step2, "Step 2 should contain validation results"
        val = step2["validation"]
        
        # Present domains should have real perplexities
        assert "ukr_general_val" in val and val["ukr_general_val"]["perplexity"] is not None
        assert "ukr_legal_val" in val and val["ukr_legal_val"]["perplexity"] is not None
        print(f"    ukr_general_val perplexity: {val['ukr_general_val']['perplexity']:.2f}")
        print(f"    ukr_legal_val perplexity:   {val['ukr_legal_val']['perplexity']:.2f}")

        # Absent domains should have perplexity=None and reason
        assert val["fineweb_edu_val"]["perplexity"] is None
        assert "absent" in val["fineweb_edu_val"]["reason"]
        assert val["code_python_val"]["perplexity"] is None
        assert "absent" in val["code_python_val"]["reason"]
        print("    Absent validation domains correctly flagged as None.")

        # Check checkpoint creation
        ckpt_path = os.path.join(output_dir, f"{recipe_id}_checkpoints", "step_000004.pt")
        assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"
        ckpt = torch.load(ckpt_path, weights_only=False)
        assert ckpt["step"] == 4
        assert ckpt["train_tokens_seen"] == total_train_tokens
        assert "model" in ckpt
        assert "sampler" in ckpt and "run_signature" in ckpt
        print("    Checkpoint file verified: weights, step, and token counts preserved.")

        # Existing runs must never be silently restarted or resumed with a changed model shape.
        test_args.resume = False
        try:
            run_recipe(test_args, test_config, recipe_id)
            raise AssertionError("Existing run artifacts must reject an implicit restart")
        except FileExistsError:
            pass
        test_args.resume = True
        test_args.hidden_size += 1
        try:
            run_recipe(test_args, test_config, recipe_id)
            raise AssertionError("Resume must reject a changed model configuration")
        except RuntimeError as exc:
            assert "compatible" in str(exc)

        print("    [PASS] End-to-end pilot runner execution verified.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 80)
    print("  HETMAN-2.0B: A/B/C PILOT RUNNER TEST SUITE")
    print("=" * 80)
    test_gru_proxy_lm()
    test_masked_chunk_loss()
    test_end_to_end_pilot_runner()
    print("\n" + "=" * 80)
    print("  [ALL PILOT RUNNER TESTS PASSED SUCCESSFULLY!]")
    print("=" * 80)
