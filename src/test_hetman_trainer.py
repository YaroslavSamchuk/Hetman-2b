"""CPU-sized checks for the full-model chunked training loss path."""

from __future__ import annotations

import argparse
import json
import signal
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from configuration_hetman import HetmanConfig
from dataset_builder import BinaryShardWriter
from modeling_hetman import HetmanForCausalLM
import train_hetman
from train_hetman import _sequence_loss


def tiny_config() -> HetmanConfig:
    return HetmanConfig(
        vocab_size=1114,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=34,
        num_deltanet_layers=28,
        num_global_layers=6,
        num_attention_heads=2,
        deltanet_head_dim=16,
        global_head_dim=16,
        global_attention_heads=2,
        global_kv_heads=1,
        hopfield_num_groups=2,
        hopfield_codebook_size=8,
        hopfield_val_dim=8,
        hopfield_expanded_dim=16,
        hopfield_topk_1=2,
        hopfield_topk_2=2,
        max_position_embeddings=32,
    )


def test_extended_labels_cover_chunk_boundaries() -> None:
    torch.manual_seed(7)
    model = HetmanForCausalLM(tiny_config()).eval()
    inputs = torch.tensor([[1, 10, 11, 12]])
    labels_with_lookahead = torch.tensor([[1, 10, 11, 12, 13, 14]])
    output = model(inputs, labels=labels_with_lookahead)
    expected_ce = F.cross_entropy(
        output.logits.reshape(-1, output.logits.shape[-1]),
        labels_with_lookahead[:, 1:5].reshape(-1),
    )
    assert torch.allclose(output.ce_loss, expected_ce, atol=1e-6, rtol=1e-5)
    assert output.mtp_loss is not None and torch.isfinite(output.mtp_loss)

    two_token_chunk = model(
        torch.tensor([[1, 10]]),
        labels=torch.tensor([[1, 10, 11, 12]]),
    )
    assert two_token_chunk.mtp_loss is not None and two_token_chunk.mtp_loss.item() > 0.0


def test_chunked_full_model_loss_masks_padding_and_backpropagates() -> None:
    torch.manual_seed(11)
    model = HetmanForCausalLM(tiny_config())
    token_ids = torch.tensor([1, 20, 21, 22, 23, 24, 3, 3])
    mask = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.bool)
    result = _sequence_loss(
        model,
        token_ids,
        mask,
        torch.device("cpu"),
        chunk_tokens=2,
        backward=True,
        precision="fp32",
        commit_weight=0.005,
        ortho_weight=model.config.hopfield_ortho_weight,
    )
    assert result["ce_tokens"] == 5
    assert result["mtp_tokens"] == 4
    assert all(torch.isfinite(torch.tensor(value)) for value in result.values())
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_training_checkpoint_resume_after_signal() -> None:
    project_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="hetman_full_train_test_") as temporary:
        root = Path(temporary)
        train_base = root / "train"
        val_base = root / "val"
        for base, split, rows in (
            (train_base, "train", [[1, 10, 11, 12, 13, 14, 15, 2], [1, 20, 21, 22, 23, 24, 25, 2]]),
            (val_base, "val", [[1, 30, 31, 32, 33, 34, 35, 2]]),
        ):
            writer = BinaryShardWriter(str(base), sequence_length=8)
            for row in rows:
                writer.add_sequence(row, source_id="test", split=split)
            writer.close()

        config_path = root / "model.json"
        config_path.write_text(json.dumps(asdict(tiny_config())), encoding="utf-8")
        args = argparse.Namespace(
            train_shard=train_base,
            val_shard=val_base,
            tokenizer=project_root / "tokenizer" / "tokenizer.json",
            model_config=config_path,
            output_dir=root / "run",
            device="cpu",
            precision="fp32",
            epochs=1,
            max_steps=2,
            chunk_tokens=3,
            learning_rate=1e-4,
            weight_decay=0.0,
            grad_clip=1.0,
            commit_weight=0.005,
            eval_every=1,
            eval_sequences=1,
            checkpoint_every=1,
            seed=17,
            resume=False,
        )

        original = train_hetman._sequence_loss
        sent = {"value": False}

        def interrupt_after_first_train_step(*call_args, **call_kwargs):
            result = original(*call_args, **call_kwargs)
            if call_kwargs.get("backward") and not sent["value"]:
                sent["value"] = True
                signal.raise_signal(signal.SIGTERM)
            return result

        train_hetman._sequence_loss = interrupt_after_first_train_step
        try:
            try:
                train_hetman.run(args)
            except SystemExit as exc:
                assert exc.code == 130
            else:
                raise AssertionError("Signal interruption did not stop the training segment")
        finally:
            train_hetman._sequence_loss = original

        assert (args.output_dir / "checkpoints" / "step_00000001.pt").is_file()
        args.resume = True
        train_hetman.run(args)
        assert (args.output_dir / "checkpoints" / "step_00000002.pt").is_file()
        records = [json.loads(line) for line in (args.output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [record["step"] for record in records] == [1, 2]


if __name__ == "__main__":
    test_extended_labels_cover_chunk_boundaries()
    test_chunked_full_model_loss_masks_padding_and_backpropagates()
    test_training_checkpoint_resume_after_signal()
    print("[PASS] Full-model chunked training, padding masks, gradients, signal checkpoint, and resume verified.")
