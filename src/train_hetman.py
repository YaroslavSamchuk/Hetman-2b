"""Single-node corpus-backed Hetman trainer with chunked stateful BPTT and safe resume."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F

try:
    from src.configuration_hetman import HetmanConfig
    from src.modeling_hetman import HetmanForCausalLM
    from src.dataset_builder import MMapIndexedDataset
    from src.provenance import canonical_sha256, sha256_file
except ImportError:  # direct invocation
    from configuration_hetman import HetmanConfig
    from modeling_hetman import HetmanForCausalLM
    from dataset_builder import MMapIndexedDataset
    from provenance import canonical_sha256, sha256_file


def _detach_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {key: _detach_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    return value


def _atomic_save(payload: Dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        torch.save(payload, temporary)
        with temporary.open("r+b") as f:
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _trim_metrics(path: Path, through_step: int) -> None:
    if not path.exists():
        return
    kept = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(record.get("step", 0)) <= through_step:
                kept.append(json.dumps(record, ensure_ascii=False) + "\n")
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as f:
            f.writelines(kept)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sequence_loss(
    model: HetmanForCausalLM,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    chunk_tokens: int,
    backward: bool,
    precision: str,
    commit_weight: float,
    ortho_weight: float,
) -> Dict[str, float]:
    """Run one sequence in bounded chunks, preserving forward state and detaching at chunk boundaries."""
    ids = token_ids.to(device=device, dtype=torch.long).reshape(1, -1)
    mask = attention_mask.to(device=device, dtype=torch.bool).reshape(1, -1)
    sequence_len = ids.shape[1]
    if sequence_len < 2:
        raise ValueError("Training sequences must contain at least two tokens")

    primary_total = int(mask[:, 1:].sum().item())
    mtp_total = int(mask[:, 2:].sum().item())
    input_total = int(mask.sum().item())
    if primary_total == 0 or input_total == 0:
        raise ValueError("Training sequence contains no valid next-token targets")

    state = None
    ce_sum = 0.0
    mtp_sum = 0.0
    commit_sum = 0.0
    ortho_sum = 0.0
    started = 0
    last_input_end = sequence_len - 1

    while started < last_input_end:
        end = min(started + chunk_tokens, last_input_end)
        inputs = ids[:, started:end]
        labels_end = min(sequence_len, end + 2)
        labels = ids[:, started:labels_end].clone()
        label_mask = mask[:, started:labels_end]
        labels.masked_fill_(~label_mask, -100)
        prefix_mask = mask[:, :end]
        valid_inputs = mask[:, started:end]
        valid_input_count = int(valid_inputs.sum().item())

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda" and precision == "bf16"),
        ):
            output = model(
                inputs,
                labels=labels,
                past_states=state,
                offset=started,
                attention_mask=prefix_mask,
                position_ids=torch.arange(started, end, device=device).unsqueeze(0),
                token_loss_mask=valid_inputs,
            )

        primary_count = min(inputs.shape[1], labels.shape[1] - 1)
        mtp_count = min(inputs.shape[1], max(0, labels.shape[1] - 2))
        chunk_primary = int((labels[:, 1:1 + primary_count] != -100).sum().item())
        chunk_mtp = int((labels[:, 2:2 + mtp_count] != -100).sum().item()) if mtp_count else 0
        ce_value = output.ce_loss if output.ce_loss is not None else output.loss.new_zeros(())
        mtp_value = output.mtp_loss if output.mtp_loss is not None else output.loss.new_zeros(())
        commit_value = output.hopfield_loss
        ortho_value = output.ortho_loss

        ce_sum += float(ce_value.detach().float().item()) * chunk_primary
        mtp_sum += float(mtp_value.detach().float().item()) * chunk_mtp
        commit_sum += float(commit_value.detach().float().item()) * valid_input_count
        ortho_sum += float(ortho_value.detach().float().item()) * valid_input_count

        if backward:
            loss = ce_value * (chunk_primary / primary_total)
            if mtp_total:
                loss = loss + 0.3 * mtp_value * (chunk_mtp / mtp_total)
            loss = loss + commit_weight * commit_value * (valid_input_count / input_total)
            loss = loss + ortho_weight * ortho_value * (valid_input_count / input_total)
            loss.backward()

        state = _detach_tree(output.past_states)
        started = end

    return {
        "ce_nll": ce_sum / primary_total,
        "ce_tokens": float(primary_total),
        "mtp_nll": mtp_sum / mtp_total if mtp_total else 0.0,
        "mtp_tokens": float(mtp_total),
        "hopfield_loss": commit_sum / input_total,
        "ortho_loss": ortho_sum / input_total,
        "input_tokens": float(input_total),
    }


def _model_config(args: argparse.Namespace) -> HetmanConfig:
    values: Dict[str, Any] = {}
    if args.model_config:
        with args.model_config.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        values = raw.get("architecture", raw)
        # Map the grant/config naming to the executable model's constructor.
        aliases = {
            "total_layers": "num_hidden_layers",
            "deltanet_layers": "num_deltanet_layers",
            "global_layers": "num_global_layers",
            "hopfield_groups": "hopfield_num_groups",
        }
        values = {aliases.get(key, key): value for key, value in values.items()}
        allowed = set(HetmanConfig.__dataclass_fields__)
        values = {key: value for key, value in values.items() if key in allowed}
    return HetmanConfig(**values)


def _validate_shard_token_ids(shard: MMapIndexedDataset, name: str, vocab_size: int) -> None:
    if shard.total_tokens == 0:
        raise ValueError(f"{name} shard contains no tokens")
    token_map = shard.bin_mmap
    minimum = int(token_map.min())
    maximum = int(token_map.max())
    if minimum < 0 or maximum >= vocab_size:
        raise ValueError(
            f"{name} shard token IDs range from {minimum} to {maximum}, outside model vocabulary 0..{vocab_size - 1}"
        )


def _run_signature(args: argparse.Namespace, config: HetmanConfig, train_base: Path, val_base: Path) -> str:
    payload: Dict[str, Any] = {
        "model_config": vars(config),
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "chunk_tokens": args.chunk_tokens,
        "precision": args.precision,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "train_shards": {},
        "val_shards": {},
    }
    for split, base in (("train_shards", train_base), ("val_shards", val_base)):
        for suffix in (".bin", ".idx"):
            path = base.with_suffix(suffix)
            payload[split][suffix] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    if args.tokenizer.is_file():
        payload["tokenizer_sha256"] = sha256_file(args.tokenizer)
    return canonical_sha256(payload)


def run(args: argparse.Namespace) -> None:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch installation has no available CUDA device")
    if args.device == "cuda" and args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA device does not support BF16; choose --precision fp32 or use a supported GPU")

    config = _model_config(args)
    if config.num_hidden_layers != 34 or config.num_deltanet_layers != 28 or config.num_global_layers != 6:
        raise ValueError("The implemented Hetman topology is fixed at 34 layers: 28 DeltaNet + 6 global attention")
    if config.global_attention_heads * config.global_head_dim != config.hidden_size:
        raise ValueError("global_attention_heads * global_head_dim must equal hidden_size")
    if config.deltanet_head_dim * config.num_attention_heads <= 0:
        raise ValueError("DeltaNet head dimensions must be positive")
    if config.hopfield_topk_1 > config.hopfield_codebook_size or config.hopfield_topk_2 > config.hopfield_codebook_size:
        raise ValueError("Hopfield top-k values cannot exceed the configured codebook size")
    if not args.tokenizer.is_file():
        raise FileNotFoundError(f"Tokenizer is missing: {args.tokenizer}")
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("Training requires the tokenizers package to validate vocabulary compatibility") from exc
    tokenizer_vocab = Tokenizer.from_file(str(args.tokenizer)).get_vocab_size(with_added_tokens=True)
    if tokenizer_vocab != config.vocab_size:
        raise ValueError(f"Tokenizer vocabulary ({tokenizer_vocab}) does not match model config ({config.vocab_size})")

    train_shard = MMapIndexedDataset(str(args.train_shard))
    val_shard = MMapIndexedDataset(str(args.val_shard)) if args.val_shard else None
    if len(train_shard) == 0:
        raise ValueError("Training shard is empty")
    if train_shard.sequence_length < 2:
        raise ValueError("Training shard sequence length must be at least two tokens")
    if val_shard is not None and val_shard.sequence_length != train_shard.sequence_length:
        raise ValueError("Training and validation shard sequence lengths differ")
    _validate_shard_token_ids(train_shard, "Training", config.vocab_size)
    if val_shard is not None:
        _validate_shard_token_ids(val_shard, "Validation", config.vocab_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    signature = _run_signature(args, config, Path(args.train_shard), Path(args.val_shard) if args.val_shard else Path(args.train_shard))
    model = HetmanForCausalLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    order = torch.randperm(len(train_shard)).tolist()
    position = 0
    completed_steps = 0
    epoch = 0
    stop_requested = False
    previous_handlers = {}
    checkpoints = sorted(
        (path for path in checkpoint_dir.glob("step_*.pt") if re.fullmatch(r"step_\d{8}\.pt", path.name)),
        key=lambda path: int(path.stem.split("_")[1]),
    )
    if (checkpoints or metrics_path.exists()) and not args.resume:
        raise FileExistsError("Training artifacts already exist; pass --resume for a compatible checkpoint or choose a new output directory")

    def save_checkpoint(step: int) -> Path:
        path = checkpoint_dir / f"step_{step:08d}.pt"
        _atomic_save({
            "step": step,
            "epoch": epoch,
            "order": order,
            "position": position,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "signature": signature,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            },
        }, path)
        existing = sorted(
            checkpoint_dir.glob("step_*.pt"), key=lambda item: int(item.stem.split("_")[1])
        )
        for old in existing[:-3]:
            old.unlink(missing_ok=True)
        return path

    if args.resume:
        if not checkpoints:
            raise FileNotFoundError("--resume requested but no trainer checkpoint exists")
        checkpoint = None
        errors = []
        for candidate in reversed(checkpoints):
            try:
                item = torch.load(candidate, map_location="cpu", weights_only=False)
                if item.get("signature") != signature:
                    raise ValueError("training config, tokenizer, or shard content differs")
                checkpoint = item
                break
            except Exception as exc:
                errors.append(f"{candidate.name}: {exc}")
        if checkpoint is None:
            raise RuntimeError("No readable compatible checkpoint: " + "; ".join(errors))
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        completed_steps = int(checkpoint["step"])
        epoch = int(checkpoint["epoch"])
        order = list(checkpoint["order"])
        position = int(checkpoint["position"])
        rng = checkpoint.get("rng", {})
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if device.type == "cuda" and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
        _trim_metrics(metrics_path, completed_steps)

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[sig] = signal.signal(sig, request_stop)
        except ValueError:
            pass

    if not args.resume:
        save_checkpoint(0)

    steps_per_epoch = len(train_shard)
    max_steps = args.max_steps or steps_per_epoch * args.epochs
    start_time = time.time()
    try:
        model.train()
        while completed_steps < max_steps:
            if position >= len(order):
                epoch += 1
                if epoch >= args.epochs:
                    break
                order = torch.randperm(len(train_shard)).tolist()
                position = 0
            sequence_index = order[position]
            position += 1
            item = train_shard[sequence_index]
            optimizer.zero_grad(set_to_none=True)
            losses = _sequence_loss(
                model,
                torch.from_numpy(np.asarray(item["input_ids"], dtype=np.int64)),
                torch.from_numpy(np.asarray(item["attention_mask"], dtype=np.bool_)),
                device,
                args.chunk_tokens,
                backward=True,
                precision=args.precision,
                commit_weight=args.commit_weight,
                ortho_weight=config.hopfield_ortho_weight,
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(f"Non-finite gradient norm at step {completed_steps + 1}; checkpoint remains at prior safe step")
            optimizer.step()
            completed_steps += 1
            record: Dict[str, Any] = {
                "step": completed_steps,
                "epoch": epoch,
                "sequence_index": sequence_index,
                "ce_nll": losses["ce_nll"],
                "ce_perplexity": math.exp(min(losses["ce_nll"], 80.0)),
                "mtp_nll": losses["mtp_nll"],
                "hopfield_loss": losses["hopfield_loss"],
                "ortho_loss": losses["ortho_loss"],
                "gradient_norm": float(grad_norm.detach().float().item()),
                "elapsed_seconds": time.time() - start_time,
            }
            should_eval = val_shard is not None and (completed_steps % args.eval_every == 0 or stop_requested)
            if should_eval:
                model.eval()
                eval_nll = 0.0
                eval_tokens = 0
                with torch.no_grad():
                    for val_index in range(min(len(val_shard), args.eval_sequences)):
                        val_item = val_shard[val_index]
                        val_loss = _sequence_loss(
                            model,
                            torch.from_numpy(np.asarray(val_item["input_ids"], dtype=np.int64)),
                            torch.from_numpy(np.asarray(val_item["attention_mask"], dtype=np.bool_)),
                            device,
                            args.chunk_tokens,
                            backward=False,
                            precision=args.precision,
                            commit_weight=args.commit_weight,
                            ortho_weight=config.hopfield_ortho_weight,
                        )
                        eval_nll += val_loss["ce_nll"] * val_loss["ce_tokens"]
                        eval_tokens += int(val_loss["ce_tokens"])
                record["validation"] = {
                    "tokens": eval_tokens,
                    "mean_nll": eval_nll / max(eval_tokens, 1),
                    "perplexity": math.exp(min(eval_nll / max(eval_tokens, 1), 80.0)),
                }
                model.train()

            if completed_steps % args.checkpoint_every == 0 or should_eval or stop_requested:
                record["checkpoint"] = str(save_checkpoint(completed_steps))
            with metrics_path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            print(
                f"step {completed_steps}/{max_steps} ce_nll={losses['ce_nll']:.4f} "
                f"ppl={math.exp(min(losses['ce_nll'], 80.0)):.2f} grad={float(grad_norm):.3f}"
            )
            if stop_requested:
                break
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        train_shard.close()
        if val_shard is not None:
            val_shard.close()

    if stop_requested:
        raise SystemExit(130)
    print(f"Training segment finished at step {completed_steps}; metrics: {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shard", type=Path, required=True, help="Shard base path without .bin/.idx")
    parser.add_argument("--val-shard", type=Path, default=None, help="Optional validation shard base path")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=None, help="Optional Hetman config JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None, help="Bounded diagnostic run; omit for configured epochs")
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.8e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--commit-weight", type=float, default=0.005)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-sequences", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0 or args.chunk_tokens <= 0 or args.eval_every <= 0 or args.eval_sequences <= 0 or args.checkpoint_every <= 0:
        parser.error("epochs, chunk-tokens, eval-every, eval-sequences, and checkpoint-every must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("max-steps must be positive")
    if args.learning_rate <= 0 or args.grad_clip <= 0 or args.commit_weight < 0 or args.weight_decay < 0:
        parser.error("learning rate and grad clip must be positive; loss weight and weight decay must be non-negative")
    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
