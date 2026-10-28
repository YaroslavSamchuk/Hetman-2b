"""Run the A/B/C pilot with a small PyTorch GRU proxy on mmap shards.

This runner compares data recipes. It is not the Hetman model or a substitute for
the TPU training engine. Each optimizer step consumes one packed training
sequence; the final partial sequence is masked and counts only its real tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import signal
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import zlib

try:
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset, Sampler
except ImportError as exc:  # pragma: no cover - depends on runtime environment
    raise SystemExit(
        "The pilot runner requires NumPy and PyTorch. Install them in the training environment."
    ) from exc

try:
    from src.dataset_builder import MMapIndexedDataset
    from src.provenance import sha256_file
except ImportError:  # direct invocation: python src/run_pilot_experiment.py
    from dataset_builder import MMapIndexedDataset
    from provenance import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pilot_dataset_recipes.json"
VAL_DOMAIN_SOURCES = {
    "ukr_general_val": "ubertext_2.0",
    "ukr_legal_val": "laws_ua",
    "fineweb_edu_val": "fineweb_edu",
    "code_python_val": "starcoder2_code",
}


class ShardSequences(Dataset):
    def __init__(self, shard: MMapIndexedDataset, indices: Optional[List[int]] = None):
        self.shard = shard
        self.indices = indices if indices is not None else list(range(len(shard)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.shard[self.indices[index]]
        return {
            "input_ids": torch.from_numpy(np.asarray(item["input_ids"], dtype=np.int64)),
            "attention_mask": torch.from_numpy(np.asarray(item["attention_mask"], dtype=np.int64)),
            "token_count": int(item["token_count"]),
        }


class StatefulRandomSampler(Sampler[int]):
    """Random sampler whose exact position can be saved at a checkpoint."""

    def __init__(self, data_source: Dataset, seed: int):
        self.size = len(data_source)
        self.generator = torch.Generator().manual_seed(seed)
        self.permutation: Optional[List[int]] = None
        self.position = 0

    def __iter__(self):
        while self.position < self.size:
            if self.permutation is None:
                self.permutation = torch.randperm(self.size, generator=self.generator).tolist()
            index = self.permutation[self.position]
            self.position += 1
            yield index
        self.permutation = None
        self.position = 0

    def __len__(self) -> int:
        return self.size

    def state_dict(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "generator_state": self.generator.get_state(),
            "permutation": self.permutation,
            "position": self.position,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if int(state.get("size", -1)) != self.size:
            raise ValueError("Checkpoint sampler size does not match the training shard")
        permutation = state.get("permutation")
        position = int(state.get("position", 0))
        if permutation is not None and (len(permutation) != self.size or not 0 <= position <= self.size):
            raise ValueError("Checkpoint contains an invalid sampler permutation/position")
        self.generator.set_state(state["generator_state"])
        self.permutation = permutation
        self.position = position


def _run_signature(
    args: argparse.Namespace,
    recipe: Dict[str, Any],
    config: Dict[str, Any],
    train_base: Path,
    val_base: Path,
) -> str:
    shard_stats = {}
    for split, base in (("train", train_base), ("validation", val_base)):
        shard_stats[split] = {}
        for suffix in (".bin", ".idx"):
            shard_path = base.parent / f"{base.name}{suffix}"
            stat = shard_path.stat()
            shard_stats[split][suffix] = {"size": stat.st_size, "sha256": sha256_file(shard_path)}
    payload = {
        "recipe": recipe,
        "sequence_length": config["sequence_length"],
        "seed": args.seed,
        "vocab_size": args.vocab_size,
        "hidden_size": args.hidden_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "chunk_tokens": args.chunk_tokens,
        "validation_benchmarks": config.get("validation_benchmarks", {}),
        "shards": shard_stats,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        torch.save(payload, temporary)
        with temporary.open("r+b") as f:
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _trim_metrics(path: Path, through_step: int) -> None:
    if not path.exists():
        return
    retained = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(record.get("step", 0)) <= through_step:
                retained.append(json.dumps(record, ensure_ascii=False) + "\n")
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as f:
            f.writelines(retained)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class GRUProxyLM(nn.Module):
    """Small causal recurrent LM for comparing data recipes at pilot scale."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=3)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        states, hidden = self.rnn(self.embedding(input_ids), hidden)
        return self.lm_head(states), hidden


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def source_indices(shard: MMapIndexedDataset, source_id: str) -> List[int]:
    source_crc = zlib.crc32(source_id.encode("utf-8"))
    return [
        i for i in range(len(shard))
        if int(shard.index_mmap[i]["source_crc"]) == source_crc
        and int(shard.index_mmap[i]["split"]) == 1
    ]


def masked_chunk_loss(
    model: GRUProxyLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    chunk_tokens: int,
    backward: bool,
) -> Tuple[float, int]:
    """Compute exact-vocabulary CE in bounded chunks with truncated BPTT."""
    ids = input_ids.to(device, non_blocking=True)
    mask = attention_mask.to(device, non_blocking=True)
    hidden = None
    token_total = int(mask[:, 1:].sum().item())
    if token_total == 0:
        return 0.0, 0

    nll_total = 0.0
    for start in range(0, ids.shape[1] - 1, chunk_tokens):
        end = min(start + chunk_tokens, ids.shape[1] - 1)
        x = ids[:, start:end]
        y = ids[:, start + 1:end + 1]
        valid = mask[:, start + 1:end + 1]
        logits, hidden = model(x, hidden)
        hidden = hidden.detach()
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = y.reshape(-1)
        flat_valid = valid.reshape(-1).bool()
        if flat_valid.any():
            loss_sum = F.cross_entropy(flat_logits[flat_valid], flat_targets[flat_valid], reduction="sum")
            nll_total += float(loss_sum.detach().item())
            if backward:
                (loss_sum / token_total).backward()
    return nll_total, token_total


@torch.no_grad()
def evaluate_domain(
    model: GRUProxyLM,
    dataset: Dataset,
    device: torch.device,
    max_tokens: int,
    chunk_tokens: int,
) -> Dict[str, Any]:
    model.eval()
    nll = 0.0
    tokens = 0
    for i in range(len(dataset)):
        item = dataset[i]
        item_tokens = int(item["token_count"])
        if tokens and tokens + item_tokens > max_tokens:
            break
        seq_nll, seq_tokens = masked_chunk_loss(
            model, item["input_ids"].unsqueeze(0), item["attention_mask"].unsqueeze(0),
            device, chunk_tokens, backward=False,
        )
        nll += seq_nll
        tokens += seq_tokens
        if tokens >= max_tokens:
            break
    if tokens == 0:
        return {"tokens": 0, "perplexity": None, "reason": "no validation target tokens"}
    mean_nll = nll / tokens
    return {"tokens": tokens, "mean_nll": mean_nll, "perplexity": math.exp(min(mean_nll, 80.0))}


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_recipe(args: argparse.Namespace, config: Dict[str, Any], recipe_id: str) -> None:
    recipe = config["recipes"][recipe_id]
    output_dir = Path(args.output_dir).resolve()
    shard_dir = Path(args.shard_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    recipe_shard_dir = shard_dir / recipe_id
    if (recipe_shard_dir / f"{recipe_id}_train.bin").is_file():
        shard_dir = recipe_shard_dir
    train_base = shard_dir / f"{recipe_id}_train"
    val_base = shard_dir / f"{recipe_id}_val"
    train_shard = MMapIndexedDataset(str(train_base))
    val_shard = MMapIndexedDataset(str(val_base))

    expected_tokens = int(recipe["target_tokens"])
    if train_shard.total_tokens != expected_tokens:
        raise ValueError(
            f"{recipe_id}: train shard has {train_shard.total_tokens:,} tokens; "
            f"expected exactly {expected_tokens:,}"
        )
    if train_shard.sequence_length != int(config["sequence_length"]):
        raise ValueError(f"{recipe_id}: train shard sequence length does not match pilot config")
    if val_shard.sequence_length != int(config["sequence_length"]):
        raise ValueError(f"{recipe_id}: validation shard sequence length does not match pilot config")
    if len(train_shard) == 0:
        raise ValueError(f"{recipe_id}: training shard is empty")

    full_steps = len(train_shard)
    step_limit = min(full_steps, args.max_steps) if args.max_steps else full_steps
    if step_limit <= 0:
        raise ValueError(f"{recipe_id}: no training steps were selected")
    if args.resume and args.workers != 0:
        raise ValueError("Exact checkpoint resume currently requires --workers 0")

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch build has no available CUDA device")

    dataset = ShardSequences(train_shard)
    sampler = StatefulRandomSampler(dataset, seed)
    model = GRUProxyLM(args.vocab_size, args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    log_path = output_dir / f"{recipe_id}_metrics.jsonl"
    checkpoint_dir = output_dir / f"{recipe_id}_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    signature = _run_signature(args, recipe, config, train_base, val_base)
    existing_checkpoints = sorted(
        (p for p in checkpoint_dir.glob("step_*.pt") if re.fullmatch(r"step_\d{6,}\.pt", p.name)),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if (existing_checkpoints or log_path.exists()) and not args.resume:
        raise FileExistsError(
            f"Existing run artifacts found for {recipe_id}. Use --resume to continue a compatible run, "
            "or choose a new --output-dir; existing files are never overwritten implicitly."
        )

    start_step = 1
    seen_tokens = 0
    if args.resume:
        if not existing_checkpoints:
            raise FileNotFoundError(f"--resume requested, but no checkpoints exist in {checkpoint_dir}")
        checkpoint = None
        checkpoint_path = None
        errors = []
        for candidate in reversed(existing_checkpoints):
            try:
                candidate_data = torch.load(candidate, map_location="cpu", weights_only=False)
                if candidate_data.get("run_signature") != signature:
                    raise ValueError("run configuration or training shard does not match")
                checkpoint, checkpoint_path = candidate_data, candidate
                break
            except Exception as exc:
                errors.append(f"{candidate.name}: {exc}")
        if checkpoint is None:
            raise RuntimeError("No compatible, readable checkpoint found: " + "; ".join(errors))
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        sampler.load_state_dict(checkpoint["sampler"])
        start_step = int(checkpoint["step"]) + 1
        seen_tokens = int(checkpoint["train_tokens_seen"])
        rng = checkpoint.get("rng", {})
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if device.type == "cuda" and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
        _trim_metrics(log_path, int(checkpoint["step"]))
        print(f"[{recipe_id}] resumed at step {start_step:,} from {checkpoint_path.name}")
    elif log_path.exists():
        raise FileExistsError(f"Metrics already exist at {log_path}; choose --resume or a new output directory")

    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=args.workers)
    iterator = iter(loader)
    domain_datasets: Dict[str, Dataset] = {}
    eval_tokens_by_domain = {
        item["name"]: int(item.get("target_tokens", 500_000))
        for item in config.get("validation_benchmarks", {}).get("perplexity_domains", [])
    }
    for domain_name, corpus_id in VAL_DOMAIN_SOURCES.items():
        if domain_name not in eval_tokens_by_domain:
            continue
        indices = source_indices(val_shard, corpus_id)
        if indices:
            domain_datasets[domain_name] = ShardSequences(val_shard, indices)

    print(f"[{recipe_id}] device={device}, train_tokens={train_shard.total_tokens:,}, steps={step_limit:,}")
    start_time = time.time()
    last_step = start_step - 1
    stop_requested = False
    previous_handlers = {}

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[sig] = signal.signal(sig, request_stop)
        except ValueError:  # signal handlers can only be installed from the main thread
            pass

    def save_checkpoint(step: int) -> Path:
        path = checkpoint_dir / f"step_{step:06d}.pt"
        _atomic_torch_save({
            "recipe": recipe_id,
            "step": step,
            "train_tokens_seen": seen_tokens,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "sampler": sampler.state_dict(),
            "run_signature": signature,
            "seed": seed,
            "proxy_model": {"type": "GRUProxyLM", "vocab_size": args.vocab_size, "hidden_size": args.hidden_size},
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            },
        }, path)
        saved = sorted(
            (p for p in checkpoint_dir.glob("step_*.pt") if re.fullmatch(r"step_\d{6,}\.pt", p.name)),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        for old in saved[:-3]:
            old.unlink(missing_ok=True)
        return path

    if not args.resume:
        save_checkpoint(0)

    try:
        for step in range(start_step, step_limit + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            nll, valid_tokens = masked_chunk_loss(
                model,
                batch["input_ids"],
                batch["attention_mask"],
                device,
                args.chunk_tokens,
                backward=True,
            )
            if valid_tokens == 0:
                raise RuntimeError(f"{recipe_id}: sequence at step {step} contains no next-token targets")
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            seen_tokens += int(batch["token_count"].item())
            last_step = step

            record: Dict[str, Any] = {
                "recipe": recipe_id,
                "step": step,
                "train_tokens_seen": seen_tokens,
                "train_nll": nll / valid_tokens,
                "train_perplexity": math.exp(min(nll / valid_tokens, 80.0)),
                "elapsed_seconds": time.time() - start_time,
            }
            should_evaluate = not args.no_eval and (step % args.eval_every == 0 or step == step_limit)
            if should_evaluate:
                record["validation"] = {
                    name: evaluate_domain(model, ds, device, eval_tokens_by_domain[name], args.chunk_tokens)
                    for name, ds in domain_datasets.items()
                }
                for missing in set(eval_tokens_by_domain) - set(domain_datasets):
                    record["validation"][missing] = {"tokens": 0, "perplexity": None, "reason": "source absent from this recipe's validation shard"}
            if step % args.checkpoint_every == 0 or step == step_limit or stop_requested:
                record["checkpoint"] = str(save_checkpoint(step))
            append_jsonl(log_path, record)
            if should_evaluate:
                print(f"[{recipe_id}] step {step:,}/{step_limit:,} loss={record['train_nll']:.4f}")
            if stop_requested:
                if "checkpoint" not in record:
                    save_checkpoint(step)
                print(f"[{recipe_id}] stop requested; checkpoint saved after completed step {step:,}.")
                break
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

    if stop_requested:
        raise SystemExit(130)

    print(f"[{recipe_id}] complete: {seen_tokens:,} training tokens observed; metrics={log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=("all", "recipe_a_baseline", "recipe_b_cleaned", "recipe_c_balanced"), default="all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--shard-dir", default=str(PROJECT_ROOT / "data" / "processed"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "runs" / "pilot_abc"))
    parser.add_argument("--device", default=None, help="PyTorch device, e.g. cpu or cuda; defaults to CUDA when available")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=None, help="Defaults to the validated tokenizer vocabulary size")
    parser.add_argument("--tokenizer", type=Path, default=PROJECT_ROOT / "data" / "processed" / "tokenizer" / "tokenizer.json")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--chunk-tokens", type=int, default=128, help="Bounded output projection chunk size; hidden state is detached between chunks")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume", action="store_true", help="Resume the latest checkpoint after validating run and shard compatibility")
    parser.add_argument("--no-eval", action="store_true", help="Skip validation; intended for the launcher's one-step hardware acceptance check")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional short run; default consumes every train sequence once")
    args = parser.parse_args()
    if args.chunk_tokens <= 0 or args.eval_every <= 0 or args.checkpoint_every <= 0 or args.hidden_size <= 0:
        parser.error("chunk-tokens, eval-every, checkpoint-every, and hidden-size must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("max-steps must be positive")
    return args


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    expected_vocab = config.get("tokenizer_vocab_size")
    if args.vocab_size is None or expected_vocab is not None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise SystemExit("Pilot runner requires the `tokenizers` package to validate vocabulary size") from exc
        if not args.tokenizer.is_file():
            raise SystemExit(f"Tokenizer file not found: {args.tokenizer}")
        tokenizer_vocab = Tokenizer.from_file(str(args.tokenizer)).get_vocab_size(with_added_tokens=True)
        if expected_vocab is not None and tokenizer_vocab != int(expected_vocab):
            raise SystemExit(
                f"Tokenizer has {tokenizer_vocab:,} entries; config requires {int(expected_vocab):,}. "
                "Refuse to train with a mismatched vocabulary."
            )
        if args.vocab_size is not None and args.vocab_size != tokenizer_vocab:
            raise SystemExit(f"--vocab-size {args.vocab_size:,} does not match tokenizer vocabulary {tokenizer_vocab:,}")
        args.vocab_size = tokenizer_vocab
    recipe_ids = list(config["recipes"]) if args.recipe == "all" else [args.recipe]
    for recipe_id in recipe_ids:
        run_recipe(args, config, recipe_id)


if __name__ == "__main__":
    main()
