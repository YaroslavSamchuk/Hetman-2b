"""Experimental local generation from a checkpoint written by train_hetman.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train_hetman import _model_config
from modeling_hetman import HetmanForCausalLM


def _build_prompt_ids(tokenizer, prompt: str, system_prompt: str) -> list[int]:
    expected = {
        "<s>": 1,
        "</s>": 2,
        "<pad>": 3,
        "<|im_start|>": 4,
        "<|im_end|>": 5,
    }
    for token, token_id in expected.items():
        actual = tokenizer.token_to_id(token)
        if actual != token_id:
            raise ValueError(f"Tokenizer special token {token!r} has ID {actual}, expected {token_id}")
    def escape_delimiters(text: str) -> str:
        return text.replace("<|im_start|>", "&lt;|im_start|&gt;").replace("<|im_end|>", "&lt;|im_end|&gt;")

    formatted = (
        f"<|im_start|>system\n{escape_delimiters(system_prompt)}<|im_end|>\n"
        f"<|im_start|>user\n{escape_delimiters(prompt)}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return [expected["<s>"]] + tokenizer.encode(formatted, add_special_tokens=False).ids


def run(args: argparse.Namespace) -> str:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.tokenizer.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if args.temperature < 0 or not torch.isfinite(torch.tensor(args.temperature)):
        raise ValueError("temperature must be finite and non-negative")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("Generation requires the `tokenizers` package") from exc

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    config = _model_config(args)
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if vocab_size != config.vocab_size:
        raise ValueError(f"Tokenizer vocabulary ({vocab_size}) does not match model config ({config.vocab_size})")
    input_ids = _build_prompt_ids(tokenizer, args.prompt, args.system_prompt)
    if len(input_ids) + args.max_new_tokens > config.max_position_embeddings:
        raise ValueError("Prompt plus max_new_tokens exceeds the configured context length")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain the model state saved by train_hetman.py")
    model = HetmanForCausalLM(config)
    model.load_state_dict(state, strict=True)
    model.to(torch.device(args.device))
    model.eval()

    ids = torch.tensor([input_ids], dtype=torch.long, device=args.device)
    with torch.inference_mode():
        generated = model.generate(
            ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=(2, 5),
            pad_token_id=3,
            do_sample=args.temperature > 0,
        )
    new_ids = generated[0, len(input_ids):].tolist()
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trainer checkpoint file, e.g. step_00001000.pt")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=None, help="Optional Hetman architecture config JSON")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system-prompt", default="Answer clearly and state uncertainty when you cannot verify a claim.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        print(run(parse_args()))
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
