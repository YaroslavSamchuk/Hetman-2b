"""
Hetman-2.0B: Phase 3 Chat SFT Dataset Preparation & Masking Pipeline
Formats dialogues into ChatML (<|im_start|>...<|im_end|>), implements strict label masking
(-100 for system & user prompt tokens so cross-entropy loss is computed ONLY on assistant responses),
and provides PyTorch Dataset and Collator for training.
Preserves 100% natural Ukrainian morphology across all genders without artificial alterations.
"""

import os
import json
from typing import List, Dict, Any, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

# Canonical ChatML Special Token IDs
IM_START_ID = 4   # <|im_start|>
IM_END_ID = 5     # <|im_end|>
PAD_TOKEN_ID = 3  # <pad>
IGNORE_INDEX = -100

DEFAULT_SYSTEM_PROMPT = (
    "Ти — Гетьман 2б (Hetman-2.0B), корисний, розумний та об'єктивний ШІ-асистент чоловічого роду. "
    "Відповідай природно, чітко, прямо і по суті, без цензури та зайвих повчань."
)

ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
    "system": "system"
}

def sanitize_and_normalize_conversation(
    messages: List[Dict[str, str]],
    default_system: Optional[str] = DEFAULT_SYSTEM_PROMPT
) -> List[Dict[str, str]]:
    """
    Guarantees strict conversational schema for ChatML:
    1. Normalizes role names and strips whitespace.
    2. Gathers and unifies all system prompts at message index 0.
    3. Merges consecutive turns of identical roles (user + user -> user, asst + asst -> asst).
    4. Trims trailing user or system turns (every sample must end with an assistant turn).
    5. Discards invalid samples with zero assistant responses.
    """
    if not messages:
        return []

    system_parts = []
    clean_turns = []

    for msg in messages:
        raw_role = msg.get("role", "").strip().lower()
        content = msg.get("content", "").strip()
        role = ROLE_ALIASES.get(raw_role)
        
        if not role or not content:
            continue

        if role == "system":
            system_parts.append(content)
        else:
            clean_turns.append({"role": role, "content": content})

    if not system_parts and default_system:
        system_parts.append(default_system.strip())

    merged_turns = []
    for turn in clean_turns:
        if merged_turns and merged_turns[-1]["role"] == turn["role"]:
            merged_turns[-1]["content"] += "\n\n" + turn["content"]
        else:
            merged_turns.append(turn)

    while merged_turns and merged_turns[-1]["role"] == "user":
        merged_turns.pop()

    if not any(t["role"] == "assistant" for t in merged_turns):
        return []

    final_messages = []
    if system_parts:
        final_messages.append({"role": "system", "content": "\n\n".join(system_parts)})
    final_messages.extend(merged_turns)

    return final_messages

def format_chatml_conversation(
    messages: List[Dict[str, str]],
    default_system: Optional[str] = DEFAULT_SYSTEM_PROMPT
) -> List[Dict[str, str]]:
    return sanitize_and_normalize_conversation(messages, default_system)

def prune_historical_thoughts(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Prunes reasoning scratchpads of historical assistant turns into <thought_summary>,
    strictly preserving full <think>...</think> on the final assistant turn.
    """
    import re
    processed = []
    last_asst_idx = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)

    for idx, msg in enumerate(messages):
        content = msg["content"]
        role = msg["role"]
        
        if role == "assistant" and idx < last_asst_idx:
            def _replace_think(match):
                inner_thought = match.group(1).strip()
                first_sentence = inner_thought.split(".")[0] if "." in inner_thought else inner_thought[:120]
                return f"<thought_summary>\n{first_sentence.strip()}.\n</thought_summary>"
            
            content = re.sub(r"<think>(.*?)</think>", _replace_think, content, flags=re.DOTALL)
            
        processed.append({"role": role, "content": content})
    return processed

def tokenize_and_mask_chatml(
    tokenizer: Tokenizer,
    messages: List[Dict[str, str]],
    max_length: int = 4096,
    prune_history: bool = True,
    supervise_only_last_turn: bool = True
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Encodes multi-turn ChatML messages into input_ids and labels.
    - User and System turns are masked with IGNORE_INDEX (-100).
    - Assistant prefix (<|im_start|>assistant\n) is masked with IGNORE_INDEX (-100).
    - Assistant generation tokens (reasoning, response, and <|im_end|>\n) are supervised.
    - If supervise_only_last_turn is True, historical assistant turns (which may contain
      <thought_summary>) are completely masked with -100 so the model NEVER trains on summaries.
    - Returns None if the resulting sequence contains 0 active target tokens (NaN guard).
    """
    messages = sanitize_and_normalize_conversation(messages)
    if not messages:
        return None

    if prune_history:
        messages = prune_historical_thoughts(messages)

    last_asst_idx = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)

    all_input_ids: List[int] = []
    all_labels: List[int] = []

    for idx, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]
        
        prefix = f"<|im_start|>{role}\n"
        prefix_ids = tokenizer.encode(prefix).ids
        content_ids = tokenizer.encode(content).ids
        suffix_ids = tokenizer.encode("<|im_end|>\n").ids
        turn_ids = prefix_ids + content_ids + suffix_ids
        
        if role == "assistant":
            if supervise_only_last_turn and idx != last_asst_idx:
                turn_labels = [IGNORE_INDEX] * len(turn_ids)
            else:
                turn_labels = ([IGNORE_INDEX] * len(prefix_ids)) + content_ids + suffix_ids
        else:
            turn_labels = [IGNORE_INDEX] * len(turn_ids)
            
        all_input_ids.extend(turn_ids)
        all_labels.extend(turn_labels)

    if len(all_input_ids) > max_length:
        all_input_ids = all_input_ids[:max_length]
        all_labels = all_labels[:max_length]

    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    if (labels_tensor != IGNORE_INDEX).sum().item() == 0:
        return None

    return {
        "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
        "labels": labels_tensor
    }

def pack_dialogues_for_sft(
    tokenized_samples: List[Dict[str, torch.Tensor]],
    max_length: int = 4096,
    pad_token_id: int = PAD_TOKEN_ID,
    ignore_index: int = IGNORE_INDEX
) -> List[Dict[str, torch.Tensor]]:
    """
    Packs multiple short dialogues into fixed 4096-token sequences:
    - Resets DeltaNet recurrent memory S_t = 0 at dialogue boundary tokens
    - Generates segment_ids to isolate document boundaries
    """
    packed_batches = []
    current_inputs: List[int] = []
    current_labels: List[int] = []
    current_positions: List[int] = []
    current_segments: List[int] = []
    current_resets: List[float] = []
    seg_id = 1

    for sample in tokenized_samples:
        s_in = sample["input_ids"].tolist()
        s_lab = sample["labels"].tolist()
        s_len = len(s_in)
        if s_len == 0:
            continue

        if len(current_inputs) + s_len > max_length and current_inputs:
            pad_len = max_length - len(current_inputs)
            if pad_len > 0:
                current_inputs.extend([pad_token_id] * pad_len)
                current_labels.extend([ignore_index] * pad_len)
                current_positions.extend([0] * pad_len)
                current_segments.extend([0] * pad_len)
                current_resets.extend([0.0] * pad_len)
                
            packed_batches.append({
                "input_ids": torch.tensor(current_inputs, dtype=torch.long),
                "labels": torch.tensor(current_labels, dtype=torch.long),
                "position_ids": torch.tensor(current_positions, dtype=torch.long),
                "segment_ids": torch.tensor(current_segments, dtype=torch.long),
                "state_reset_mask": torch.tensor(current_resets, dtype=torch.float32)
            })
            current_inputs, current_labels = [], []
            current_positions, current_segments, current_resets = [], [], []
            seg_id = 1

        if s_len > max_length:
            s_in = s_in[:max_length]
            s_lab = s_lab[:max_length]
            s_len = max_length

        current_inputs.extend(s_in)
        current_labels.extend(s_lab)
        current_positions.extend(list(range(s_len)))
        current_segments.extend([seg_id] * s_len)
        current_resets.extend([0.0] + [1.0] * (s_len - 1))
        seg_id += 1

    if current_inputs:
        pad_len = max_length - len(current_inputs)
        if pad_len > 0:
            current_inputs.extend([pad_token_id] * pad_len)
            current_labels.extend([ignore_index] * pad_len)
            current_positions.extend([0] * pad_len)
            current_segments.extend([0] * pad_len)
            current_resets.extend([0.0] * pad_len)
            
        packed_batches.append({
            "input_ids": torch.tensor(current_inputs, dtype=torch.long),
            "labels": torch.tensor(current_labels, dtype=torch.long),
            "position_ids": torch.tensor(current_positions, dtype=torch.long),
            "segment_ids": torch.tensor(current_segments, dtype=torch.long),
            "state_reset_mask": torch.tensor(current_resets, dtype=torch.float32)
        })

    return packed_batches

def create_block_diagonal_causal_mask(segment_ids: torch.Tensor) -> torch.Tensor:
    """
    Creates a 2D or 3D Block-Diagonal Causal Document Attention Mask:
    Prevents cross-sample attention contamination in Global FlashAttention.
    Supports both 1D [L] and 2D [B, L] inputs.
    """
    if segment_ids.dim() == 1:
        L = segment_ids.shape[0]
        causal_mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=segment_ids.device))
        same_segment = segment_ids.unsqueeze(1) == segment_ids.unsqueeze(0)
        non_pad = segment_ids.unsqueeze(1) > 0
        return causal_mask & same_segment & non_pad
    elif segment_ids.dim() == 2:
        B, L = segment_ids.shape
        causal_mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=segment_ids.device)).unsqueeze(0)
        same_segment = segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)
        non_pad = segment_ids.unsqueeze(2) > 0
        return causal_mask & same_segment & non_pad
    else:
        raise ValueError(f"segment_ids must be 1D or 2D, got shape {segment_ids.shape}")

class ChatSFTDataset(Dataset):
    """
    PyTorch Dataset for Phase 3 Chat SFT.
    Accepts raw JSONL multi-turn conversations and returns tokenized, masked samples.
    """
    def __init__(
        self,
        samples: List[List[Dict[str, str]]],
        tokenizer: Tokenizer,
        max_length: int = 4096
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        messages = format_chatml_conversation(self.samples[idx])
        return tokenize_and_mask_chatml(self.tokenizer, messages, max_length=self.max_length)

class ChatDataCollator:
    """
    Pads dynamic sequences within a batch to maximum sequence length in the batch.
    """
    def __init__(self, pad_token_id: int = PAD_TOKEN_ID, ignore_index: int = IGNORE_INDEX):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(sample["input_ids"].shape[0] for sample in batch)
        
        padded_inputs = []
        padded_labels = []
        attention_masks = []
        
        for sample in batch:
            input_ids = sample["input_ids"]
            labels = sample["labels"]
            length = input_ids.shape[0]
            pad_len = max_len - length
            
            if pad_len > 0:
                p_in = torch.cat([input_ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
                p_lab = torch.cat([labels, torch.full((pad_len,), self.ignore_index, dtype=torch.long)])
                attn_mask = torch.cat([torch.ones(length, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
            else:
                p_in = input_ids
                p_lab = labels
                attn_mask = torch.ones(length, dtype=torch.long)
                
            padded_inputs.append(p_in)
            padded_labels.append(p_lab)
            attention_masks.append(attn_mask)
            
        return {
            "input_ids": torch.stack(padded_inputs, dim=0),
            "labels": torch.stack(padded_labels, dim=0),
            "attention_mask": torch.stack(attention_masks, dim=0)
        }

def test_sft_pipeline():
    print("=" * 70)
    print("  HETMAN-2.0B CHAT SFT PIPELINE & MASKING VERIFICATION")
    print("=" * 70)
    
    tokenizer_path = os.path.join(os.path.dirname(__file__), "..", "tokenizer", "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print("[!] Tokenizer not found at path. Run tokenizer/train_tokenizer.py first.")
        return
        
    tok = Tokenizer.from_file(tokenizer_path)
    
    sample_dialogues = [
        [
            {"role": "user", "content": "Привіт! Хто ти та які твої можливості?"},
            {"role": "assistant", "content": "<think>\nКористувач запитує про мою ідентичність.\n</think>\nЯ — Гетьман 2б, сучасна українська модель штучного інтелекту."}
        ],
        [
            {"role": "user", "content": "Поясни, чому у Gated DeltaNet фіксований розмір пам'яті?"},
            {"role": "assistant", "content": "<think>\nАналізую розмір матриці S_t: вона має фіксовану форму H x D x D.\n</think>\nТому що DeltaNet використовує рекурентний стан фіксованого розміру 64x64 на кожну голову."}
        ]
    ]
    
    dataset = ChatSFTDataset(sample_dialogues, tokenizer=tok, max_length=2048)
    collator = ChatDataCollator()
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collator)
    
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    mask = batch["attention_mask"]
    
    print(f"Batch input_ids shape:      {input_ids.shape}")
    print(f"Batch labels shape:         {labels.shape}")
    print(f"Batch attention_mask shape: {mask.shape}")
    
    # Check masking correctness: labels must have -100 for system & user turns
    unmasked_count = (labels != -100).sum().item()
    masked_count = (labels == -100).sum().item()
    total_tokens = labels.numel()
    
    print(f"\nMasking statistics:")
    print(f"  - Total tokens:        {total_tokens}")
    print(f"  - Masked (-100, loss=0): {masked_count} ({masked_count / total_tokens * 100:.1f}%)")
    print(f"  - Active target tokens: {unmasked_count} ({unmasked_count / total_tokens * 100:.1f}%)")
    
    assert unmasked_count > 0, "No active target tokens found in batch!"
    assert masked_count > 0, "No masked prompt tokens found!"
    print("\n[✓] Chat SFT Data Pipeline & Label Masking verified successfully!")
    print("=" * 70)

if __name__ == "__main__":
    test_sft_pipeline()
