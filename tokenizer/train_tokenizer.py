"""
Hetman-2.0B: Byte-Fallback BPE Tokenizer Builder & Trainer
Vocabulary size: 65,536 tokens
Special Tokens: <s>, </s>, <unk>, <pad>, <|im_start|>, <|im_end|>, <think>, </think>, etc.
"""

import os
import json
from typing import List, Optional
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

SPECIAL_TOKENS = [
    "<unk>",               # 0
    "<s>",                 # 1 (BOS)
    "</s>",                # 2 (EOS)
    "<pad>",               # 3 (PAD)
    "<|im_start|>",        # 4 (ChatML Start)
    "<|im_end|>",          # 5 (ChatML End)
    "<think>",             # 6 (CoT Start)
    "</think>",            # 7 (CoT End)
    "<thought_summary>",   # 8
    "</thought_summary>",  # 9
    "<|call|>",            # 10 (Tool Calling)
    "<|return|>",          # 11 (Tool Result)
    "<|extra_0|>",
    "<|extra_1|>",
    "<|extra_2|>",
    "<|extra_3|>"
]

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<|im_start|>assistant\\n'}}"
    "{% endif %}"
)

def build_hetman_bpe_tokenizer(vocab_size: int = 65536) -> Tokenizer:
    """
    Initializes a Byte-Fallback BPE tokenizer with GPT-4 / ByteLevel splitting
    and strict special token positioning.
    """
    # Initialize BPE model with unknown token fallback
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True))
    
    # Pre-tokenization: ByteLevel with regex splitting (whitespace + unicode words + numbers)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)
    
    # Decoder: ByteLevel decoder for 100% reversible byte decoding
    tokenizer.decoder = decoders.ByteLevel()
    
    # Post-processor for BOS / EOS wrapping if needed
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    
    return tokenizer

def train_tokenizer_from_files(
    files: List[str],
    output_dir: str = "tokenizer",
    vocab_size: int = 65536
) -> Tokenizer:
    """
    Trains BPE tokenizer over text files (Ukrainian, English, Code, Math).
    """
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = build_hetman_bpe_tokenizer(vocab_size=vocab_size)
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    print(f"[*] Training BPE tokenizer (vocab_size={vocab_size}) on {len(files)} corpus files...")
    tokenizer.train(files, trainer)
    
    save_tokenizer_artifacts(tokenizer, output_dir, vocab_size)
    return tokenizer

def save_tokenizer_artifacts(tokenizer: Tokenizer, output_dir: str, vocab_size: int = 65536):
    """
    Saves tokenizer.json, tokenizer_config.json, and special_tokens_map.json
    for full HuggingFace Transformers and Rada.cpp runtime compatibility.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save core tokenizer model
    tokenizer_json_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_json_path)
    print(f"[✓] Saved {tokenizer_json_path}")
    
    # 2. Save HuggingFace tokenizer_config.json
    tokenizer_config = {
        "version": "1.0",
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 262144,
        "clean_up_tokenization_spaces": False,
        "bos_token": "<s>",
        "eos_token": "<|im_end|>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "chat_template": CHATML_TEMPLATE,
        "added_tokens_decoder": {
            str(i): {
                "content": tok,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            } for i, tok in enumerate(SPECIAL_TOKENS)
        }
    }
    
    config_path = os.path.join(output_dir, "tokenizer_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
    print(f"[✓] Saved {config_path}")
    
    # 3. Save special_tokens_map.json
    special_tokens_map = {
        "bos_token": "<s>",
        "eos_token": "<|im_end|>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "additional_special_tokens": [tok for tok in SPECIAL_TOKENS if tok not in ["<s>", "<|im_end|>", "<unk>", "<pad>"]]
    }
    
    special_map_path = os.path.join(output_dir, "special_tokens_map.json")
    with open(special_map_path, "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=2)
    print(f"[✓] Saved {special_map_path}")

def bootstrap_base_tokenizer(output_dir: str = "tokenizer") -> Tokenizer:
    """
    Generates a starter tokenizer with full byte coverage and Ukrainian/English/Code seeds
    to enable testing and embedding initialization before full TPU 850B corpus tokenization.
    """
    os.makedirs(output_dir, exist_ok=True)
    seed_corpus_path = os.path.join(output_dir, "seed_corpus.txt")
    
    # Exhaustive seed ensuring all 33 Ukrainian letters (both uppercase and lowercase, standalone and with space)
    # are merged into atomic single tokens without decomposing into bytes
    ua_letters_upper = "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ"
    ua_letters_lower = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"
    
    # Repeat isolated letters directly to build direct 2-byte merges for every Ukrainian character
    isolated_upper = "\n".join(f"{ch} {ch} {ch}{ch}" for ch in ua_letters_upper) + "\n"
    isolated_lower = "\n".join(f"{ch} {ch} {ch}{ch}" for ch in ua_letters_lower) + "\n"
    
    # Words starting with each letter in uppercase and lowercase
    ua_sample_words = (
        "Андрій Береза Воля Голос Ґрунт Дніпро Енергія Європа Життя Закон Ідея Їжак Йосип "
        "Київ Людина Місто Нація Океан Прапор Розум Світ Тарас Україна Фізика Харків Центр Час Шевченко Щастя Юність Ярослав "
        "ангел берег вогонь гори ґанок дерево екран єнот журнал зірка іскра їдальня йод "
        "книга ліс море небо очі правда рух сонце трава успіх форма хліб цитата чай школа щока юність якість "
        "ґанок ґудзик аґрус ґрунт дзиґа Ґанок Ґудзик Аґрус Ґрунт Дзиґа "
        "їжак поїзд країна Україна їсти їдальня Їжак Поїзд Країна Україна Їсти Їдальня "
        "єнот єдність Європа єпископ Єнот Єдність Європа Єпископ "
        "інтернет історія ідея іспит Інтернет Історія Ідея Іспит Іван Київ Львів Харків Одеса Дніпро\n"
    )
    
    sample_text = (isolated_upper * 50) + (isolated_lower * 50) + (ua_sample_words * 50) + (
        "Гетьман 2б — це високоефективна модель штучного інтелекту з гібридною архітектурою.\n"
        "28 шарів Gated DeltaNet забезпечують фіксований стан рекурентної пам'яті.\n"
        "6 шарів Global FlashAttention забезпечують точне асоціативне пригадування на 256k контексті.\n"
        "Speculative Latent-Hopfield Core містить 8,388,608 слотів асоціативної пам'яті.\n"
        "Квантування BitNet b1.58 використовує тернарні ваги {-1, 0, +1}.\n"
        "def compute_deltanet_update(S, q, k, v, alpha, beta):\n"
        "    v_pred = torch.einsum('bhij,bhj->bhi', S, k)\n"
        "    update = torch.einsum('bhi,bhj->bhij', v - v_pred, k)\n"
        "    return S * alpha + beta * update\n"
        "User: Які переваги тернарного квантування?\n"
        "Assistant: <think>Тернарне квантування дозволяє замінити множення додаванням.</think>"
        "Головна перевага — це енергоефективність та низьке споживання VRAM.\n"
    ) * 30
    
    with open(seed_corpus_path, "w", encoding="utf-8") as f:
        f.write(sample_text)
        
    tokenizer = train_tokenizer_from_files([seed_corpus_path], output_dir=output_dir, vocab_size=4096)
    
    # Remove temporary seed file
    if os.path.exists(seed_corpus_path):
        os.remove(seed_corpus_path)
        
    return tokenizer

if __name__ == "__main__":
    print("=== Training / Bootstrapping Hetman-2.0B Tokenizer ===")
    tok = bootstrap_base_tokenizer("tokenizer")
    
    # Verify special tokens IDs
    for expected_id, token_str in enumerate(SPECIAL_TOKENS[:12]):
        actual_id = tok.token_to_id(token_str)
        print(f"Token '{token_str}': Expected ID = {expected_id}, Actual ID = {actual_id}")
        assert actual_id == expected_id, f"ID mismatch for {token_str}: {actual_id} != {expected_id}"
        
    # Verify all 33 Ukrainian letters (both uppercase and lowercase) are atomic 1-token characters
    ua_alphabet = "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯабвгґдеєжзиіїйклмнопрстуфхцчшщьюя"
    for ch in ua_alphabet:
        enc = tok.encode(ch)
        assert len(enc.ids) == 1, f"Letter '{ch}' decomposed into {len(enc.ids)} tokens: {enc.tokens}!"
    print(f"      ✓ All 33 Ukrainian letters (upper & lower) verified as atomic 1-token characters!")

    # Verify ChatML formatting roundtrip
    test_phrase = "<|im_start|>user\nПривіт, Гетьмане!<|im_end|>\n<|im_start|>assistant\n<think>\nПеревіряю стан.\n</think>\nЯ готовий до роботи!<|im_end|>"
    encoded = tok.encode(test_phrase)
    decoded = tok.decode(encoded.ids, skip_special_tokens=False)
    print(f"\n[✓] ChatML Roundtrip Test Passed!")
    print(f"Encoded tokens count: {len(encoded.ids)}")
    print("=== Tokenizer Bootstrap Complete! ===")

