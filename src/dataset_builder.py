"""
Hetman-2.0B: Memory-Mapped Binary Shard Writer & Split Generation Suite (Step 5)
Implements:
1. Deterministic Pre-Curation Validation Splitter (stable 1% held-out per source).
2. Memory-Mapped Binary Shard Writer & Reader (.bin / .idx, uint32, zero-copy O(1) access).
3. Source-Aware Sequence Packer (strict source & split boundary isolation, no cross-source contamination).
4. Full Dataset Recipe Orchestrator (reads real JSONL/Parquet/Text, applies curation, builds Recipes A/B/C).
5. Cross-split Deduplication (guarantees 0% validation-to-train leakage).
"""

import os
import sys
import zlib
import json
import struct
import hashlib
from typing import List, Dict, Optional, Tuple, Iterator, Any
import numpy as np

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None

try:
    from src.data_curation_pipeline import DataCurationPipeline, DecontaminationAuditor, DocumentMetadata
except ImportError:
    from data_curation_pipeline import DataCurationPipeline, DecontaminationAuditor, DocumentMetadata

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Header specification for .idx file
IDX_MAGIC = b"HETMAN_BINIDX\x00"  # 14 bytes
IDX_VERSION = 2                    # 2 bytes uint16
HEADER_SIZE = 64                   # 64 bytes fixed header
INDEX_RECORD_FORMAT = "<QIIB3s"    # 20 bytes: offset(Q), token_count(I), source_crc(I), split(B), pad(3s)
INDEX_RECORD_SIZE = struct.calcsize(INDEX_RECORD_FORMAT)

SPLIT_MAP = {"train": 0, "val": 1, "test": 2}
INV_SPLIT_MAP = {v: k for k, v in SPLIT_MAP.items()}


# ==============================================================================
# 1. DETERMINISTIC PRE-CURATION VALIDATION SPLITTER
# ==============================================================================

class DocumentSplitter:
    """
    Deterministically partitions raw documents into train (99%) and validation (1%)
    before any curation or filtering occurs, using a stable cryptographic hash.
    Guarantees reproducible splits regardless of iteration order, sharding, or worker count.
    """
    def __init__(self, fraction_val: float = 0.01, seed: int = 42):
        self.fraction_val = fraction_val
        self.seed = seed
        self.threshold = int(fraction_val * 1_000_000)

    def assign_split(self, source_id: str, doc_id: str) -> str:
        key = f"{self.seed}:{source_id}:{doc_id}".encode("utf-8")
        h = int(hashlib.sha256(key).hexdigest()[:8], 16) % 1_000_000
        return "val" if h < self.threshold else "train"


# ==============================================================================
# 2. MEMORY-MAPPED BINARY SHARD WRITER (.bin / .idx)
# ==============================================================================

class BinaryShardWriter:
    """
    Writes tokenized sequences into high-performance memory-mapped shards:
    - .bin: Flat binary stream of little-endian uint32 token IDs.
    - .idx: Binary index file containing sequence byte offsets, token counts, source IDs, and splits.
    """
    def __init__(
        self,
        base_path: str,
        sequence_length: int = 4096,
        dtype: str = "uint32"
    ):
        self.base_path = base_path
        self.bin_path = f"{base_path}.bin"
        self.idx_path = f"{base_path}.idx"
        self.sequence_length = sequence_length
        self.dtype = dtype
        
        os.makedirs(os.path.dirname(os.path.abspath(self.bin_path)), exist_ok=True)
        self.bin_file = open(self.bin_path, "wb")
        self.idx_file = open(self.idx_path, "wb")
        
        # Write dummy header to .idx (will be finalized on close)
        self.idx_file.write(b"\x00" * HEADER_SIZE)
        
        self.total_sequences = 0
        self.total_tokens = 0
        self.current_byte_offset = 0

    def add_sequence(
        self,
        token_ids: List[int],
        source_id: str = "unknown",
        split: str = "train"
    ):
        """Appends one sequence of tokens to .bin and registers its index entry."""
        count = len(token_ids)
        if count == 0:
            return
        if count > self.sequence_length:
            raise ValueError(
                f"Sequence has {count} tokens, exceeding configured length {self.sequence_length}"
            )
        if split not in SPLIT_MAP:
            raise ValueError(f"Unsupported split {split!r}; expected one of {tuple(SPLIT_MAP)}")
        if any(token_id < 0 or token_id > 0xFFFFFFFF for token_id in token_ids):
            raise ValueError("Token IDs must fit in uint32")
            
        # Write binary tokens to .bin (uint32 little-endian)
        arr = np.asarray(token_ids, dtype="<u4")
        bytes_data = arr.tobytes()
        self.bin_file.write(bytes_data)
        
        # Prepare index record
        source_crc = zlib.crc32(source_id.encode("utf-8"))
        split_code = SPLIT_MAP.get(split, 0)
        record = struct.pack(
            INDEX_RECORD_FORMAT,
            self.current_byte_offset,
            count,
            source_crc,
            split_code,
            b"\x00\x00\x00"
        )
        self.idx_file.write(record)
        
        self.current_byte_offset += len(bytes_data)
        self.total_sequences += 1
        self.total_tokens += count

    def close(self) -> Dict[str, Any]:
        """Flushes buffers and writes the final verified header into .idx."""
        self.bin_file.flush()
        bin_size = self.bin_file.tell()
        self.bin_file.close()
        
        # Write real 64-byte header
        self.idx_file.seek(0)
        # Format: magic(14s), version(H), dtype_code(H), seq_len(I), total_seqs(Q), total_tokens(Q), padding(26s) = 64 bytes
        header = struct.pack(
            "<14sHHIQQ26s",
            IDX_MAGIC,
            IDX_VERSION,
            4, # 4 bytes per token (uint32)
            self.sequence_length,
            self.total_sequences,
            self.total_tokens,
            b"\x00" * 26
        )
        self.idx_file.write(header)
        self.idx_file.flush()
        self.idx_file.seek(0, os.SEEK_END)
        idx_size = self.idx_file.tell()
        self.idx_file.close()
        
        return {
            "bin_path": self.bin_path,
            "idx_path": self.idx_path,
            "total_sequences": self.total_sequences,
            "total_tokens": self.total_tokens,
            "bin_size_bytes": bin_size,
            "idx_size_bytes": idx_size
        }

    def abort(self) -> None:
        """Close incomplete shard files without writing a valid-looking header."""
        for handle in (self.bin_file, self.idx_file):
            if not handle.closed:
                handle.close()


# ==============================================================================
# 3. MEMORY-MAPPED DATASET READER (ZERO-COPY O(1) ACCESS)
# ==============================================================================

class MMapIndexedDataset:
    """
    Zero-copy memory-mapped dataset reader. Opens .idx and .bin for instant O(1)
    sequence indexing during training without loading the dataset into memory.
    """
    def __init__(self, base_path: str):
        self.bin_path = f"{base_path}.bin"
        self.idx_path = f"{base_path}.idx"
        
        if not os.path.exists(self.bin_path) or not os.path.exists(self.idx_path):
            raise FileNotFoundError(f"Shard files missing: {self.bin_path} or {self.idx_path}")
            
        # Read .idx header
        with open(self.idx_path, "rb") as f:
            header_bytes = f.read(HEADER_SIZE)
            if len(header_bytes) != HEADER_SIZE:
                raise ValueError("Index file is shorter than its header")
            magic, version, dtype_code, self.sequence_length, self.total_sequences, self.total_tokens, _ = struct.unpack(
                "<14sHHIQQ26s",
                header_bytes
            )
            if magic != IDX_MAGIC:
                raise ValueError(f"Corrupt index header magic: {magic}")
            if version != IDX_VERSION:
                raise ValueError(f"Unsupported index version: {version}")
            if dtype_code != 4:
                raise ValueError(f"Unsupported token dtype code: {dtype_code}")
                
            # Read index records into structured array
            idx_records_bytes = f.read()
            if len(idx_records_bytes) % INDEX_RECORD_SIZE:
                raise ValueError("Index file ends with an incomplete record")
            num_records = len(idx_records_bytes) // INDEX_RECORD_SIZE
            if num_records != self.total_sequences:
                raise ValueError(f"Index count mismatch: header={self.total_sequences}, actual={num_records}")

            if self.sequence_length < 2:
                raise ValueError("Shard sequence length must be at least two tokens")
            if self.total_sequences:
                index_dtype = np.dtype([
                    ("offset", "<u8"), ("count", "<u4"), ("source_crc", "<u4"),
                    ("split", "u1"), ("pad", "V3")
                ])
                records = np.frombuffer(idx_records_bytes, dtype=index_dtype)
                counts = records["count"].astype(np.uint64, copy=False)
                offsets = records["offset"]
                if np.any(counts < 2) or np.any(counts > self.sequence_length):
                    raise ValueError("Index contains an invalid sequence token count")
                expected_offsets = np.concatenate((np.zeros(1, dtype=np.uint64), np.cumsum(counts[:-1], dtype=np.uint64) * 4))
                if not np.array_equal(offsets, expected_offsets):
                    raise ValueError("Index sequence offsets are not contiguous")
                if int(counts.sum(dtype=np.uint64)) != self.total_tokens:
                    raise ValueError("Index token total disagrees with its header")
                if np.any(~np.isin(records["split"], tuple(SPLIT_MAP.values()))):
                    raise ValueError("Index contains an unsupported split tag")

        bin_size = os.path.getsize(self.bin_path)
        if bin_size % 4 or bin_size // 4 != self.total_tokens:
            raise ValueError("Binary shard size does not match the index token total")
                
        # Lazy initialization state: memmaps are opened on-demand per process
        # to ensure zero IPC pickling overhead and complete safety with multi-worker DataLoader.
        self._index_mmap = None
        self._bin_mmap = None
        self._init_pid = None

    @property
    def index_mmap(self):
        self._ensure_open()
        return self._index_mmap

    @index_mmap.setter
    def index_mmap(self, val):
        self._index_mmap = val

    @property
    def bin_mmap(self):
        self._ensure_open()
        return self._bin_mmap

    @bin_mmap.setter
    def bin_mmap(self, val):
        self._bin_mmap = val

    def _ensure_open(self):
        current_pid = os.getpid()
        if self._index_mmap is None or self._bin_mmap is None or self._init_pid != current_pid:
            self._index_mmap = np.memmap(
                self.idx_path,
                dtype=[
                    ("offset", "<u8"),
                    ("count", "<u4"),
                    ("source_crc", "<u4"),
                    ("split", "u1"),
                    ("pad", "V3")
                ],
                mode="r",
                offset=HEADER_SIZE,
                shape=(self.total_sequences,)
            )
            self._bin_mmap = np.memmap(
                self.bin_path,
                dtype="<u4",
                mode="r"
            )
            self._init_pid = current_pid

    def __getstate__(self) -> Dict[str, Any]:
        """Excludes open memmap objects from pickle payload for lightweight worker spawning."""
        state = self.__dict__.copy()
        state["_index_mmap"] = None
        state["_bin_mmap"] = None
        state["_init_pid"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)
        self._index_mmap = None
        self._bin_mmap = None
        self._init_pid = None

    def close(self):
        """Releases OS file mapping handles cleanly."""
        if self._index_mmap is not None:
            if hasattr(self._index_mmap, "_mmap") and self._index_mmap._mmap is not None:
                try:
                    self._index_mmap._mmap.close()
                except Exception:
                    pass
            self._index_mmap = None
        if self._bin_mmap is not None:
            if hasattr(self._bin_mmap, "_mmap") and self._bin_mmap._mmap is not None:
                try:
                    self._bin_mmap._mmap.close()
                except Exception:
                    pass
            self._bin_mmap = None
        self._init_pid = None

    def __len__(self) -> int:
        return self.total_sequences

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self.total_sequences:
            raise IndexError(f"Index {idx} out of range [0, {self.total_sequences})")
            
        self._ensure_open()
        entry = self.index_mmap[idx]
        byte_offset = int(entry["offset"])
        token_count = int(entry["count"])
        source_crc = int(entry["source_crc"])
        split_code = int(entry["split"])
        
        # Word offset in uint32 array (4 bytes per uint32)
        token_offset = byte_offset // 4
        tokens = self.bin_mmap[token_offset : token_offset + token_count]
        
        # Build attention mask and document boundary state_reset_mask:
        # Resets DeltaNet recurrent state S_t = 0 at BOS tokens (id=1) and sequence start
        if token_count < self.sequence_length:
            padded_tokens = np.pad(
                tokens,
                (0, self.sequence_length - token_count),
                mode="constant",
                constant_values=3 # PAD_TOKEN_ID
            )
            attention_mask = np.zeros(self.sequence_length, dtype=np.int64)
            attention_mask[:token_count] = 1
            
            resets = np.where((tokens == 1) | (np.arange(token_count) == 0), 0.0, 1.0).astype(np.float32)
            state_reset_mask = np.pad(
                resets,
                (0, self.sequence_length - token_count),
                mode="constant",
                constant_values=0.0
            )
        else:
            padded_tokens = tokens
            attention_mask = np.ones(self.sequence_length, dtype=np.int64)
            state_reset_mask = np.where((tokens == 1) | (np.arange(self.sequence_length) == 0), 0.0, 1.0).astype(np.float32)
            
        return {
            "input_ids": padded_tokens,
            "attention_mask": attention_mask,
            "state_reset_mask": state_reset_mask,
            "token_count": token_count,
            "source_crc": source_crc,
            "split": INV_SPLIT_MAP.get(split_code, "unknown")
        }


# ==============================================================================
# 4. SOURCE-AWARE SEQUENCE PACKER (STRICT BOUNDARY ISOLATION)
# ==============================================================================

class SourceAwareSequencePacker:
    """
    Packs documents into exact sequence_length chunks, strictly isolated per source and per split.
    Prevents cross-source and cross-split contamination:
    1. Sequences for source A contain ONLY source A tokens and are registered with source A CRC.
    2. When transitioning from source A to source B, un-emitted leftover tokens are NOT blended into source B.
    3. Train and Val streams use independent packer and writer instances.
    4. Supports emitting the final 512-token remainder sequence on the designated last source.
    """
    def __init__(
        self,
        writer: BinaryShardWriter,
        sequence_length: int = 4096,
        bos_token_id: int = 1,
        eos_token_id: int = 5,
        add_bos: bool = True
    ):
        self.writer = writer
        self.sequence_length = sequence_length
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.add_bos = add_bos
        
        self.active_source_id: Optional[str] = None
        self.active_split: str = "train"
        self.target_sequences: int = 0
        self.emitted_for_source: int = 0
        self.buffer: List[int] = []
        
        self.total_full_sequences: int = 0
        self.total_tokens_emitted: int = 0
        self.remainder_sequence_emitted: bool = False

    def start_source(self, source_id: str, target_sequences: int, split: str = "train"):
        """Switches active source and isolates previous buffer to prevent cross-source leakage."""
        if len(self.buffer) > 0 and self.active_source_id is not None:
            # Clear remaining buffer so leftover tokens from previous source do not leak
            self.buffer.clear()
            
        self.active_source_id = source_id
        self.active_split = split
        self.target_sequences = target_sequences
        self.emitted_for_source = 0

    def add_document(self, token_ids: List[int]) -> bool:
        """
        Adds document tokens to the active source buffer and emits complete sequences.
        Returns True if the target sequence quota for this source is fulfilled.
        """
        if self.emitted_for_source >= self.target_sequences:
            return True

        if not token_ids:
            return False

        doc_tokens = []
        if self.add_bos:
            if token_ids[0] != self.bos_token_id:
                doc_tokens.append(self.bos_token_id)
        doc_tokens.extend(token_ids)
        if doc_tokens[-1] != self.eos_token_id:
            doc_tokens.append(self.eos_token_id)
            
        self.buffer.extend(doc_tokens)
        
        while len(self.buffer) >= self.sequence_length:
            if self.emitted_for_source >= self.target_sequences:
                break
                
            chunk = self.buffer[: self.sequence_length]
            self.buffer = self.buffer[self.sequence_length :]
            
            self.writer.add_sequence(chunk, source_id=self.active_source_id, split=self.active_split)
            self.emitted_for_source += 1
            self.total_full_sequences += 1
            self.total_tokens_emitted += self.sequence_length
            
        return self.emitted_for_source >= self.target_sequences

    def finish_source(
        self,
        emit_remainder: bool = False,
        remainder_token_count: int = 512,
        allow_incomplete: bool = False
    ) -> Dict[str, Any]:
        """
        Finishes current source allocation. If emit_remainder is True, emits an explicit final sequence
        containing up to remainder_token_count tokens.
        """
        if self.emitted_for_source < self.target_sequences and not allow_incomplete:
            raise ValueError(
                f"Source '{self.active_source_id}' emitted {self.emitted_for_source} of "
                f"{self.target_sequences} required sequences."
            )
            
        remainder_emitted = 0
        if emit_remainder and len(self.buffer) > 0:
            count = min(remainder_token_count, len(self.buffer))
            chunk = self.buffer[:count]
            self.writer.add_sequence(chunk, source_id=self.active_source_id, split=self.active_split)
            self.total_tokens_emitted += count
            remainder_emitted = count
            self.remainder_sequence_emitted = True
            self.buffer = self.buffer[count:]
            
        stats = {
            "source_id": self.active_source_id,
            "target_sequences": self.target_sequences,
            "emitted_sequences": self.emitted_for_source,
            "remainder_tokens": remainder_emitted
        }
        self.buffer.clear()
        return stats


# Legacy alias for backward compatibility with simple single-stream packers
SequencePacker = SourceAwareSequencePacker


# ==============================================================================
# 5. DATASET RECIPE BUILDER & CORPUS INGESTION ORCHESTRATOR
# ==============================================================================

class DatasetRecipeBuilder:
    """
    Orchestrates end-to-end dataset creation for Recipes A, B, and C:
    1. Ingests raw files (.jsonl, .parquet, .txt) from configured corpus paths or HETMAN_DATA_ROOT.
    2. Applies deterministic pre-curation 1% held-out validation split per source.
    3. Runs benchmark decontamination on validation documents to guarantee zero eval leak.
    4. Curates training documents via DataCurationPipeline (Recipes B & C) or baseline filters (Recipe A).
    5. Performs cross-split train-to-val deduplication check.
    6. Emits pure memory-mapped shards (.bin / .idx) with exact sequence quotas and source metadata.
    7. Generates audit manifest.jsonl and summary stats.json.
    """
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs",
                "pilot_dataset_recipes.json"
            )
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.config_path = os.path.abspath(config_path)
            
        val_cfg = self.config.get("validation_split", {})
        self.splitter = DocumentSplitter(
            fraction_val=val_cfg.get("fraction_per_source", 0.01),
            seed=val_cfg.get("seed", 42)
        )
        self.target_tokens = int(self.config.get("total_tokens_per_recipe", 200_000_000))
        self.seq_len = int(self.config.get("sequence_length", 4096))

    def _load_benchmark_entries(self) -> List[Dict[str, str]]:
        settings = self.config.get("benchmark_decontamination", {})
        if not settings.get("required", False):
            return []
        configured_path = settings.get("prompt_file")
        if not configured_path:
            raise ValueError("benchmark_decontamination.required is true but prompt_file is not configured")
        path = configured_path if os.path.isabs(configured_path) else os.path.join(PROJECT_ROOT, configured_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Required benchmark prompt file is missing: {path}. Add the approved evaluation prompts "
                "before building any recipe that claims benchmark decontamination."
            )
        entries: List[Dict[str, str]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid benchmark JSONL at {path}:{line_number}: {exc}") from exc
                if not isinstance(entry, dict):
                    raise ValueError(f"Benchmark entry at {path}:{line_number} must be a JSON object")
                if not isinstance(entry.get("benchmark"), str) or not isinstance(entry.get("prompt"), str):
                    raise ValueError(f"Benchmark entry at {path}:{line_number} requires string benchmark and prompt fields")
                entries.append(entry)
        if not entries:
            raise ValueError(f"Required benchmark prompt file contains no entries: {path}")
        return entries

    def get_corpus_iterator(
        self,
        corpus_id: str,
        data_root: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Yields normalized raw document dictionaries from local corpus files.
        Supports .jsonl, .jsonl.gz, .parquet, and .txt.
        """
        corpus_info = self.config.get("corpora_sources", {}).get(corpus_id)
        if not corpus_info:
            raise KeyError(f"Corpus '{corpus_id}' is not defined in pilot_dataset_recipes.json")
            
        env_root = os.getenv(self.config.get("data_root_env_var", "HETMAN_DATA_ROOT"))
        base_root = data_root or env_root
        rel_path = corpus_info.get("local_path", "")
        
        if os.path.isabs(rel_path):
            effective_path = rel_path
        elif base_root:
            effective_path = os.path.join(base_root, corpus_id)
        else:
            effective_path = os.path.normpath(os.path.join(PROJECT_ROOT, rel_path))
            
        if not os.path.exists(effective_path):
            raise FileNotFoundError(
                f"Corpus '{corpus_id}' not found at '{effective_path}'. "
                f"Place raw data files in this folder or set HETMAN_DATA_ROOT."
            )
            
        doc_type = corpus_info.get("doc_type", "general_web")
        license_str = corpus_info.get("license", "Unknown")
        text_col = corpus_info.get("text_column", "text")
        
        files_to_read = []
        if os.path.isdir(effective_path):
            for root, dirs, files in os.walk(effective_path):
                dirs.sort()
                for f in sorted(files):
                    if f.endswith((".jsonl", ".parquet", ".txt", ".jsonl.gz")):
                        files_to_read.append(os.path.join(root, f))
        else:
            files_to_read.append(effective_path)
            
        if not files_to_read:
            raise FileNotFoundError(f"No valid data files (.jsonl, .parquet, .txt) found in '{effective_path}'.")
            
        doc_counter = 0
        for file_path in files_to_read:
            if file_path.endswith((".jsonl", ".jsonl.gz")):
                import gzip
                open_fn = gzip.open if file_path.endswith(".gz") else open
                with open_fn(file_path, "rt", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        text = record.get(text_col, record.get("text", record.get("content", "")))
                        if not text:
                            continue
                        doc_id = str(record.get("id", f"{corpus_id}_{doc_counter:08d}"))
                        doc_counter += 1
                        yield {
                            "id": doc_id,
                            "source": corpus_id,
                            "type": doc_type,
                            "license": license_str,
                            "text": text
                        }
            elif file_path.endswith(".parquet"):
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(file_path)
                available_columns = set(parquet_file.schema.names)
                columns = [c for c in (text_col, "id") if c in available_columns]
                if text_col not in columns:
                    raise ValueError(f"Parquet corpus '{corpus_id}' has no configured text column '{text_col}'")
                for batch in parquet_file.iter_batches(batch_size=2048, columns=columns):
                    values = batch.to_pydict()
                    texts = values[text_col]
                    ids = values.get("id", [None] * len(texts))
                    for text_value, id_value in zip(texts, ids):
                        if not isinstance(text_value, str) or not text_value.strip():
                            continue
                        doc_id = str(id_value or f"{corpus_id}_{doc_counter:08d}")
                        doc_counter += 1
                        yield {
                            "id": doc_id,
                            "source": corpus_id,
                            "type": doc_type,
                            "license": license_str,
                            "text": text_value
                        }
            elif file_path.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    for content in f:
                        if content.strip():
                            doc_counter += 1
                            yield {
                                "id": f"{corpus_id}_{doc_counter:08d}",
                                "source": corpus_id,
                                "type": doc_type,
                                "license": license_str,
                                "text": content.strip()
                            }

    def verify_val_train_isolation(
        self,
        train_hashes: Dict[str, str],
        val_hashes: Dict[str, str]
    ) -> Tuple[bool, int]:
        """Guarantees 0% exact duplicate leakage between train and validation."""
        collisions = 0
        for h, v_id in val_hashes.items():
            if h in train_hashes:
                collisions += 1
        return (collisions == 0), collisions

    def build_recipe(
        self,
        recipe_id: str,
        output_dir: str,
        data_root: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        mock_corpora: Optional[Dict[str, Iterator[Dict[str, Any]]]] = None,
        val_target_sequences_ratio: float = 0.01
    ) -> Dict[str, Any]:
        """
        Executes complete recipe construction:
        - Shards train into {recipe_id}_train.bin / .idx
        - Shards val into {recipe_id}_val.bin / .idx
        - Strictly isolates source boundaries to guarantee sequence purity.
        """
        recipes = self.config.get("recipes", {})
        if recipe_id not in recipes:
            raise KeyError(f"Recipe '{recipe_id}' not found in configuration. Available: {list(recipes.keys())}")
            
        recipe_cfg = recipes[recipe_id]
        curation_applied = recipe_cfg.get("curation_applied", False)
        
        # Load Tokenizer
        if tokenizer_path is None:
            tokenizer_path = os.path.join(PROJECT_ROOT, "tokenizer", "tokenizer.json")
            
        if os.path.exists(tokenizer_path) and Tokenizer is not None:
            tok = Tokenizer.from_file(tokenizer_path)
            encode_fn = lambda txt: tok.encode(txt).ids
        else:
            if mock_corpora is None:
                raise FileNotFoundError(
                    f"Tokenizer unavailable at {tokenizer_path!r}; refusing to build real shards "
                    "with synthetic token IDs. Install the tokenizers package and provide tokenizer.json."
                )

            # Stable test-only IDs. Never use Python hash(), whose result is salted per process.
            def encode_fn(txt: str) -> List[int]:
                return [
                    int.from_bytes(hashlib.sha256(word.encode("utf-8")).digest()[:4], "little") % 1000 + 10
                    for word in txt.split()
                ]
            
        benchmark_entries = self._load_benchmark_entries()
        decontam = DecontaminationAuditor(benchmark_entries) if benchmark_entries else None
        pipeline = DataCurationPipeline(benchmark_entries=benchmark_entries) if curation_applied else None
        
        os.makedirs(output_dir, exist_ok=True)
        train_writer = BinaryShardWriter(os.path.join(output_dir, f"{recipe_id}_train"), sequence_length=self.seq_len)
        val_writer = BinaryShardWriter(os.path.join(output_dir, f"{recipe_id}_val"), sequence_length=self.seq_len)
        
        train_packer = SourceAwareSequencePacker(train_writer, sequence_length=self.seq_len, eos_token_id=5)
        val_packer = SourceAwareSequencePacker(val_writer, sequence_length=self.seq_len, eos_token_id=5)
        
        manifest_path = os.path.join(output_dir, f"{recipe_id}_manifest.jsonl")
        manifest_file = open(manifest_path, "w", encoding="utf-8")
        
        train_hashes: Dict[str, str] = {}
        val_hashes: Dict[str, str] = {}
        
        mixture = recipe_cfg.get("mixture", {})
        mixture_keys = list(mixture.keys())
        source_summary: Dict[str, Any] = {}
        
        try:
            for s_idx, (source_id, alloc) in enumerate(mixture.items()):
                target_seqs = int(alloc.get("sequences_4k", 0))
                val_target_seqs = max(1, int(target_seqs * val_target_sequences_ratio))
                is_last_source = (s_idx == len(mixture_keys) - 1)
                
                train_packer.start_source(source_id, target_sequences=target_seqs, split="train")
                val_packer.start_source(source_id, target_sequences=val_target_seqs, split="val")
                
                if mock_corpora and source_id in mock_corpora:
                    doc_iter = mock_corpora[source_id]
                else:
                    doc_iter = self.get_corpus_iterator(source_id, data_root=data_root)
                    
                docs_processed = 0
                train_accepted = 0
                val_accepted = 0
                
                for raw_doc in doc_iter:
                    docs_processed += 1
                    raw_doc.setdefault("source", source_id)
                    raw_doc.setdefault("type", self.config.get("corpora_sources", {}).get(source_id, {}).get("doc_type", "general_web"))
                    raw_doc.setdefault("license", self.config.get("corpora_sources", {}).get(source_id, {}).get("license", "Unknown"))
                    doc_id = raw_doc["id"]
                    doc_text = raw_doc["text"]
                    
                    # Step 1: Pre-curation split assignment (1% held-out)
                    assigned_split = self.splitter.assign_split(source_id, doc_id)
                    
                    if assigned_split == "val":
                        # Check benchmark decontamination on validation
                        if decontam and decontam.check_contamination(doc_text)[0]:
                            continue
                        clean_val, _ = pipeline.normalizer.normalize(doc_text, doc_type=raw_doc.get("type", "general_web")) if pipeline else (doc_text, {})
                        val_hashes[hashlib.sha256(clean_val.encode("utf-8")).hexdigest()] = doc_id
                        toks = encode_fn(clean_val)
                        val_packer.add_document(toks)
                        val_accepted += 1
                    else:
                        # Training sample processing
                        if curation_applied:
                            meta = pipeline.process_document(raw_doc)
                            manifest_file.write(json.dumps(meta.__dict__, ensure_ascii=False) + "\n")
                            if meta.split != "train":
                                continue
                            clean_text, _ = pipeline.normalizer.normalize(doc_text, doc_type=raw_doc.get("type", "general_web"))
                        else:
                            # Recipe A: Baseline minimal hygiene (naive TTR >= 0.20)
                            words = doc_text.split()
                            if len(words) < 20:
                                continue
                            if (len(set(words)) / max(1, len(words))) < 0.20:
                                continue
                            clean_text = doc_text
                            
                        train_hashes[hashlib.sha256(clean_text.encode("utf-8")).hexdigest()] = doc_id
                        toks = encode_fn(clean_text)
                        is_fulfilled = train_packer.add_document(toks)
                        train_accepted += 1
                        
                        if is_fulfilled and val_packer.emitted_for_source >= val_target_seqs:
                            break
                            
                # Finish source with strict boundary isolation
                train_src_stats = train_packer.finish_source(
                    emit_remainder=is_last_source and self.target_tokens % self.seq_len > 0,
                    remainder_token_count=(self.target_tokens % self.seq_len) if is_last_source else 0,
                    allow_incomplete=bool(mock_corpora)
                )
                val_src_stats = val_packer.finish_source(
                    emit_remainder=False,
                    allow_incomplete=bool(mock_corpora)
                )
                
                source_summary[source_id] = {
                    "docs_processed": docs_processed,
                    "train_docs_accepted": train_accepted,
                    "val_docs_accepted": val_accepted,
                    "train_sequences_emitted": train_src_stats["emitted_sequences"],
                    "val_sequences_emitted": val_src_stats["emitted_sequences"]
                }
            # Verify cross-split isolation before finalizing shard headers.
            is_isolated, collisions = self.verify_val_train_isolation(train_hashes, val_hashes)
            if not is_isolated:
                raise RuntimeError(f"Detected {collisions} leaking documents between train and validation pools!")

            train_stats = train_writer.close()
            val_stats = val_writer.close()
        except BaseException:
            train_writer.abort()
            val_writer.abort()
            raise
        finally:
            active_iterator = locals().get("doc_iter")
            if active_iterator is not None and hasattr(active_iterator, "close"):
                active_iterator.close()
            manifest_file.close()
        
        result_stats = {
            "recipe_id": recipe_id,
            "target_tokens": self.target_tokens,
            "train_shards": train_stats,
            "val_shards": val_stats,
            "source_summary": source_summary,
            "train_val_isolated": is_isolated,
            "manifest_file": manifest_path
        }
        
        stats_path = os.path.join(output_dir, f"{recipe_id}_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(result_stats, f, indent=2, ensure_ascii=False)
            
        return result_stats


# ==============================================================================
# 6. VERIFICATION TEST SUITE
# ==============================================================================

def run_dataset_builder_verification():
    print("=" * 82)
    print("  HETMAN-2.0B: DATASET BUILDER & MMAP SHARD VERIFICATION SUITE")
    print("=" * 82)

    # 1. Test DocumentSplitter determinism and 1% split ratio
    splitter = DocumentSplitter(fraction_val=0.01, seed=42)
    val_count = 0
    total_test_docs = 10_000
    for i in range(total_test_docs):
        sp = splitter.assign_split("ubertext_2.0", f"doc_{i:06d}")
        if sp == "val":
            val_count += 1
            
    val_pct = (val_count / total_test_docs) * 100.0
    assert 0.7 <= val_pct <= 1.3, f"Validation split ratio out of bounds: {val_pct}%"
    print(f"[OK] DocumentSplitter verified: {val_count} / {total_test_docs} docs assigned to val ({val_pct:.2f}% ~ 1.00%)")

    # 2. Test BinaryShardWriter and MMapIndexedDataset round-trip
    test_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "test_shards")
    os.makedirs(test_dir, exist_ok=True)
    shard_base = os.path.join(test_dir, "test_shard_4k")
    
    writer = BinaryShardWriter(shard_base, sequence_length=4096)
    np.random.seed(42)
    
    num_test_seqs = 20
    original_seqs = []
    for s_idx in range(num_test_seqs):
        toks = list(np.random.randint(10, 1000, size=4096))
        src = f"source_{s_idx % 3}"
        original_seqs.append((toks, src))
        writer.add_sequence(toks, source_id=src, split="train")
        
    writer_stats = writer.close()
    print(f"[OK] BinaryShardWriter closed: {writer_stats['total_sequences']} sequences ({writer_stats['total_tokens']:,} tokens)")
    
    dataset = MMapIndexedDataset(shard_base)
    assert len(dataset) == num_test_seqs, f"Dataset length mismatch: {len(dataset)} vs {num_test_seqs}"
    
    for s_idx in range(num_test_seqs):
        item = dataset[s_idx]
        expected_toks, expected_src = original_seqs[s_idx]
        assert np.array_equal(item["input_ids"], np.array(expected_toks, dtype=np.uint32)), f"Token mismatch at seq {s_idx}"
        assert item["token_count"] == 4096, "Token count mismatch"
        assert item["attention_mask"].sum() == 4096, "Mask sum mismatch"
        assert item["source_crc"] == zlib.crc32(expected_src.encode("utf-8")), "CRC mismatch"
        assert item["split"] == "train", "Split mismatch"
        
    print(f"[OK] MMapIndexedDataset verified: Zero-copy read matches original tokens bit-for-bit across {num_test_seqs} sequences.")

    # 3. Test SourceAwareSequencePacker strict boundary isolation
    test_pack_base = os.path.join(test_dir, "test_boundary_isolation")
    b_writer = BinaryShardWriter(test_pack_base, sequence_length=4096)
    b_packer = SourceAwareSequencePacker(b_writer, sequence_length=4096)
    
    # Source A: feed 5000 tokens (target: 1 sequence = 4096 tokens, leftover 904 tokens)
    b_packer.start_source("source_A", target_sequences=1, split="train")
    b_packer.add_document([100] * 2500)
    b_packer.add_document([100] * 2500)
    b_packer.finish_source(emit_remainder=False)
    
    # Source B: feed 5000 tokens (target: 1 sequence = 4096 tokens)
    b_packer.start_source("source_B", target_sequences=1, split="train")
    b_packer.add_document([200] * 2500)
    b_packer.add_document([200] * 2500)
    b_packer.finish_source(emit_remainder=True, remainder_token_count=512)
    b_writer.close()
    
    b_dataset = MMapIndexedDataset(test_pack_base)
    assert len(b_dataset) == 3, f"Expected 3 sequences (2 full + 1 remainder), got {len(b_dataset)}"
    
    seq0 = b_dataset[0] # Source A
    seq1 = b_dataset[1] # Source B
    seq2 = b_dataset[2] # Source B remainder
    
    # Sequence 0 must be 100% source A (tokens=100 with BOS=1, EOS=5)
    assert seq0["source_crc"] == zlib.crc32(b"source_A")
    assert all(tok in (1, 100, 5) for tok in seq0["input_ids"][:4096]), "Source A sequence contaminated!"
    
    # Sequence 1 must be 100% source B (tokens=200 with BOS=1, EOS=5, NOT polluted by source A's leftover 904 tokens!)
    assert seq1["source_crc"] == zlib.crc32(b"source_B")
    assert all(tok in (1, 200, 5) for tok in seq1["input_ids"][:4096]), "Source B sequence contaminated with Source A leftover!"
    
    # Sequence 2 must be remainder with count=512
    assert seq2["source_crc"] == zlib.crc32(b"source_B")
    assert seq2["token_count"] == 512
    assert seq2["attention_mask"].sum() == 512
    print("[OK] SourceAwareSequencePacker verified: Zero cross-source buffer carryover; sequence tags 100% pure.")

    # 4. Test DatasetRecipeBuilder end-to-end recipe build with mock data
    builder = DatasetRecipeBuilder()
    mock_corpora = {
        "ubertext_2.0": [
            {"id": f"ub_{i}", "source": "ubertext_2.0", "type": "general_web", "license": "CC-BY",
             "text": f"Українська преса та веб-новини про технології та інновації номер {i}. " * 15}
            for i in range(100)
        ],
        "fineweb_edu": [
            {"id": f"fw_{i}", "source": "fineweb_edu", "type": "educational", "license": "ODC-By",
             "text": f"Educational high-quality scientific text about physics and chemistry number {i}. " * 15}
            for i in range(100)
        ],
        "laws_ua": [
            {"id": f"law_{i}", "source": "laws_ua", "type": "legal", "license": "Public Domain",
             "text": f"Стаття {i}. Верховна Рада України приймає закони відповідно до Конституції України. " * 15}
            for i in range(100)
        ]
    }
    
    recipe_output_dir = os.path.join(test_dir, "recipe_a_test_run")
    recipe_res = builder.build_recipe(
        recipe_id="recipe_a_baseline",
        output_dir=recipe_output_dir,
        mock_corpora=mock_corpora,
        val_target_sequences_ratio=0.05
    )
    
    assert os.path.exists(os.path.join(recipe_output_dir, "recipe_a_baseline_train.bin"))
    assert os.path.exists(os.path.join(recipe_output_dir, "recipe_a_baseline_train.idx"))
    assert os.path.exists(os.path.join(recipe_output_dir, "recipe_a_baseline_val.bin"))
    assert os.path.exists(os.path.join(recipe_output_dir, "recipe_a_baseline_val.idx"))
    assert recipe_res["train_val_isolated"] is True
    print(f"[OK] DatasetRecipeBuilder end-to-end verified: Created shards for train & val with 0% split leakage.")

    print("=" * 82)
    print("  ALL DATASET BUILDER & MMAP SHARD TESTS PASSED SUCCESSFULLY!")
    print("=" * 82)


if __name__ == "__main__":
    run_dataset_builder_verification()
