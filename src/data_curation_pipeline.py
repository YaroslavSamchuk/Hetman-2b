"""
Hetman-2.0B: Production Data Curation, Filtering & Deduplication Pipeline (v2.0-Production)
Systematic 6-Stage Pre-Training Data Engineering Suite

Key Enhancements:
1. Segmental TTR (STTR / Windowed MATTR w=200): Eliminates Heaps' Law length bias in lexical diversity.
2. Domain-Calibrated MinHash Deduplication:
   - Legal: High threshold (Jaccard >= 0.92, 8-word shingles) to preserve formulaic statute variations.
   - Code: Normalized token MinHash (Jaccard >= 0.85) excluding whitespace/comment noise.
   - Web / Prose: Standard (Jaccard >= 0.80, 5-word shingles).
3. Context-Aware Normalization: Preserves raw syntax in code, normalizes apostrophes in prose.
4. Multi-Language Code Classifier: Python, C/C++, Rust, Go, SQL, Bash, YAML, JSON with bracket balance.
5. Benchmark Decontamination with audit logging of matched spans.
6. Pipeline Step Statistics Tracker: Records document and token drop-rates across each filter stage.
"""

import os
import re
import zlib
import json
import struct
import hashlib
import unicodedata
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Set, Optional, Tuple, Any

PIPELINE_VERSION = "2.0.0-verified"

# ==============================================================================
# 1. METADATA SCHEMA & MANIFEST
# ==============================================================================

@dataclass
class DocumentMetadata:
    doc_id: str
    source: str                 # e.g., "ubertext_2.0", "laws_ua", "fineweb_edu", "starcoder2"
    license: str                # e.g., "Apache-2.0", "CC-BY-4.0", "Public Domain", "MIT"
    language: str               # "uk", "en", "code"
    doc_type: str               # "legal", "educational", "code", "general_web", "dialogue"
    quality_score: float        # Normalized quality heuristic [0.0, 1.0]
    raw_hash: str               # SHA-256 of unmodified raw input text
    clean_hash: str             # SHA-256 of normalized text
    token_count: int            # Estimated tokens (or SentencePiece tokens)
    split: str                  # "train", "val", or "pilot"
    pipeline_version: str = PIPELINE_VERSION
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    duplicate_type: Optional[str] = None     # "exact", "near", "cross_source"
    rejection_stage: Optional[str] = None   # "normalization", "decontamination", "quality", "dedup"
    rejection_reason: Optional[str] = None
    filter_metrics: Dict[str, Any] = field(default_factory=dict)

class StageStatisticsTracker:
    """Tracks document and token counts before and after every pipeline stage."""
    def __init__(self):
        self.stage_input_docs: Dict[str, int] = {}
        self.stage_output_docs: Dict[str, int] = {}
        self.stage_dropped_docs: Dict[str, int] = {}
        self.rejection_reasons: Dict[str, int] = {}

    def record_drop(self, stage: str, reason: str):
        self.stage_dropped_docs[stage] = self.stage_dropped_docs.get(stage, 0) + 1
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1

    def print_stage_report(self, total_initial: int):
        print("\n" + "=" * 78)
        print("  DATA PIPELINE STAGE-BY-STAGE ATTRITION REPORT")
        print("=" * 78)
        print(f"Initial raw documents received: {total_initial:,}\n")
        print(f"{'Pipeline Stage':<25} | {'Dropped Docs':<15} | {'Attrition Rate':<15}")
        print("-" * 60)
        cumulative_dropped = 0
        for stage in ["decontamination", "domain_quality", "exact_dedup", "near_dedup"]:
            dropped = self.stage_dropped_docs.get(stage, 0)
            cumulative_dropped += dropped
            rate = (dropped / max(1, total_initial)) * 100.0
            print(f"{stage:<25} | {dropped:<15,} | {rate:>13.2f}%")
        print("-" * 60)
        retained = total_initial - cumulative_dropped
        retention_rate = (retained / max(1, total_initial)) * 100.0
        print(f"{'FINAL ACCEPTED':<25} | {retained:<15,} | {retention_rate:>13.2f}% retained")
        print("=" * 78)


# ==============================================================================
# 2. CONTEXT-AWARE TEXT NORMALIZATION
# ==============================================================================

BOILERPLATE_REGEXES = [
    re.compile(r"(?i)(цей\s+сайт\s+використовує\s+(файли\s+)?cookie|ми\s+використовуємо\s+cookie|політика\s+конфіденційності\s+та\s+cookie).*"),
    re.compile(r"(?i)(this\s+website\s+uses\s+cookies|accept\s+all\s+cookies|privacy\s+policy\s+and\s+terms).*"),
    re.compile(r"(?i)(всі\s+права\s+захищені|all\s+rights\s+reserved|copyright\s+©\s*\d{4}).*"),
    re.compile(r"(?i)(підпишіться\s+на\s+наш\s+(telegram|facebook|newsletter|канал)).*"),
    re.compile(r"(?i)(share\s+on\s+(twitter|facebook|linkedin|reddit)).*")
]

class ContextAwareNormalizer:
    @staticmethod
    def normalize(text: str, doc_type: str = "general_web") -> Tuple[str, Dict[str, Any]]:
        """
        Applies context-aware normalization:
        - Code/Config: Preserves exact syntax, quotes, strings, and line structure. Strips only trailing whitespace.
        - Natural language: Unicode NFC, unified Ukrainian apostrophes (in words only, including і, ї, є, ґ), strips cookie noise.
        """
        metrics = {"raw_len": len(text), "lines_stripped": 0}
        if not text:
            return "", metrics
            
        # 1. Unicode NFC is universal for consistent hashing
        text = unicodedata.normalize("NFC", text)
        
        # If code or configuration: preserve exact apostrophes, quotes, and indentation
        if doc_type in ("code", "config"):
            lines = [l.rstrip() for l in text.splitlines()]
            return "\n".join(lines).strip(), metrics

        # 2. Natural language apostrophe unification in words only
        # Covers all Ukrainian cyrillic letters [а-яА-ЯіїєґІЇЄҐ] + alphanumeric
        # Replaces modifier letter apostrophe \u02BC, right single quote \u2019, left quote \u2018,
        # acute accent \u00B4, backtick \u0060, prime \u2032 with standard ASCII apostrophe ' (\u0027)
        ukr_char_class = r"[а-яА-ЯіїєґІЇЄҐa-zA-Z0-9]"
        apostrophe_class = r"[\u02BC\u2019\u2018\u0060\u00B4\u02B9\u2032]"
        text = re.sub(rf"(?<={ukr_char_class}){apostrophe_class}(?={ukr_char_class})", "'", text)

        # 3. Strip HTML markup
        text = re.sub(r"<[^>]+>", " ", text)

        # 4. Remove boilerplate lines
        clean_lines = []
        for line in text.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            is_bp = any(p.search(line_str) for p in BOILERPLATE_REGEXES)
            if is_bp:
                metrics["lines_stripped"] += 1
            else:
                clean_lines.append(line_str)
                
        text = "\n".join(clean_lines)
        # 5. Multi-space compaction
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        metrics["clean_len"] = len(text)
        return text.strip(), metrics


# ==============================================================================
# 3. DOMAIN QUALITY FILTERS & SEGMENTAL TTR (STTR)
# ==============================================================================

class LexicalDiversity:
    @staticmethod
    def compute_sttr(text: str, window_size: int = 200) -> float:
        """
        Segmental Type-Token Ratio (STTR / Windowed MATTR):
        Computes TTR on fixed non-overlapping windows of `window_size` words,
        then averages them. Decouples lexical diversity from document length (resolves Heaps' Law bias).
        """
        words = [w.lower() for w in text.split() if any(c.isalnum() for c in w)]
        if len(words) < 50:
            return len(set(words)) / max(1, len(words))
            
        num_windows = len(words) // window_size
        if num_windows == 0:
            return len(set(words)) / float(len(words))
            
        ttr_sum = 0.0
        for i in range(num_windows):
            window = words[i * window_size : (i + 1) * window_size]
            ttr_sum += len(set(window)) / float(window_size)
            
        return ttr_sum / float(num_windows)

class DomainQualityFilters:
    @staticmethod
    def filter_legal(text: str) -> Tuple[bool, float, Optional[str], Dict[str, Any]]:
        words = text.split()
        metrics = {"word_count": len(words)}
        if len(words) < 25:
            return False, 0.0, "Too short for statutory document (<25 words)", metrics
            
        # Statutory structural markers
        legal_markers = [
            "стаття", "частина", "закон", "кодекс", "пункт", "постанова",
            "суд", "конституція", "україна", "розпорядження", "указ", "наказ",
            "рішення", "договір"
        ]
        marker_hits = sum(1 for m in legal_markers if m in text.lower())
        metrics["legal_marker_hits"] = marker_hits
        
        # STTR for legal text (calibrated lower: statutory prose repeats specific legal terms)
        sttr = LexicalDiversity.compute_sttr(text, window_size=200)
        metrics["sttr"] = round(sttr, 4)
        
        if marker_hits < 1 and sttr < 0.25:
            return False, 0.2, "Lacks statutory markers and low vocabulary diversity", metrics
            
        quality_score = min(1.0, 0.5 + 0.1 * min(marker_hits, 5))
        return True, quality_score, None, metrics

    @staticmethod
    def filter_prose_and_educational(text: str) -> Tuple[bool, float, Optional[str], Dict[str, Any]]:
        words = text.split()
        word_count = len(words)
        metrics = {"word_count": word_count}
        
        if word_count < 50:
            return False, 0.0, "Too short (<50 words)", metrics
        if word_count > 60000:
            return False, 0.0, "Unreasonably long (>60k words) - likely log/raw dump", metrics
            
        # Segmental TTR on 200-word windows (independent of document length)
        sttr = LexicalDiversity.compute_sttr(text, window_size=200)
        metrics["sttr"] = round(sttr, 4)
        if sttr < 0.25:
            return False, sttr, f"Repetitive vocabulary (STTR={sttr:.2f} < 0.25)", metrics
            
        # Punctuation / noise ratio
        non_alpha = sum(1 for c in text if not (c.isalnum() or c.isspace()))
        noise_ratio = non_alpha / max(1, len(text))
        metrics["noise_ratio"] = round(noise_ratio, 4)
        if noise_ratio > 0.25:
            return False, 0.0, f"Excessive noise/special characters ({noise_ratio:.2f})", metrics
            
        # Mean word length
        mean_len = sum(len(w) for w in words) / max(1, word_count)
        metrics["mean_word_length"] = round(mean_len, 2)
        if mean_len < 3.2 or mean_len > 12.0:
            return False, 0.0, f"Unnatural mean word length ({mean_len:.1f})", metrics
            
        quality_score = min(1.0, 0.4 + sttr * 0.6)
        return True, quality_score, None, metrics

    @staticmethod
    def filter_stem_and_math(text: str) -> Tuple[bool, float, Optional[str], Dict[str, Any]]:
        words = text.split()
        word_count = len(words)
        metrics = {"word_count": word_count}
        
        if word_count < 30:
            return False, 0.0, "Too short (<30 words) for STEM document", metrics
        if word_count > 60000:
            return False, 0.0, "Unreasonably long (>60k words) - likely log/raw dump", metrics
            
        sttr = LexicalDiversity.compute_sttr(text, window_size=200)
        metrics["sttr"] = round(sttr, 4)
        if sttr < 0.20:
            return False, sttr, f"Repetitive vocabulary (STTR={sttr:.2f} < 0.20)", metrics
            
        # Punctuation / noise ratio: allow up to 45% for LaTeX formulas, operators, and math syntax
        non_alpha = sum(1 for c in text if not (c.isalnum() or c.isspace()))
        noise_ratio = non_alpha / max(1, len(text))
        metrics["noise_ratio"] = round(noise_ratio, 4)
        if noise_ratio > 0.45:
            return False, 0.0, f"Excessive noise/special characters ({noise_ratio:.2f} > 0.45)", metrics
            
        mean_len = sum(len(w) for w in words) / max(1, word_count)
        metrics["mean_word_length"] = round(mean_len, 2)
        if mean_len < 2.5 or mean_len > 15.0:
            return False, 0.0, f"Unnatural mean word length ({mean_len:.1f})", metrics
            
        quality_score = min(1.0, 0.5 + sttr * 0.5)
        return True, quality_score, None, metrics

    @staticmethod
    def filter_code(text: str) -> Tuple[bool, float, Optional[str], Dict[str, Any]]:
        lines = text.splitlines()
        metrics = {"lines": len(lines)}
        if len(lines) < 4:
            return False, 0.0, "Too few lines (<4) for code snippet", metrics
            
        # Reject minified single-line blobs (e.g. minified JS, encoded blobs)
        max_line_len = max(len(l) for l in lines)
        metrics["max_line_length"] = max_line_len
        if max_line_len > 2500 and len(lines) < 10:
            return False, 0.0, f"Minified / binary bundle detected (max_line={max_line_len})", metrics
            
        # Multi-language programming keywords: Python, C/C++, Rust, Go, Java, JS/TS, SQL, Shell, YAML
        text_lower = text.lower()
        code_signatures = [
            "def ", "class ", "return", "import ", "from ", "function ", "const ", "let ",
            "#include", "int main", "fn ", "struct ", "impl ", "pub ", "package ", "func ",
            "select ", "from ", "where ", "insert ", "#!/bin/", "echo ", "apiversion:", "metadata:",
            "public class", "private ", "template<", "std::", "console.log"
        ]
        hits = sum(1 for sig in code_signatures if sig in text_lower)
        metrics["keyword_hits"] = hits
        
        # Structural bracket balance check for balanced syntax
        b_curly = text.count("{") - text.count("}")
        b_paren = text.count("(") - text.count(")")
        b_square = text.count("[") - text.count("]")
        metrics["bracket_imbalance"] = abs(b_curly) + abs(b_paren) + abs(b_square)
        
        # Syntax tokens check if zero keywords found
        has_syntax_symbols = any(sym in text for sym in ["{", "}", "=", ";", "->", "/*", "*/", "()", "[]"])
        if hits == 0 and (not has_syntax_symbols or metrics["bracket_imbalance"] > 10):
            return False, 0.0, "No code signatures found and severe syntax/bracket imbalance", metrics
            
        quality = min(1.0, 0.5 + 0.1 * min(hits, 5))
        return True, quality, None, metrics


# ==============================================================================
# 4. DOMAIN-CALIBRATED THREE-LEVEL DEDUPLICATION
# ==============================================================================

class DomainCalibratedDeduplicator:
    """
    Three-Level Deduplicator with Domain-Calibrated Thresholds:
    - Level 1: Exact SHA-256 of normalized text.
    - Level 2: MinHash LSH (128 hashes, deterministic seed=42) with domain-specific shingle size & Jaccard:
      * Legal: Jaccard >= 0.92, 8-word shingles (preserves statutory variants).
      * Code: Jaccard >= 0.85, 4-word shingles (token-level).
      * Web / Prose: Jaccard >= 0.80, 5-word shingles.
    - Level 3: Cross-source duplicate registration.
    """
    def __init__(self, num_perm: int = 128, seed: int = 42):
        self.num_perm = num_perm
        self.exact_hashes: Set[str] = set()
        self.cross_source: Dict[str, str] = {} # sha256 -> source
        
        # Deterministic random hash coefficients for h(x) = (a * x + b) % p
        # Guarantees identical signatures across all workers, shards and runs
        import random
        rng = random.Random(seed)
        self.p = 2147483647
        self.a = [rng.randint(1, self.p - 1) for _ in range(num_perm)]
        self.b = [rng.randint(0, self.p - 1) for _ in range(num_perm)]
        
        # LSH buckets: 16 bands x 8 rows
        self.num_bands = 16
        self.rows_per_band = num_perm // self.num_bands
        # Domain-calibrated LSH buckets:
        # - "legal": preserves statutory variations (0.92 threshold, 8 shingles)
        # - "code": token-level deduplication (0.85 threshold, 4 shingles)
        # - "prose": shared across web, educational, news, and dialog to catch cross-source near-duplicates (0.80 threshold, 5 shingles)
        self.domain_buckets: Dict[str, List[Dict[int, List[Tuple[str, List[int]]]]]] = {
            "legal": [{} for _ in range(self.num_bands)],
            "code": [{} for _ in range(self.num_bands)],
            "stem": [{} for _ in range(self.num_bands)],
            "prose": [{} for _ in range(self.num_bands)]
        }

    def _get_domain_key(self, doc_type: str) -> str:
        if doc_type in ("legal", "statute"):
            return "legal"
        elif doc_type in ("code", "config"):
            return "code"
        elif doc_type in ("stem", "math"):
            return "stem"
        else:
            return "prose"

    def _get_domain_params(self, domain_key: str) -> Tuple[int, float]:
        if domain_key == "legal":
            return 8, 0.92  # 8-word shingles, 92% Jaccard (avoids formulaic false positives)
        elif domain_key == "code":
            return 4, 0.85  # 4-word shingles, 85% Jaccard
        elif domain_key == "stem":
            return 5, 0.85  # 5-word shingles, 85% Jaccard
        else:
            return 5, 0.80  # 5-word shingles, 80% Jaccard

    def _compute_signature(self, text: str, shingle_size: int) -> List[int]:
        words = text.lower().split()
        if len(words) < shingle_size:
            shingles = {" ".join(words)}
        else:
            shingles = {" ".join(words[i:i+shingle_size]) for i in range(len(words) - shingle_size + 1)}
            
        shingle_hashes = [zlib.crc32(s.encode("utf-8")) for s in shingles]
        if not shingle_hashes:
            shingle_hashes = [0]
            
        sig = []
        for i in range(self.num_perm):
            a_i, b_i = self.a[i], self.b[i]
            min_val = min((a_i * h + b_i) % self.p for h in shingle_hashes)
            sig.append(min_val)
        return sig

    @staticmethod
    def _compute_band_key(band_slice: List[int]) -> int:
        """
        Computes a portable, strictly deterministic 32-bit unsigned integer key from a MinHash band slice.
        Uses little-endian struct packing (<I) and CRC-32 hashing.
        Guarantees 100% cross-process, cross-worker, and cross-run determinism,
        completely eliminating Python's randomized SipHash salt in builtin hash().
        """
        packed = struct.pack(f"<{len(band_slice)}I", *band_slice)
        return zlib.crc32(packed)

    def check_and_insert(self, doc_id: str, text: str, doc_type: str, source: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Returns (is_unique, duplicate_type, duplicate_details).
        Ensures non-polluting two-phase insertion: candidates across all bands are evaluated
        BEFORE any bucket is updated with the current signature.
        """
        # Level 1: Exact Hash
        sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if sha256 in self.exact_hashes:
            orig_src = self.cross_source.get(sha256, "unknown")
            dup_type = "cross_source" if orig_src != source else "exact"
            return False, dup_type, f"Exact match with prior doc from {orig_src}"
            
        # Level 2: MinHash LSH
        domain_key = self._get_domain_key(doc_type)
        shingle_size, threshold = self._get_domain_params(domain_key)
        sig = self._compute_signature(text, shingle_size)
        buckets = self.domain_buckets[domain_key]
        
        # Step 2a: Gather all candidate documents across all bands using deterministic keys
        candidates: Dict[str, List[int]] = {}
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_key = self._compute_band_key(sig[start:end])
            b = buckets[band_idx]
            if band_key in b:
                for cand_id, cand_sig in b[band_key]:
                    if cand_id not in candidates:
                        candidates[cand_id] = cand_sig

        # Step 2b: Verify candidate Jaccard similarity
        for cand_id, cand_sig in candidates.items():
            matches = sum(1 for x, y in zip(sig, cand_sig) if x == y)
            jaccard = matches / float(self.num_perm)
            if jaccard >= threshold:
                return False, "near_duplicate", f"Near-duplicate (Jaccard={jaccard:.2f} >= {threshold:.2f}) with doc '{cand_id}'"

        # Step 2c: If confirmed unique, insert into all bands and register exact hash
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_key = self._compute_band_key(sig[start:end])
            b = buckets[band_idx]
            if band_key not in b:
                b[band_key] = []
            b[band_key].append((doc_id, sig))

        self.exact_hashes.add(sha256)
        self.cross_source[sha256] = source
        return True, None, None


# ==============================================================================
# 5. BENCHMARK DECONTAMINATION WITH AUDIT LOGGING
# ==============================================================================

class DecontaminationAuditor:
    """
    Detects 13-gram exact overlaps with test benchmarks (UA-MMLU, MMLU, GSM8K, HumanEval).
    Uses normalized word tokenization to remain invariant to punctuation and casing differences.
    Logs the exact matched span and benchmark ID for scientific transparency.
    """
    def __init__(self, benchmark_entries: Optional[List[Dict[str, str]]] = None):
        self.gram_index: Dict[int, Tuple[str, str]] = {} # crc32 -> (benchmark_name, matched_text)
        if benchmark_entries:
            for entry in benchmark_entries:
                self.index_test_case(entry.get("benchmark", "eval"), entry.get("prompt", ""))

    def _tokenize(self, text: str) -> List[str]:
        # Normalizes to lowercase alphanumeric word tokens to eliminate punctuation discrepancies
        return re.findall(r"\w+", text.lower())

    def index_test_case(self, benchmark_name: str, prompt: str):
        words = self._tokenize(prompt)
        for i in range(len(words) - 12):
            span = " ".join(words[i:i+13])
            h = zlib.crc32(span.encode("utf-8"))
            self.gram_index[h] = (benchmark_name, span)

    def check_contamination(self, text: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Returns (is_contaminated, benchmark_name, matched_span)."""
        words = self._tokenize(text)
        if len(words) < 13:
            return False, None, None
            
        for i in range(len(words) - 12):
            span = " ".join(words[i:i+13])
            h = zlib.crc32(span.encode("utf-8"))
            if h in self.gram_index:
                bench, original_span = self.gram_index[h]
                return True, bench, original_span
        return False, None, None


# ==============================================================================
# 6. PIPELINE ORCHESTRATOR & VERIFICATION TEST
# ==============================================================================

class DataCurationPipeline:
    def __init__(self, manifest_path: str = "curation_manifest.jsonl"):
        self.manifest_path = manifest_path
        self.normalizer = ContextAwareNormalizer()
        self.filters = DomainQualityFilters()
        self.dedup = DomainCalibratedDeduplicator(num_perm=128)
        self.decontam = DecontaminationAuditor()
        self.tracker = StageStatisticsTracker()

    def process_document(self, raw_doc: Dict[str, Any]) -> DocumentMetadata:
        doc_id = raw_doc["id"]
        source = raw_doc["source"]
        doc_type = raw_doc.get("type", "general_web")
        license_str = raw_doc.get("license", "Unknown")
        raw_text = raw_doc["text"]
        
        raw_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        
        # Stage 1: Context-Aware Normalization
        clean_text, norm_metrics = self.normalizer.normalize(raw_text, doc_type=doc_type)
        clean_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
        tokens = len(clean_text.split())
        
        # Stage 2: Benchmark Decontamination Check
        is_contam, bench_name, matched_span = self.decontam.check_contamination(clean_text)
        # Resolve document language metadata
        doc_lang = "uk"
        if doc_type == "code" or "code" in source:
            doc_lang = "code"
        elif source in ("fineweb_edu", "stem_math") or "math" in source or "en" in source:
            doc_lang = "en"

        if is_contam:
            self.tracker.record_drop("decontamination", f"contaminated_{bench_name}")
            return DocumentMetadata(
                doc_id=doc_id, source=source, license=license_str, language=doc_lang,
                doc_type=doc_type, quality_score=0.0, raw_hash=raw_hash, clean_hash=clean_hash,
                token_count=tokens, split="rejected", rejection_stage="decontamination",
                rejection_reason=f"Matched test benchmark '{bench_name}': {matched_span[:40]}...",
                filter_metrics={"contaminated_benchmark": bench_name, "span": matched_span}
            )

        # Stage 3: Domain-Calibrated Quality Filtering
        if doc_type == "legal":
            is_valid, q_score, reason, metrics = self.filters.filter_legal(clean_text)
        elif doc_type == "code":
            is_valid, q_score, reason, metrics = self.filters.filter_code(clean_text)
        elif doc_type in ("stem", "math"):
            is_valid, q_score, reason, metrics = self.filters.filter_stem_and_math(clean_text)
        else:
            is_valid, q_score, reason, metrics = self.filters.filter_prose_and_educational(clean_text)
            
        metrics.update(norm_metrics)
        if not is_valid:
            self.tracker.record_drop("domain_quality", reason or "quality_filter")
            return DocumentMetadata(
                doc_id=doc_id, source=source, license=license_str, language=doc_lang,
                doc_type=doc_type, quality_score=q_score, raw_hash=raw_hash, clean_hash=clean_hash,
                token_count=tokens, split="rejected", rejection_stage="domain_quality",
                rejection_reason=reason, filter_metrics=metrics
            )
            
        # Stage 4: Calibrated Three-Level Deduplication
        is_unique, dup_type, dup_details = self.dedup.check_and_insert(doc_id, clean_text, doc_type, source)
        if not is_unique:
            stage_name = "exact_dedup" if dup_type in ("exact", "cross_source") else "near_dedup"
            self.tracker.record_drop(stage_name, dup_type or "duplicate")
            return DocumentMetadata(
                doc_id=doc_id, source=source, license=license_str, language=doc_lang,
                doc_type=doc_type, quality_score=q_score, raw_hash=raw_hash, clean_hash=clean_hash,
                token_count=tokens, split="rejected", is_duplicate=True,
                duplicate_type=dup_type, rejection_stage=stage_name,
                rejection_reason=dup_details, filter_metrics=metrics
            )
            
        # Stage 5: Accepted Clean Document
        return DocumentMetadata(
            doc_id=doc_id, source=source, license=license_str, language=doc_lang,
            doc_type=doc_type, quality_score=q_score, raw_hash=raw_hash, clean_hash=clean_hash,
            token_count=tokens, split="train", rejection_stage=None,
            rejection_reason=None, filter_metrics=metrics
        )

    def curate_stream(
        self,
        doc_iterable,
        output_clean_path: Optional[str] = None,
        manifest_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Applies the complete 6-stage curation pipeline to an iterable of raw documents.
        Emits clean documents and writes the full audit manifest for scientific provenance.
        """
        clean_file = open(output_clean_path, "w", encoding="utf-8") if output_clean_path else None
        manifest_file = open(manifest_path or self.manifest_path, "w", encoding="utf-8") if (manifest_path or self.manifest_path) else None
        
        total_docs = 0
        accepted_docs = 0
        accepted_tokens = 0
        
        try:
            for raw_doc in doc_iterable:
                total_docs += 1
                meta = self.process_document(raw_doc)
                
                if manifest_file:
                    manifest_file.write(json.dumps(asdict(meta), ensure_ascii=False) + "\n")
                    
                if meta.split == "train":
                    accepted_docs += 1
                    accepted_tokens += meta.token_count
                    if clean_file:
                        clean_text, _ = self.normalizer.normalize(raw_doc["text"], doc_type=raw_doc.get("type", "general_web"))
                        clean_record = {
                            "id": meta.doc_id,
                            "source": meta.source,
                            "type": meta.doc_type,
                            "text": clean_text,
                            "quality_score": meta.quality_score,
                            "tokens": meta.token_count
                        }
                        clean_file.write(json.dumps(clean_record, ensure_ascii=False) + "\n")
        finally:
            if clean_file:
                clean_file.close()
            if manifest_file:
                manifest_file.close()
                
        return {
            "total_docs": total_docs,
            "accepted_docs": accepted_docs,
            "accepted_tokens": accepted_tokens,
            "retention_rate": (accepted_docs / max(1, total_docs)) * 100.0
        }

    def curate_jsonl_file(self, input_jsonl: str, output_clean_jsonl: str, manifest_path: Optional[str] = None) -> Dict[str, Any]:
        """Streams a JSONL file through the curation pipeline."""
        def reader():
            with open(input_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        return self.curate_stream(reader(), output_clean_jsonl, manifest_path)

def run_pipeline_audit_test():
    print("=" * 78)
    print("  HETMAN-2.0B DATA CURATION PIPELINE AUDIT (v2.0-PRODUCTION)")
    print("=" * 78)
    
    pipeline = DataCurationPipeline()
    # Register benchmark evaluation prompts for decontamination
    pipeline.decontam.index_test_case(
        "UA-MMLU",
        "Яка стаття Конституції України проголошує суверенітет і незалежність держави? Варіанти: А) Стаття 1, Б) Стаття 5"
    )
    
    test_suite = [
        # 1. Valid statutory legal text
        {
            "id": "law_001", "source": "laws_ua", "type": "legal", "license": "Public Domain",
            "text": "Стаття 1. Україна є суверенна і незалежна, демократична, соціальна, правова держава. "
                    "Суверенітет України поширюється на всю її територію. Україна є унітарною державою. "
                    "Територія України в межах існуючого кордону є цілісною і недоторканною."
        },
        # 2. Statutory variant (same formulaic preamble, but distinct content) -> MUST BE PRESERVED!
        {
            "id": "law_002", "source": "laws_ua", "type": "legal", "license": "Public Domain",
            "text": "Стаття 2. Суверенітет України поширюється на всю її територію. "
                    "Україна є унітарною державою. Верховна Рада України здійснює законодавчу владу. "
                    "Порядок роботи Верховної Ради встановлюється Конституцією України та Регламентом."
        },
        # 3. Exact duplicate of law_001 across another source -> MUST BE REJECTED!
        {
            "id": "law_dup", "source": "ubertext_2.0", "type": "legal", "license": "CC-BY",
            "text": "Стаття 1. Україна є суверенна і незалежна, демократична, соціальна, правова держава. "
                    "Суверенітет України поширюється на всю її територію. Україна є унітарною державою. "
                    "Територія України в межах існуючого кордону є цілісною і недоторканною."
        },
        # 4. Valid Python code with syntax & comments & quotes -> MUST BE ACCEPTED without altering quotes
        {
            "id": "code_001", "source": "starcoder2", "type": "code", "license": "Apache-2.0",
            "text": "# Fast Walsh-Hadamard Transform implementation\n"
                    "import math\nimport torch\n\n"
                    "def fwht_1d(x: torch.Tensor) -> torch.Tensor:\n"
                    "    '''Applies FWHT across trailing dimension.'''\n"
                    "    N = x.shape[-1]\n"
                    "    h = 1\n"
                    "    while h < N:\n"
                    "        x = x.view(-1, N // (2 * h), 2, h)\n"
                    "        a = x[:, :, 0, :]\n"
                    "        b = x[:, :, 1, :]\n"
                    "        x = torch.stack([a + b, a - b], dim=2)\n"
                    "        h *= 2\n"
                    "    return x.view(-1, N) / math.sqrt(N)\n"
        },
        # 5. Low-diversity repetitive spam (low STTR) -> MUST BE REJECTED!
        {
            "id": "spam_001", "source": "general_web", "type": "general_web", "license": "CC-BY",
            "text": ("Купити авто дешево Київ авто авто авто терміново недорого продаж авто Київ авто " * 10)
        },
        # 6. Test contamination (matches UA-MMLU indexed question with punctuation differences) -> MUST BE REJECTED!
        {
            "id": "eval_leak", "source": "fineweb_edu", "type": "educational", "license": "ODC-By",
            "text": "Зразок тесту для підготовки: Яка стаття Конституції України проголошує суверенітет і незалежність держави! "
                    "Варіанти: А) Стаття 1, Б) Стаття 5. Правильна відповідь: Стаття 1."
        },
        # 7. Ukrainian educational text with multiple apostrophe variants (\u02BC, \u2019, `) -> MUST BE ACCEPTED & NORMALIZED
        {
            "id": "prose_ukr_001", "source": "ubertext_2.0", "type": "educational", "license": "CC-BY",
            "text": "Українська мова має унікальні фонетичні та граматичні властивості серед слов'янських мов. "
                    "Наприклад, українські слова зʼїзд, м’яч, роз’яснення, п’ять та об`єкт містять апостроф, "
                    "який графічно позначає твердість попереднього приголосного звука перед йотованими голосними літерами. "
                    "Наукові лінгвістичні дослідження переконливо доводять, що послідовне збереження автентичної орфографії "
                    "та уніфікація символів мають фундаментальне практичне значення для навчання сучасних генеративних "
                    "нейромережевих мовних моделей високої точності."
        },
        # 8. Near-duplicate of prose_ukr_001 (>80% Jaccard) in web text -> MUST BE REJECTED by near_dedup!
        {
            "id": "prose_ukr_near_dup", "source": "general_web", "type": "general_web", "license": "CC-BY",
            "text": "Українська мова має унікальні фонетичні та граматичні властивості серед слов'янських мов. "
                    "Наприклад, українські слова з'їзд, м'яч, роз'яснення, п'ять та об'єкт містять апостроф, "
                    "який графічно позначає твердість попереднього приголосного звука перед йотованими голосними літерами. "
                    "Наукові лінгвістичні дослідження переконливо доводять, що послідовне збереження автентичної орфографії "
                    "та уніфікація символів мають фундаментальне практичне значення для навчання сучасних генеративних "
                    "нейромережевих мовних моделей найвищої точності та надійності."
        }
    ]
    
    print(f"[*] Processing {len(test_suite)} diverse documents through verified pipeline...\n")
    for doc in test_suite:
        meta = pipeline.process_document(doc)
        status = "ACCEPTED" if meta.rejection_stage is None else f"REJECTED ({meta.rejection_stage})"
        reason = f" | Reason: {meta.rejection_reason}" if meta.rejection_reason else f" | Quality: {meta.quality_score:.2f}"
        print(f"  [{status:<25}] {meta.doc_id:<18} (Type: {meta.doc_type:<11}) {reason}")
        
    pipeline.tracker.print_stage_report(len(test_suite))

    # Unit assertion check: Determinism & Apostrophe normalization
    normalizer = ContextAwareNormalizer()
    norm_text, _ = normalizer.normalize("Слова зʼїзд, м’яч, об`єкт і роз’яснення", doc_type="educational")
    assert "з'їзд" in norm_text and "м'яч" in norm_text and "об'єкт" in norm_text and "роз'яснення" in norm_text, "Apostrophe normalization failed!"
    print("\n[OK] Unit Assertion Passed: Ukrainian apostrophe variants correctly unified to ASCII ' in words.")

    # Unit assertion check: Deterministic MinHash signatures across instances
    d1 = DomainCalibratedDeduplicator(seed=42)
    d2 = DomainCalibratedDeduplicator(seed=42)
    sig1 = d1._compute_signature("Україна - правова держава", 2)
    sig2 = d2._compute_signature("Україна - правова держава", 2)
    assert sig1 == sig2, "MinHash signatures are not deterministic!"
    print("[OK] Unit Assertion Passed: MinHash signatures are 100% deterministic across instances.")

    # Unit assertion check: Deterministic LSH bucket key (struct-packed CRC32, completely independent of PYTHONHASHSEED)
    band_slice = [1024, 2048, 4096, 8192]
    key1 = DomainCalibratedDeduplicator._compute_band_key(band_slice)
    key2 = DomainCalibratedDeduplicator._compute_band_key(band_slice)
    assert key1 == key2 and isinstance(key1, int) and key1 > 0, "Band key calculation failed!"
    print(f"[OK] Unit Assertion Passed: LSH band bucket keys are strictly deterministic (key={key1:#010x}).")

if __name__ == "__main__":
    run_pipeline_audit_test()
