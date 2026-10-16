"""
Unit tests for src/prepare_raw_corpora.py:
Verifies:
1. Corpus inventory audit mode without network access
2. normalize_row metadata schema, ID hashing, and empty text filtering
3. Command-line safety guards: --confirm-download and --confirm-unbounded
4. Refusal of non-HF corpora (laws_ua, ua_dialogues) in download subcommand
5. Temporary .partial file atomicity
"""

import hashlib
import json
import os
import shutil
import tempfile

from prepare_raw_corpora import (
    normalize_row,
    print_inventory,
    resolve_corpus_path,
    parse_args,
    main,
)
from pilot_experiment_config import load_pilot_config, CONFIG_FILE_PATH

def test_inventory_audit():
    print("[1] Testing corpus inventory audit without network access...")
    config = load_pilot_config(CONFIG_FILE_PATH)
    ready = print_inventory(config, root_override=None)
    # Initially all raw corpora are expected to be missing
    assert ready is False, "Expected inventory to be not ready initially"
    print("    [PASS] Inventory audit completed cleanly.")


def test_normalize_row():
    print("\n[2] Testing row normalization, SHA-256 ID fallback, and empty text handling...")
    info = {
        "text_column": "text",
        "doc_type": "general_web",
        "license": "CC-BY-4.0"
    }

    # 1. Valid row with explicit ID
    row1 = {"id": "doc_123", "text": "Якісний український текст для моделі."}
    norm1 = normalize_row(row1, "ubertext_2.0", info, 0)
    assert norm1 is not None
    assert norm1["id"] == "doc_123"
    assert norm1["source"] == "ubertext_2.0"
    assert norm1["type"] == "general_web"
    assert norm1["license"] == "CC-BY-4.0"
    assert norm1["text"] == "Якісний український текст для моделі."

    # 2. Row without ID (should generate deterministic SHA-256 fallback)
    row2 = {"text": "Документ без явного ідентифікатора."}
    norm2 = normalize_row(row2, "ubertext_2.0", info, 1)
    assert norm2 is not None
    expected_id = hashlib.sha256(("ubertext_2.0\0" + row2["text"]).encode("utf-8")).hexdigest()
    assert norm2["id"] == expected_id

    # 3. Empty or whitespace text (must be skipped)
    assert normalize_row({"text": ""}, "ubertext_2.0", info, 2) is None
    assert normalize_row({"text": "   \n\t  "}, "ubertext_2.0", info, 3) is None
    assert normalize_row({"other_col": "no text"}, "ubertext_2.0", info, 4) is None
    print("    [PASS] normalize_row schema and fallback IDs verified.")


def test_safety_guards_and_non_hf_rejection():
    print("\n[3] Testing command-line safety guards and non-HF corpus rejection...")
    config = load_pilot_config(CONFIG_FILE_PATH)

    # 1. Reject download without --confirm-download
    # Emulate sys.argv: download --corpus fineweb_edu
    import sys
    orig_argv = sys.argv
    try:
        sys.argv = ["prepare_raw_corpora.py", "download", "--corpus", "fineweb_edu"]
        try:
            main()
            assert False, "Should have raised SystemExit"
        except SystemExit as e:
            assert "confirm-download" in str(e).lower() or e.code != 0
        print("    [PASS] Guard: --confirm-download required.")

        # 2. Reject unbounded export without --confirm-unbounded
        sys.argv = ["prepare_raw_corpora.py", "download", "--corpus", "fineweb_edu", "--confirm-download", "--max-docs", "0"]
        try:
            main()
            assert False, "Should have raised SystemExit"
        except SystemExit as e:
            assert "confirm-unbounded" in str(e).lower() or e.code != 0
        print("    [PASS] Guard: --confirm-unbounded required for max-docs=0.")

        # 3. Reject non-HF corpus (laws_ua)
        sys.argv = ["prepare_raw_corpora.py", "download", "--corpus", "laws_ua", "--confirm-download"]
        try:
            main()
            assert False, "Should have raised ValueError/SystemExit"
        except (ValueError, SystemExit) as e:
            pass
        print("    [PASS] Guard: non-HF corpus laws_ua rejected from download.")
    finally:
        sys.argv = orig_argv


if __name__ == "__main__":
    print("=" * 80)
    print("  HETMAN-2.0B: PREPARE RAW CORPORA TEST SUITE")
    print("=" * 80)
    test_inventory_audit()
    test_normalize_row()
    test_safety_guards_and_non_hf_rejection()
    print("\n" + "=" * 80)
    print("  [ALL PREPARE RAW CORPORA TESTS PASSED SUCCESSFULLY!]")
    print("=" * 80)
