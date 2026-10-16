"""
Hetman-2.0B: Pilot Experiment Configuration & Dataset Recipes Manager (Step 5)
Provides schema validation, corpus availability audits, token allocation math,
and dry-run manifest inspection for the 200M-token A/B/C pilot training experiment.
"""

import os
import sys
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional

CONFIG_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "pilot_dataset_recipes.json"
)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_pilot_config(config_path: str = CONFIG_FILE_PATH) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Pilot configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def audit_corpus_paths(config: Dict[str, Any], root_override: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Audits the availability of each corpus path on disk, taking into account
    environment variables and optional CLI root overrides.
    """
    env_root = os.getenv(config.get("data_root_env_var", "HETMAN_DATA_ROOT"))
    base_root = root_override or env_root
    
    status_report = {}
    corpora = config.get("corpora_sources", {})
    
    for corpus_id, info in corpora.items():
        rel_path = info.get("local_path", "")
        if os.path.isabs(rel_path):
            effective_path = rel_path
        elif base_root:
            # HETMAN_DATA_ROOT points directly at the corpus root (the directory
            # containing ubertext_2.0/, laws_ua/, etc.), not the project root.
            effective_path = os.path.join(base_root, corpus_id)
        else:
            effective_path = os.path.normpath(os.path.join(PROJECT_ROOT, rel_path))
        exists = os.path.exists(effective_path)
        
        # Check if directory has any jsonl/parquet files
        file_count = 0
        total_size_bytes = 0
        if exists:
            if os.path.isdir(effective_path):
                for root, _, files in os.walk(effective_path):
                    for f in files:
                        if f.endswith((".jsonl", ".parquet", ".txt", ".json")):
                            file_count += 1
                            total_size_bytes += os.path.getsize(os.path.join(root, f))
            else:
                file_count = 1
                total_size_bytes = os.path.getsize(effective_path)
                
        status_report[corpus_id] = {
            "name": info.get("name"),
            "effective_path": effective_path,
            "exists": exists,
            "file_count": file_count,
            "size_mb": round(total_size_bytes / (1024 * 1024), 2),
            "hf_fallback": info.get("hf_dataset_id"),
            "doc_type": info.get("doc_type")
        }
    return status_report

def validate_experiment_config(config: Dict[str, Any]) -> None:
    """Fail early on inconsistent token targets or invalid mixture weights."""
    target = int(config["total_tokens_per_recipe"])
    sequence_length = int(config["sequence_length"])
    for recipe_id, recipe in config.get("recipes", {}).items():
        mixture = recipe.get("mixture", {})
        percentage_sum = sum(float(item["percentage"]) for item in mixture.values())
        token_sum = sum(int(item["tokens"]) for item in mixture.values())
        if abs(percentage_sum - 100.0) > 1e-6:
            raise ValueError(f"{recipe_id}: mixture percentages sum to {percentage_sum}, not 100")
        if token_sum != target or int(recipe.get("target_tokens", -1)) != target:
            raise ValueError(f"{recipe_id}: token allocations must sum exactly to {target:,}")
        sequence_sum = sum(int(item["sequences_4k"]) for item in mixture.values())
        if sequence_sum != int(config["total_sequences_per_recipe"]):
            raise ValueError(
                f"{recipe_id}: rounded source sequence estimates sum to {sequence_sum:,}, "
                f"not {config['total_sequences_per_recipe']:,}"
            )

def print_experiment_specification(config: Dict[str, Any]):
    print("=" * 86)
    print("  HETMAN-2.0B: PILOT EXPERIMENT SPECIFICATION (STEP 5 - 200M TOKENS A/B/C)")
    print("=" * 86)
    print(f"Total tokens per recipe:   {config['total_tokens_per_recipe']:,} tokens")
    print(f"Sequence length:           {config['sequence_length']:,} tokens")
    print(f"Sequences per recipe:      {config['total_sequences_per_recipe']:,} sequences (at 4k)")
    accounting = config.get("token_accounting", {})
    if accounting:
        print(f"Exact token accounting:    {accounting.get('full_sequences'):,} full + {accounting.get('remainder_tokens'):,} remainder tokens")
    print(f"Environment variable:      {config.get('data_root_env_var')} (override data root)")
    print("-" * 86)
    
    print("\n[1] TOKEN ALLOCATION ACROSS RECIPES A, B, AND C:")
    print(f"{'Recipe':<22} | {'Domain/Source':<24} | {'Share (%)':<10} | {'Tokens':<13} | {'Sequences (4k)':<12}")
    print("-" * 86)
    
    recipes = config.get("recipes", {})
    for r_id, r_info in recipes.items():
        r_name = r_id.replace("recipe_", "").upper()
        mixture = r_info.get("mixture", {})
        for idx, (c_id, alloc) in enumerate(mixture.items()):
            prefix = r_name if idx == 0 else ""
            pct = alloc.get("percentage", 0.0)
            tok = alloc.get("tokens", 0)
            seq = alloc.get("sequences_4k", 0)
            print(f"{prefix:<22} | {c_id:<24} | {pct:>8.1f}%  | {tok:>12,} | {seq:>12,}")
        print("-" * 86)

    print("\n[2] CORPUS STORAGE & READINESS AUDIT (ON CURRENT SYSTEM):")
    corpus_status = audit_corpus_paths(config)
    print(f"{'Corpus ID':<18} | {'Status':<12} | {'Files':<7} | {'Size (MB)':<10} | {'Configured Path':<30}")
    print("-" * 86)
    all_ready = True
    for c_id, stat in corpus_status.items():
        st_label = "READY" if stat["exists"] and stat["file_count"] > 0 else "MISSING"
        if st_label == "MISSING":
            all_ready = False
        print(f"{c_id:<18} | {st_label:<12} | {stat['file_count']:<7} | {stat['size_mb']:<10.1f} | {stat['effective_path']:<30}")
    print("=" * 86)
    
    if not all_ready:
        print("[!] Note: Raw input corpora are not currently located at default local paths.")
        print("    Configure local paths by placing data into 'data/raw/<corpus_id>' or")
        print("    set the HETMAN_DATA_ROOT environment variable before running token curation.\n")
    else:
        print("[*] All input corpora detected on disk. Ready for dataset generation.\n")

if __name__ == "__main__":
    cfg = load_pilot_config()
    validate_experiment_config(cfg)
    print_experiment_specification(cfg)
