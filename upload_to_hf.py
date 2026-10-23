"""
Hetman-2.0B: One-Click Hugging Face Uploader
Uploads the complete hf_package directory to https://huggingface.co/YSamchuk/Hetman-2b
"""

import os
import sys
import getpass
from huggingface_hub import HfApi

def upload():
    repo_id = "YSamchuk/Hetman-2b"
    package_dir = os.path.join(os.path.dirname(__file__), "hf_package")
    
    print("=" * 70)
    print(f"  HETMAN-2.0B: UPLOADING TO HUGGING FACE ({repo_id})")
    print("=" * 70)
    
    token = os.environ.get("HF_TOKEN")
    if not token and len(sys.argv) > 1:
        token = sys.argv[1]
    if not token:
        try:
            token = getpass.getpass("[*] Enter Hugging Face Access Token (hidden, or set HF_TOKEN env var): ")
        except Exception:
            token = None
        
    api = HfApi(token=token if token else None)
    
    print(f"[*] Package directory: {package_dir}")
    print(f"[*] Target repository: https://huggingface.co/{repo_id}")
    
    try:
        # 1. Ensure repository exists
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print("[✓] Repository verified / created successfully.")

        # 2. Upload complete model package excluding local caches
        api.upload_folder(
            folder_path=package_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message="feat: publish official Hetman-2.0B architecture, tokenizer, and model card",
            ignore_patterns=["**/__pycache__/**", "*.pyc", ".git*", "*.tmp"]
        )
        print("\n[✓] SUCCESS: All model files uploaded to Hugging Face successfully!")
        print(f"    View your repository at: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"\n[!] Upload error: {e}")
        print("\nTip: If you need to authenticate, you can:")
        print("  1. Run: hf auth login (or huggingface-cli login)")
        print("  2. Or set: $env:HF_TOKEN=\"hf_...\" (PowerShell) / export HF_TOKEN=\"hf_...\" (Bash)")
        print("  3. Or upload the files directly via the Hugging Face web UI: https://huggingface.co/YSamchuk/Hetman-2b")

if __name__ == "__main__":
    upload()
