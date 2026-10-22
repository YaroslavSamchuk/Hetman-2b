"""
Hetman-2.0B: One-Click Hugging Face Uploader
Uploads the complete hf_package directory to https://huggingface.co/YSamchuk/Hetman-2b
"""

import os
import sys
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
        
    api = HfApi(token=token)
    
    print(f"[*] Package directory: {package_dir}")
    print(f"[*] Target repository: https://huggingface.co/{repo_id}")
    
    try:
        api.upload_folder(
            folder_path=package_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message="feat: publish official Hetman-2.0B architecture, tokenizer, and model card"
        )
        print("\n[✓] SUCCESS: All 10 files uploaded to Hugging Face successfully!")
        print(f"    View your repository at: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"\n[!] Upload error: {e}")
        print("\nTip: If you need to authenticate, you can:")
        print("  1. Run: hf auth login")
        print("  2. Or run: python upload_to_hf.py <YOUR_HF_TOKEN>")
        print("  3. Or upload the files directly via the Hugging Face web UI: https://huggingface.co/YSamchuk/Hetman-2b")

if __name__ == "__main__":
    upload()
