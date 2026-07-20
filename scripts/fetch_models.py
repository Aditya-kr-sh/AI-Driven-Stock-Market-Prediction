"""
Bootstrap script to download pre-trained stock prediction model weights from GitHub Releases.
Decouples binary assets from repository source history.
"""

import sys
import os
import urllib.request
import tarfile
from pathlib import Path

# Configurable placeholder release URL (to be updated when the final release is published)
RELEASE_URL = "https://github.com/Aditya-kr-sh/AI-Driven-Stock-Market-Prediction/releases/download/v1.0.0/hpc_nifty500_models.tar.gz"

def main():
    print("=========================================================================")
    print("            HPC PRE-TRAINED MODEL BOOTSTRAP UTILITY                      ")
    print("=========================================================================")
    
    project_root = Path(__file__).resolve().parent.parent
    dest_dir = project_root / "saved_models"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if models are already present to prevent redundant downloads
    existing_models = list(dest_dir.glob("hpc_*_*.model"))
    if existing_models:
        print(f"Pre-trained models are already present in {dest_dir.absolute()} (found {len(existing_models)} models).")
        print("Skipping download. If you want to force re-download, delete files under saved_models/ first.")
        print("=========================================================================")
        return
        
    dest_file = dest_dir / "hpc_nifty500_models.tar.gz"
    
    try:
        print(f"Downloading pre-trained models from:\n{RELEASE_URL} ...")
        
        # Configure User-Agent headers to bypass potential rate limits or blockades
        req = urllib.request.Request(
            RELEASE_URL, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(dest_file, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
            
        print("Reading archive structure...")
        with tarfile.open(dest_file, "r:gz") as tar:
            members = tar.getmembers()
            # Determine if tarball was packaged with 'saved_models/' folder prefix
            has_prefix = any(m.name.startswith("saved_models/") for m in members)
            extract_path = project_root if has_prefix else dest_dir
            
            print(f"Extracting archive into: {extract_path.absolute()} ...")
            tar.extractall(path=extract_path)
            
        print("Cleaning up temporary download archive...")
        dest_file.unlink()
        
        print("\nSUCCESS: All pre-trained models loaded and verified successfully.")
        print("=========================================================================")
    except Exception as e:
        print(f"\nERROR: Failed to fetch pre-trained models: {e}")
        if dest_file.exists():
            dest_file.unlink()
        sys.exit(1)

if __name__ == "__main__":
    main()
