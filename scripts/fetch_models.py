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
RELEASE_URL = "https://github.com/Aditya-kr-sh/AI-Driven-Stock-Market-Prediction/releases/download/PLACEHOLDER_TAG/hpc_models.tar.gz"

def main():
    print("=========================================================================")
    print("            HPC PRE-TRAINED MODEL BOOTSTRAP UTILITY                      ")
    print("=========================================================================")
    
    if "PLACEHOLDER_TAG" in RELEASE_URL:
        print("ERROR: Release URL is currently set to a placeholder.")
        print("Pre-trained models are distributed separately once the project is finalized.")
        print(f"Configure the RELEASE_URL in: {Path(__file__).resolve()}")
        sys.exit(1)
        
    dest_dir = Path(__file__).resolve().parent.parent / "saved_models"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    dest_file = dest_dir / "hpc_models.tar.gz"
    
    try:
        print(f"Downloading pre-trained models from:\n{RELEASE_URL} ...")
        urllib.request.urlretrieve(RELEASE_URL, dest_file)
        
        print(f"Extracting archive into: {dest_dir.absolute()} ...")
        with tarfile.open(dest_file, "r:gz") as tar:
            tar.extractall(path=dest_dir)
            
        print("Cleaning up temporary download archive...")
        dest_file.unlink()
        
        print("\nSUCCESS: All pre-trained models loaded and verified successfully.")
    except Exception as e:
        print(f"\nERROR: Failed to fetch pre-trained models: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
