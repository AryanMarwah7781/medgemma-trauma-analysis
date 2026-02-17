"""
Download RSNA dataset from Kaggle (full original)
Usage: python download_kaggle.py

Requires Kaggle API credentials:
1. Go to https://www.kaggle.com/account
2. Click "Create New API Token" 
3. Place kaggle.json in ~/.kaggle/kaggle.json
"""

import os
import subprocess

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "kaggle")
COMPETITION = "rsna-2023-abdominal-trauma-detection"

print("=" * 50)
print("Downloading RSNA Abdominal Trauma - Kaggle")
print("=" * 50)

# Check for kaggle credentials
kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
if not os.path.exists(kaggle_json):
    print("\n❌ Kaggle credentials not found!")
    print("Setup:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Click 'Create New API Token'")
    print(f"3. Save to {kaggle_json}")
    print("\nAlternatively, manually download from:")
    print(f"https://www.kaggle.com/competitions/{COMPETITION}/data")
    exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\nDownloading competition data to {OUTPUT_DIR}/...")
print("This is ~90 GB, may take a while...")

# Download competition data
result = subprocess.run(
    ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", OUTPUT_DIR],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print("\n✅ Kaggle dataset downloaded successfully!")
    print(f"Location: {OUTPUT_DIR}")
else:
    print(f"\n❌ Error: {result.stderr}")
