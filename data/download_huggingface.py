"""
Download RSNA dataset from HuggingFace (preprocessed version)
Usage: python download_huggingface.py
"""

from datasets import load_dataset
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "huggingface")

print("=" * 50)
print("Downloading RSNA Abdominal Trauma - HuggingFace")
print("=" * 50)

# Download segmentation dataset (206 CT scans + masks)
print("\nDownloading segmentation dataset (~3.91 GB)...")
ds = load_dataset(
    "jherng/rsna-2023-abdominal-trauma-detection",
    "segmentation",
    streaming=False  # Download to cache
)

print(f"\nDataset loaded!")
print(f"Splits: {ds}")
print(f"Number of samples: {len(ds['train'])}")

# Save dataset info
print(f"\nSaving to {OUTPUT_DIR}/")
ds.save_to_disk(OUTPUT_DIR)

print("\n✅ HuggingFace dataset downloaded successfully!")
print(f"Location: {OUTPUT_DIR}")
