# RSNA Abdominal Trauma Dataset

This folder stores the dataset for MedGemma Trauma Analysis.

## Quick Start

### Option 1: HuggingFace (Recommended for MVP)
```bash
cd data
python download_huggingface.py
```
- **Size:** ~3.91 GB
- **Samples:** 206 CT scans with segmentation masks
- **Format:** NIfTI (preprocessed)

### Option 2: Kaggle (Full dataset)
```bash
# First, set up Kaggle credentials:
# 1. Go to https://www.kaggle.com/account
# 2. Click "Create New API Token"
# 3. Save to ~/.kaggle/kaggle.json

python download_kaggle.py
```
- **Size:** ~90 GB
- **Samples:** 4711 CT scans
- **Format:** DICOM

## Data Format

The preprocessed NIfTI files have:
- Resampled voxel spacing: (2.0, 2.0, 3.0)
- Contains: liver, spleen, kidney segmentation + active extravasation masks
