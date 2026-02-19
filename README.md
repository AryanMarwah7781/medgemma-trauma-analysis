---
title: MedGemma Trauma Analysis
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
---

# MedGemma Trauma Analysis

AI-powered multi-model pipeline for CT hemorrhage detection, quantification, and clinical decision support.

## Pipeline

```
N CT slices + vitals → MedSigLIP triage → MedGemma 1.5 visual analysis
  → U-Net quantification → EAST-aligned report → SSE streaming Q&A
```

## Models Used

- `google/medgemma-1.5-4b-it` — visual CT analysis + report synthesis
- `google/medsiglip-448` — zero-shot per-slice triage scoring
- U-Net ResNet34 — voxel-level hemorrhage segmentation

## Setup (HF Spaces)

Set the following secrets in Space settings:

| Secret | Value |
|---|---|
| `HF_TOKEN` | Your HuggingFace token (read access to gated models) |
| `LORA_ADAPTER` | `AryanMarwah/medgemma-trauma-lora` (optional) |

**Note:** First startup takes ~10-15 minutes to download models.

## Local / Colab Usage

```bash
git clone https://github.com/AryanMarwah7781/medgemma-trauma-analysis
cd medgemma-trauma-analysis
pip install -r requirements.txt

export HF_TOKEN=hf_your_token
export PORT=5000              # optional, defaults to 7860
export LORA_ADAPTER=AryanMarwah/medgemma-trauma-lora  # optional

python app.py
```

For Colab with ngrok tunnel:
```bash
USE_NGROK=true NGROK_TOKEN=your_token python app.py
```

## Hackathon

Google HAI-DEF Hackathon 2026 — Deadline Feb 24, 2026
