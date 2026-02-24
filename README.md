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

> **Five specialized AI models. One pipeline. Structured trauma briefing in under 25 seconds.**

Real-time abdominal CT triage, hemorrhage quantification, and EAST guideline-aligned clinical decision support — built for the HAI-DEF MedGemma Hackathon.

---

## The Problem

Internal hemorrhage is invisible on the outside. A trauma patient can present talking with a normal BP — and be in hemorrhagic shock 40 minutes later. The CT scanner takes under 10 seconds. The bottleneck is everything after: radiology read → verbal handoff → surgical decision. That chain averages **20–45 minutes** in a busy trauma center.

This system delivers a structured clinical briefing — injury pattern, hemorrhage volume in mL, EAST management recommendation, and real-time Q&A — **while the radiologist is still reading**.

---

## Pipeline

```
Uploaded CT slices
       │
       ▼
┌──────────────────────┐
│  Layer 1: MedSigLIP  │  Zero-shot triage — which slices are suspicious?
│  google/medsiglip-448│  Runs on CPU · ~1.2s for 8 slices
└──────────┬───────────┘
           │  suspicious slices only
           ▼
┌──────────────────────┐
│  Layer 2: MedGemma   │  Multi-image CT volume interpretation
│  medgemma-1.5-4b-it  │  Interleaved image+text tokens · 4-bit NF4 · ~8–12s
└──────┬───────┬────────┘
       │       │
       ▼       ▼
┌──────────┐ ┌──────────────────────┐
│ Layer 3  │ │  Layer 4: MedGemma   │
│  U-Net   │ │  Report Synthesis    │
│  ResNet34│ │  EAST-aligned report │
│ ~2s · T4 │ │  + LoRA adapter · ~6s│
└────┬─────┘ └──────────┬───────────┘
     │ volume mL         │ clinical report
     └─────────┬─────────┘
               ▼
     ┌─────────────────────┐
     │  Layer 5: MedGemma  │
     │  Streaming Q&A (SSE)│
     │  ~2s first token    │
     └─────────────────────┘

Total pipeline on T4 GPU: < 25 seconds
```

---

## Models

| Model | Role | Details |
|---|---|---|
| [`google/medsiglip-448`](https://huggingface.co/google/medsiglip-448) | Per-slice triage | Contrastive encoder, 850M params, CPU |
| [`google/medgemma-1.5-4b-it`](https://huggingface.co/google/medgemma-1.5-4b-it) | CT analysis + report synthesis | 4B params, 4-bit NF4, T4 GPU |
| U-Net (ResNet34) | Hemorrhage segmentation | Pixel-level masks → volume in mL |
| [`AryanMarwah/medgemma-trauma-lora`](https://huggingface.co/AryanMarwah/medgemma-trauma-lora) | LoRA fine-tune | Rank-16, trained on RSNA 2023 Abdominal Trauma Detection |

---

## Key Features

- **Zero-shot CT triage** — MedSigLIP scores every slice against 5 clinically specific text candidates (hemoperitoneum, solid organ injury, active bleeding, etc.) and filters before MedGemma ever runs
- **Multi-slice volume reasoning** — Suspicious slices passed as interleaved image+text tokens (Gemma3 native format), not gridded or concatenated
- **Hemorrhage quantification** — U-Net pixel masks stacked into a 3D volume, converted to mL using real CT voxel spacing. Risk levels: LOW (<5 mL) / MODERATE (5–500 mL) / HIGH (>500 mL)
- **EAST guideline alignment** — Report synthesis maps findings to Eastern Association for the Surgery of Trauma management recommendations via LoRA fine-tuned on RSNA 2023 ground truth
- **Real-time streaming Q&A** — `TextIteratorStreamer` + Server-Sent Events. First token in ~2s. Session stored 30 min for multi-turn follow-up without re-running the pipeline
- **VRAM-adaptive** — Slice cap auto-adjusts at startup: T4/V100 (≤16GB) → 3 slices, A10G → 6, A100+ → 10. OOM recovery with cache clear + single-slice fallback

---

## Performance

| Stage | Hardware | Latency |
|---|---|---|
| MedSigLIP triage (8 slices) | CPU | ~1.2s |
| MedGemma visual analysis (3 slices, 4-bit) | T4 | ~8–12s |
| U-Net segmentation (8 slices) | T4 | ~2s |
| Report synthesis | T4 | ~6s |
| **Total pipeline** | **T4** | **< 25s** |
| Q&A first token | T4 | ~2s |
| *Human pipeline (CT → decision)* | *—* | *20–45 min* |

---

## Setup

### HuggingFace Spaces

Set the following secrets in your Space settings:

| Secret | Value |
|---|---|
| `HF_TOKEN` | HuggingFace token with read access to gated models |
| `LORA_ADAPTER` | `AryanMarwah/medgemma-trauma-lora` *(optional — enables RSNA fine-tune)* |

> **Note:** First startup downloads ~8GB of models. Allow 10–15 minutes.

### Local / Colab

```bash
git clone https://github.com/AryanMarwah7781/medgemma-trauma-analysis
cd medgemma-trauma-analysis
pip install -r requirements.txt

export HF_TOKEN=hf_your_token
export LORA_ADAPTER=AryanMarwah/medgemma-trauma-lora  # optional

python app.py
```

Colab with ngrok tunnel:
```bash
USE_NGROK=true NGROK_TOKEN=your_token python app.py
```

---

## Fine-Tuning

The LoRA adapter (`AryanMarwah/medgemma-trauma-lora`) was trained on the **RSNA 2023 Abdominal Trauma Detection** dataset — 3,147 labeled CT volumes with per-organ injury grades and active extravasation annotations. Training used TRL's `SFTTrainer` with instruction-response pairs constructed from RSNA ground truth labels mapped to EAST management criteria.

- Rank: 16, targeting attention Q/V projection layers
- Base model: `google/medgemma-1.5-4b-it`
- Adds ~2% to model size, loads via PEFT at startup

---

## Stack

- **Backend:** Flask, Server-Sent Events
- **ML:** HuggingFace Transformers, BitsAndBytes, PEFT, segmentation_models_pytorch, TRL
- **Frontend:** Vanilla JS, Inter + JetBrains Mono, glassmorphism dark UI
- **Deployment:** HuggingFace Spaces (Dockerfile, T4 GPU)

---

## Disclaimer

This is a research prototype built for the HAI-DEF MedGemma Hackathon. It is **not validated for clinical use**. All models are open-weight. No patient data was used in development or fine-tuning.

---

*Built by Aryan Marwah · HAI-DEF MedGemma Hackathon · February 2026*
