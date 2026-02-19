# MedGemma Trauma Analysis

## Project Name
**MedGemma Trauma Analysis** — Multi-Model AI Pipeline for CT Hemorrhage Detection and Clinical Decision Support

---

## Your Team
- **Aryan Marwah** — Developer & Researcher

---

## Problem Statement

In trauma emergencies, internal abdominal hemorrhage is one of the leading causes of preventable death. Rapid, accurate assessment of CT angiograms is critical — but radiologist availability is limited in high-volume trauma centers, especially overnight and in under-resourced settings.

**Challenges:**
- Manual CT interpretation takes 15–30 minutes per patient
- Hemorrhage volume quantification is subjective and inconsistent
- Time-to-intervention directly impacts mortality
- Busy EDs lack dedicated trauma radiologist coverage 24/7

**Our goal:** An AI co-pilot that triages CT scans in seconds, quantifies hemorrhage volume precisely, and generates EAST guideline-aligned clinical reports — enabling emergency physicians and surgeons to make faster, more informed decisions.

---

## Overall Solution

We built a five-layer, multi-model pipeline that takes raw CT slices + patient vitals and outputs a complete structured radiology report with clinical recommendations.

### Pipeline Architecture

```
N CT slices + vitals (HR / BP / GCS)
        │
        ▼
Layer 1: MedSigLIP-448 — zero-shot per-slice hemorrhage triage
        │  Ranks and filters to top suspicious slices
        ▼
Layer 2: MedGemma 1.5 4b-it — multi-image CT volume interpretation
        │  Identifies injury pattern, organs, bleeding description, severity
        ▼
Layer 3: U-Net (ResNet34) — voxel-level hemorrhage segmentation
        │  Stacks 2D masks → 3D volume → mL quantification
        ▼
Layer 4: MedGemma 1.5 (reused) — EAST guideline-aligned report synthesis
        │  Combines all findings into a structured clinical report
        ▼
Layer 5: SSE Streaming Q&A — real-time clinician follow-up
         Session-aware answers about the specific scan
```

---

## Technical Details

### Models

| Model | Role | Notes |
|---|---|---|
| `google/medgemma-1.5-4b-it` | Visual CT analysis + report synthesis | Released Jan 13, 2026; native multi-image support |
| `google/medsiglip-448` | Zero-shot triage scoring per slice | 850M param contrastive encoder |
| U-Net + ResNet34 | Hemorrhage segmentation | `segmentation-models-pytorch`; pre-trained on ImageNet |

**LoRA Fine-Tuning:** MedGemma 1.5 was further fine-tuned on the RSNA 2023 Abdominal Trauma Detection dataset using LoRA (rank 16, target: q/k/v/o projections). This specializes the model for CT hemorrhage findings in JSON-structured output, improving visual analysis accuracy. The adapter is loaded optionally via `LORA_ADAPTER` env var.

### Clinical Guidelines Integration

- **EAST (Eastern Association for the Surgery of Trauma)** practice management guidelines for abdominal hemorrhage are embedded into the report synthesis prompt and recommendation logic.
- **ATLS shock classification** (Class I–IV by volume) contextualizes hemorrhage quantification against clinical thresholds.

### Key Engineering Decisions

1. **Model reuse (GPU memory efficiency):** MedGemma is loaded once and shared between Layer 2 (visual analysis) and Layer 4 (report synthesis). MedSigLIP runs on CPU, freeing the full GPU for the 4B model.

2. **4-bit quantization (BitsAndBytes):** Enables running MedGemma 1.5 4B on a T4 GPU (15 GB VRAM) with minimal quality loss.

3. **LoRA adapter scoping:** The LoRA adapter is disabled (`model.disable_adapter()`) for text-only report synthesis, since it was fine-tuned on image→JSON tasks and causes token loops on long-form prose generation.

4. **Prompt-continuation for synthesis:** The report synthesis prompt pre-fills `CLINICAL INDICATION` + `FINDINGS` headers, so MedGemma writes content directly without exhausting tokens on reasoning preamble.

5. **SSE streaming Q&A:** After analysis, clinicians can ask follow-up questions about the specific scan via a streaming Server-Sent Events endpoint — powered by `TextIteratorStreamer` for real-time token output.

### Tech Stack

- **Backend:** Python 3.10, Flask, PyTorch 2.x
- **Models:** HuggingFace Transformers, BitsAndBytes (4-bit), PEFT (LoRA)
- **Segmentation:** segmentation-models-pytorch (U-Net + ResNet34)
- **Training:** RSNA 2023 Abdominal Trauma Detection dataset
- **Deployment:** Google Colab (T4/A100) with optional ngrok tunnel for live demo

### Repository

- **GitHub:** https://github.com/AryanMarwah7781/medgemma-trauma-analysis
- **Dataset:** RSNA 2023 Abdominal Trauma Detection (HuggingFace: `jherng/rsna-2023-abdominal-trauma-detection`)

---

## What the System Outputs

Given CT slices, the pipeline returns:

1. **Triage summary** — per-slice MedSigLIP hemorrhage scores, suspicious slice count
2. **Visual findings** — injury pattern, organs involved, bleeding description, severity estimate, differential diagnosis
3. **Quantification** — hemorrhage volume in mL, ATLS shock class, risk tier (LOW / MODERATE / HIGH)
4. **Full radiology report** — CLINICAL INDICATION, FINDINGS, IMPRESSION, EAST guideline recommendations, differential diagnosis, follow-up plan
5. **Session-aware Q&A** — streaming answers to clinician questions about the specific scan

---

## Impact

- **Speed:** Full pipeline runs in ~15–30 seconds per case on a T4 GPU
- **Coverage:** Handles 1 to N CT slices simultaneously (multi-slice volume analysis)
- **Clinical safety:** Template fallback ensures a complete structured report is always generated, even if LLM inference encounters issues
- **Transparency:** All AI findings are labeled; report includes disclaimer for radiologist verification
- **Guideline compliance:** EAST + ATLS standards embedded — not just findings, but actionable recommendations

---

*Submitted for Google HAI-DEF Hackathon 2026*
