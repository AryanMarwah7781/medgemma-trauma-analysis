# MedGemma Hemorrhage Quantifier

## Project Name
**MedGemma Hemorrhage Quantifier** — AI-Powered CT Scan Analysis for Trauma Care

---

## Your Team
- **Aryan Marwah** — Developer & Researcher

---

## Problem Statement

Trauma patients with internal abdominal bleeding require immediate medical attention. Currently, radiologists manually review CT angiogram scans to:
- Detect active bleeding (extravasation)
- Estimate blood loss volume
- Determine if surgery is needed

**Challenges:**
- Manual interpretation is slow and subjective
- Quantifying hemorrhage volume is imprecise
- Delayed diagnosis can be fatal

**Impact:** Our solution enables:
- Automated hemorrhage detection in seconds
- Precise volume quantification (in mL)
- Real-time decision support for emergency physicians
- Accessible on edge devices (Jetson Orin Nano)

---

## Overall Solution

We built an AI-powered hemorrhage detection system using **Google's MedGemma** model:

1. **CT scan upload** → Web interface accepts any CT image
2. **Segmentation** → U-Net with ResNet34 encoder detects bleeding regions
3. **Quantification** → Converts pixels to exact milliliter volume
4. **Risk Assessment** → Classifies as LOW/MODERATE/HIGH risk
5. **MedGemma Report** → LLM generates professional radiology reports with recommendations

**Key Feature:** The system uses MedGemma (via Ollama) to generate natural language medical reports, fulfilling the HAI-DEF model requirement.

---

## Technical Details

### Architecture
```
CT Image → U-Net Segmentation → Volume Calc → Risk Assessment → MedGemma LLM → Report
```

### Tech Stack
- **Segmentation**: PyTorch + segmentation-models-pytorch (U-Net, ResNet34 encoder)
- **LLM**: MedGemma 4B (amsaravi/medgemma-4b-it) via Ollama
- **Frontend**: Flask web app (Python)
- **Deployment**: Jetson Orin Nano (ARM64, edge-capable)

### Code & Resources
- **Repository**: https://github.com/AryanMarwah7781/medgemma-trauma-analysis
- **Dataset**: RSNA Abdominal Trauma Detection (Kaggle)
- **Live Demo**: Running on Jetson (Flask app)

### Key Features
- Runs locally (privacy-preserving, no cloud needed)
- Works offline after model download
- Real-time inference on edge devices
- Professional medical report generation

---

*Submitted for Google HAI-DEF Hackathon 2026*
