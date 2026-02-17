# MedGemma Hemorrhage Quantifier

AI-powered system for detecting and quantifying hemorrhaging in trauma patients from CT angiogram scans using MedGemma.

## 🏥 The Problem

When trauma patients arrive in the ER with internal bleeding, doctors need to answer:
1. Where is the bleeding?
2. How much blood has pooled (in ml)?
3. Do we need emergency surgery?

**Current state:** Radiologists manually eyeball CT scans — slow and subjective.

## 💡 Our Solution

AI that analyzes CT angiograms to:
- Detect active bleeding (contrast pooling outside vessels)
- Quantify exact blood volume in milliliters
- Generate structured medical reports with treatment recommendations

## 🔧 Technical Pipeline

```
CT Angiogram → Pre-processing → U-Net Segmentation → Volume Calculation → MedGemma Report
```

### Components

1. **Image Pre-processing** — Windowing, normalization
2. **Segmentation Model** — U-Net trained on RSNA data
3. **Volume Quantification** — Voxel counting → ml conversion
4. **MedGemma** — Generate natural language reports

## 📊 Dataset

[RSNA 2023 Abdominal Trauma Detection](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection)
- 206 CT scans with segmentation masks (HuggingFace preprocessed)
- Detects: liver, spleen, kidney injuries + active extravasation

## 🏆 Competition

Google HAI-DEF Hackathon
- **Deadline:** February 24, 2026
- **Eval:** Execution (30%), MedGemma usage (20%), Feasibility (20%), Problem (15%), Impact (15%)

## 📦 Tech Stack

- **Segmentation:** U-Net / nnUNet
- **LLM:** MedGemma (via Ollama/HuggingFace)
- **Data:** RSNA dataset (HuggingFace)
- **Frontend:** Web app (TBD)

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/AryanMarwah7781/medgemma-trauma-analysis.git
cd medgemma-trauma-analysis

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
medgemma-trauma-analysis/
├── data/                  # Dataset downloads
├── models/                # Trained models
├── src/
│   ├── preprocessing/    # Image processing
│   ├── segmentation/     # U-Net models
│   ├── quantification/   # Volume calculation
│   └── generation/       # MedGemma integration
├── app/                   # Web frontend
├── README.md
└── requirements.txt
```

## 📝 Submission

- Write-up (max 3 pages)
- Code (reproducible)
- Video demo (3 min)

---

Built for the Google HAI-DEF Hackathon 2026
