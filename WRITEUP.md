# MedGemma Hemorrhage Quantifier

## 1. Problem Domain

### The Clinical Need
Trauma patients with internal bleeding require rapid, objective assessment in the emergency department. When patients arrive with suspected abdominal trauma, radiologists must:
- Locate areas of active bleeding (extravasation)
- Quantify the volume of blood loss
- Determine if emergency intervention is needed

Current challenges:
- Manual CT interpretation is time-consuming and subjective
- Quantifying hemorrhage volume is imprecise
- Delayed diagnosis can be fatal in severe cases

### Our Solution
An AI-powered system that:
- Automatically analyzes CT angiogram scans
- Detects and segments active bleeding regions
- Calculates precise hemorrhage volume in milliliters
- Generates structured radiology reports with treatment recommendations

---

## 2. Effective Use of HAI-DEF Models

### MedGemma Integration
We leverage Google's MedGemma model (via Ollama) for natural language report generation:
- Input: Segmented CT analysis results (volume, risk level)
- Output: Professional radiology reports with clinical impressions

The LLM provides:
- Structured medical documentation
- Clinically accurate recommendations
- Professional radiology language

### Technical Architecture
```
CT Scan Image
      ↓
Preprocessing (normalization, windowing)
      ↓
U-Net Segmentation (ResNet34 encoder, ImageNet pretrained)
      ↓
Volume Quantification (voxel count → mL)
      ↓
Risk Assessment (LOW/MODERATE/HIGH)
      ↓
MedGemma LLM Report Generation
      ↓
Structured Medical Report
```

---

## 3. Product Feasibility

### Technical Implementation
- **Segmentation**: U-Net with ResNet34 encoder (pre-trained on ImageNet)
- **LLM**: MedGemma 4B (instruction-tuned) running locally via Ollama
- **Frontend**: Flask web application
- **Deployment**: Runs locally on Jetson Orin Nano (ARM64)

### Performance
- Real-time inference on edge device
- No cloud dependency (privacy-preserving)
- Works offline after model download

### Code Availability
All code is open-source and reproducible:
- Dataset: RSNA Abdominal Trauma Detection (Kaggle)
- Models: segmentation-models-pytorch + Ollama
- Web app: Flask + vanilla JavaScript

---

## 4. Impact Potential

### Clinical Benefits
- **Faster triage**: Automated analysis in seconds vs. minutes
- **Objective quantification**: Precise volume measurements
- **Decision support**: AI-generated recommendations assist clinicians
- **Access**: Can run on edge devices in resource-limited settings

### Real-World Applications
- Emergency departments
- Trauma centers
- Ambulances/EMS
- Developing countries with limited radiology resources

---

## 5. Execution & Communication

### Demo Video Contents
1. Introduction to the problem (30 sec)
2. System demonstration (90 sec)
3. Technical overview (30 sec)
4. Conclusion (30 sec)

### Code Organization
```
medgemma-trauma-analysis/
├── app.py                    # Flask web application
├── src/
│   ├── data_loader.py       # CT data loading
│   ├── unet_model.py         # Segmentation model
│   ├── quantification.py     # Volume calculation
│   └── medgemma_report.py   # LLM report generation
├── templates/
│   └── index.html           # Web UI
└── requirements.txt          # Dependencies
```

### Reproducibility
All dependencies are documented. Run with:
```bash
pip install -r requirements.txt
python app.py
```

---

## 6. Future Work
- Fine-tune segmentation model on medical data
- Multi-organ injury detection
- Integration with PACS systems
- Mobile app development

---

*Submitted for Google HAI-DEF Hackathon 2026*
