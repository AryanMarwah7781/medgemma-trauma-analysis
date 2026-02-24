# Building a Real-Time Trauma AI: Five Models, One Pipeline, Under 25 Seconds

*By Aryan Marwah — MedGemma Trauma Analysis*

---

## The Problem Nobody Talks About

Abdominal trauma kills people slowly. The internal hemorrhage from a liver laceration or splenic rupture is invisible on the outside. A patient walks into the ED talking, has a normal BP — and 40 minutes later is in hemorrhagic shock in the OR.

The bottleneck is not the CT scanner. Modern CT takes under 10 seconds to acquire a full abdominal series. The bottleneck is everything that happens after: a radiologist reads the study, dictates findings, a verbal handoff reaches the trauma surgeon, and a management decision is made. In a busy trauma center, this sequential, human-dependent chain takes **20–45 minutes**. The EAST guidelines for abdominal trauma are clear on what the evidence says about delay: every minute without hemorrhage control increases mortality in grade IV–V solid organ injuries.

The unmet need isn't better diagnosis — radiologists are excellent. The unmet need is **speed and structured communication**. An AI system that can triage suspicious CT slices in seconds, quantify hemorrhage volume to the milliliter, generate a guideline-aligned clinical report, and answer follow-up questions in real time doesn't replace the radiologist. It gives the trauma team a structured briefing while they wait for the full read.

That's exactly what this system does. And it does it in under 25 seconds.

---

## Architecture Overview: Five Specialized Agents

The core design principle is that **no single model does everything**. Just as a trauma team has roles — the trauma surgeon, the radiologist, the anesthesiologist — the pipeline has five specialized agents, each with a defined responsibility, passing structured outputs downstream.

```
Uploaded CT slices
       │
       ▼
┌──────────────────────────────┐
│  Layer 1: MedSigLIP-448      │  Zero-shot triage — which slices are suspicious?
│  google/medsiglip-448        │  Runs on CPU in ~1.2s
└──────────────┬───────────────┘
               │  suspicious slices only
               ▼
┌──────────────────────────────┐
│  Layer 2: MedGemma 1.5       │  Multi-image CT volume interpretation
│  medgemma-1.5-4b-it          │  Returns structured JSON findings
└──────────────┬───────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
┌──────────────┐   ┌──────────────────────────┐
│  Layer 3     │   │  Layer 4: MedGemma 1.5   │
│  U-Net       │   │  Report Synthesis         │
│  Segmentation│   │  EAST-aligned report      │
└──────┬───────┘   └──────────┬───────────────┘
       │ volume mL             │ clinical report
       └──────────┬────────────┘
                  ▼
      ┌───────────────────────┐
      │  Layer 5: MedGemma    │
      │  Streaming Q&A (SSE)  │
      └───────────────────────┘
```

---

## Layer 1: MedSigLIP-448 — The Triage Gate

Every uploaded CT series starts here. Not every slice contains hemorrhage — most don't. Passing 20 slices to MedGemma 1.5 would blow the context window, waste GPU time, and degrade output quality through distraction.

**MedSigLIP-448** (`google/medsiglip-448`) is a contrastive vision-language encoder, trained specifically on medical image-text pairs. It works like a radiologist doing a quick "windowing scan" — looking for anything suspicious before going slice by slice.

The implementation scores every slice against five clinically specific text candidates:

```python
LABELS = [
    "CT scan showing intraabdominal hemorrhage or active bleeding",
    "CT scan with liver laceration, splenic injury, or solid organ trauma",
    "CT scan showing hemoperitoneum or free fluid in the abdomen",
    "Normal CT scan of the abdomen without hemorrhage or injury",
    "CT scan with bowel perforation or mesenteric injury",
]
```

The suspicious score for each slice is the sum of probabilities across the four positive labels. Slices above threshold (default 0.25) advance to Layer 2. Crucially, even if no slices breach the threshold — meaning a genuinely normal-looking scan — we still pass the top-scoring slice to MedGemma for final confirmation. This prevents false-negative silent passes.

**Why not just use a standard CLIP model?** We tested this. General-purpose CLIP (ViT-L/14) assigns nearly identical scores to hemorrhagic and normal abdominal CT slices because it has no domain knowledge of Hounsfield units, organ anatomy, or what free fluid looks like on CT. MedSigLIP, trained on medical image-text pairs, picks up these signals reliably at zero shot.

**Result:** On an 8-slice series, MedSigLIP reduces the slice count passed to MedGemma from 8 to typically 2–3, cutting MedGemma inference time by 60–80% and preventing context dilution.

---

## Layer 2: MedGemma 1.5 — Multi-Slice Volume Interpretation

This is where the real clinical reasoning happens. **MedGemma 1.5 4B** (`google/medgemma-1.5-4b-it`) is a multimodal language model trained on medical data including CT imaging. Unlike general-purpose VLMs like GPT-4o or LLaVA, it understands:

- What hemorrhage looks like in HU ranges on abdominal CT
- Solid organ anatomy and injury patterns
- How to reason *across* multiple slices as a volume, not just describe each image independently

The suspicious slices from Layer 1 are passed in **Gemma3 native multi-image format** — interleaved as image+text tokens, not concatenated or gridded. This is key: the model sees each slice as a distinct visual input with position context.

```python
content = []
for i, img in enumerate(slices):
    content.append({"type": "image", "image": img.convert("RGB")})
    content.append({"type": "text",  "text": f"[CT Slice {i+1} of {len(slices)}]"})

content.append({"type": "text", "text": (
    f"Evaluate ALL {len(slices)} slices collectively as a single volume...\n"
    "Output ONLY the JSON object below:\n"
    '{"injury_pattern": ..., "organs_involved": [...], ...}'
)})
```

The prompt explicitly instructs MedGemma to reason across the volume as a whole, not slice-by-slice, which is how a radiologist actually reads a CT — building a 3D mental model from 2D slices.

**Structured output:** We force JSON output to make downstream processing reliable. The response parser uses brace-counting to extract valid JSON even when the model wraps it in markdown or narrative text, with a clinical narrative fallback if parsing fails entirely.

**4-bit quantization:** MedGemma 1.5 4B loads with BitsAndBytes NF4 quantization, reducing VRAM from ~8GB to ~4GB. This enables deployment on T4 GPUs (16GB VRAM, free tier on Colab/Kaggle) while preserving nearly full accuracy.

**VRAM-adaptive slice cap:** GPU VRAM is detected at startup, and the maximum number of slices passed to MedGemma is automatically adjusted:

```python
if total_vram_gb < 20:   self.max_qa_slices = 3   # T4/V100
elif total_vram_gb < 32: self.max_qa_slices = 6   # A10G
else:                    self.max_qa_slices = 10  # A100+
```

This prevents OOM crashes gracefully rather than failing the entire request.

---

## Layer 3: U-Net Segmentation — Precision Volume Quantification

MedGemma is excellent at qualitative reasoning but is not a segmentation model. For hemorrhage volume in milliliters — a number surgeons actually use to triage NOM vs. operative management — we need pixel-level precision.

A **ResNet34-backbone U-Net** (segmentation_models_pytorch) runs on all uploaded slices, producing binary hemorrhage masks. The 2D masks are stacked into a 3D volume and converted to milliliters using real CT voxel spacing:

```python
voxel_volume_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0  # mm³ → mL
volume_ml = num_hemorrhage_voxels * voxel_volume_ml
```

Volume thresholds map to clinical risk levels following published trauma literature:
- **< 5 mL** → LOW (minor bleeding, NOM appropriate)
- **5–500 mL** → MODERATE (watch and monitor, possible IR)
- **> 500 mL** → HIGH (emergent operative management)

This gives the surgeon a concrete, actionable number: *"85 mL hemoperitoneum, high risk."*

---

## Layer 4: EAST-Aligned Report Synthesis

With Layer 2 findings and Layer 3 quantification in hand, MedGemma synthesizes a full structured clinical report aligned with **EAST (Eastern Association for the Surgery of Trauma) guidelines** — the standard of care in US trauma centers.

EAST guidelines define specific management thresholds for solid organ injuries by grade. The report synthesizer constructs a context bundle containing all findings and instructs MedGemma to generate a report in the style of a radiology attending — findings, impression, and a specific management recommendation pulled directly from EAST criteria.

This is where the LoRA fine-tuning matters. The base MedGemma model understands CT anatomy, but EAST guideline adherence requires specific knowledge of injury grading schemas (AAST Organ Injury Scales) and their corresponding management algorithms. The LoRA adapter — trained on RSNA 2023 Abdominal Trauma Detection data with EAST-mapped labels — adds this trauma-specialist reasoning layer.

---

## Layer 5: Real-Time Q&A via SSE Streaming

After the initial analysis, the trauma team can ask follow-up questions directly: *"Is there active extravasation?"*, *"What's the liver injury grade?"*, *"Should I call IR or the OR?"*

MedGemma answers in real time using **Server-Sent Events (SSE)** streaming — the first token arrives in ~2 seconds, with the response building word-by-word in the browser.

The technical implementation uses HuggingFace's `TextIteratorStreamer` running in a background thread:

```python
streamer = TextIteratorStreamer(
    self.processor.tokenizer,
    skip_special_tokens=True,
    skip_prompt=True,
)

thread = threading.Thread(target=lambda: self.model.generate(**gen_kwargs))
thread.start()

for token in streamer:
    yield token  # Flask SSE response
```

Each Q&A turn sends the CT images, a structured summary of prior findings, and the new question to MedGemma. The session is stored server-side with a 30-minute TTL, so context is maintained across the conversation without re-running the pipeline.

---

## Fine-Tuning: Teaching MedGemma Trauma Structure

MedGemma 1.5 is a general medical VLM. It knows anatomy and can read CT images, but it wasn't specifically trained to:
1. Output structured JSON for programmatic parsing
2. Apply AAST injury grade schemas systematically
3. Map findings to EAST management recommendations

We trained a **LoRA adapter** (rank-16, targeting attention query/value projection layers) on the **RSNA 2023 Abdominal Trauma Detection** dataset — 3,147 labeled CT volumes with per-organ injury grades and active extravasation annotations.

Training used TRL's `SFTTrainer` with instruction-response pairs constructed from RSNA ground truth:

```python
# Example training pair
instruction = "Analyze these CT slices for abdominal trauma..."
response = json.dumps({
    "injury_pattern": "Grade III liver laceration with perihepatic hematoma",
    "organs_involved": ["liver"],
    "bleeding_description": "Perihepatic hematoma, no active extravasation",
    "severity_estimate": "moderate",
    "differential_diagnosis": ["hepatic laceration", "biloma"]
})
```

The adapter (published at `AryanMarwah/medgemma-trauma-lora`) adds ~2% to model size and loads via PEFT at startup, keeping all fine-tuned knowledge on top of the base model's general medical understanding.

---

## System Design: The Orchestrator

All five layers are coordinated by `TraumaOrchestrator` — a single class that loads all models once at Flask startup and holds them in memory for the server lifetime.

```python
class TraumaOrchestrator:
    def __init__(self, ...):
        self.triager           = MedSigLIPTriager(device="cpu")
        self.visual_analyzer   = MedGemmaVisualAnalyzer(device="auto", use_4bit=True)
        self.segmenter         = PreTrainedSegmenter(encoder="resnet34")
        self.report_synthesizer = ReportSynthesizer(self.visual_analyzer)
        self._sessions         = {}  # UUID → session context
```

Key design decisions:

**MedSigLIP on CPU, MedGemma on GPU.** MedSigLIP at 850M parameters is fast enough on CPU (~1.2s for 10 slices), freeing the entire GPU budget for MedGemma.

**Session store with TTL.** After a pipeline run, the session (images + context) is stored in memory under a UUID for 30 minutes. Q&A requests look up the session by ID, so MedGemma doesn't re-process the CT — it picks up exactly where Layer 2 left off.

**OOM recovery.** Both `analyze()` and `stream_answer()` catch `torch.cuda.OutOfMemoryError` explicitly, clear the cache, and fall back to a single slice before re-attempting. This prevents the server from crashing under memory pressure.

---

## Performance Numbers

| Stage | Hardware | Latency |
|---|---|---|
| MedSigLIP triage (8 slices) | CPU | ~1.2s |
| MedGemma visual analysis (3 slices, 4-bit) | T4 | ~8–12s |
| U-Net segmentation (8 slices) | T4 | ~2s |
| Report synthesis | T4 | ~6s |
| **Total pipeline** | **T4** | **< 25s** |
| Q&A first token | T4 | ~2s |

For reference: a trauma center's current human pipeline (CT → radiology read → verbal handoff → decision) averages 20–45 minutes. Our pipeline delivers a structured briefing in under 25 seconds — while the radiologist is still reading.

---

## Stack

- **Backend:** Flask (Python), Server-Sent Events for streaming
- **ML:** HuggingFace Transformers, BitsAndBytes, PEFT, segmentation_models_pytorch
- **Frontend:** Vanilla JS, Inter + JetBrains Mono, glassmorphism dark UI
- **Deployment:** HuggingFace Spaces (Dockerfile, T4 GPU)
- **Fine-tuning:** TRL SFTTrainer on RSNA 2023 Abdominal Trauma Detection
- **Models:** [`google/medgemma-1.5-4b-it`](https://huggingface.co/google/medgemma-1.5-4b-it), [`google/medsiglip-448`](https://huggingface.co/google/medsiglip-448)
- **LoRA adapter:** [`AryanMarwah/medgemma-trauma-lora`](https://huggingface.co/AryanMarwah/medgemma-trauma-lora)
- **Code:** [github.com/AryanMarwah7781/medgemma-trauma-analysis](https://github.com/AryanMarwah7781/medgemma-trauma-analysis)

---

## What's Next

**Structured reporting standardization.** The current report format mirrors EAST guidelines but isn't yet mapped to structured radiology reporting schemas like RADLEX or ACR templates. A future version could output FHIR-compatible DiagnosticReport resources for direct EHR integration.

**Active extravasation detection.** The current U-Net segments hemorrhage broadly. Distinguishing active arterial extravasation (requires IR immediately) from contained hematoma (may observe) is the highest-value next classification layer.

**Multi-phase CT.** Trauma CTs are often acquired in arterial and portal venous phases. Multi-phase reasoning — comparing enhancement patterns between phases — is something MedGemma's multi-image architecture could theoretically support without architectural changes.

**Prospective validation.** This is a research prototype. Clinical deployment requires prospective validation against radiologist ground truth on de-identified patient data, with sensitivity/specificity analysis against AAST injury grades. That's the real work.

---

*This project was built for the HAI-DEF MedGemma hackathon. All models are open-weight. No patient data was used. Built by Aryan Marwah.*
