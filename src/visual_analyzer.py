"""
MedGemma 1.5 Visual Analyzer
==============================
Uses google/medgemma-1.5-4b-it (HAI-DEF, released Jan 2026) for native CT volume
interpretation. This model was trained on abdominal CT data and can reason across
multiple slices simultaneously — replacing the U-Net-only pipeline with genuine
multimodal clinical understanding.

Key capabilities used here:
  - Multi-image input: up to 5 suspicious CT slices interleaved as visual tokens
  - Clinical context: patient vitals embedded in the prompt
  - Structured output: JSON findings for downstream report synthesis
  - Streaming Q&A: TextIteratorStreamer for real-time SSE responses

Setup:
    1. Accept model terms at https://huggingface.co/google/medgemma-1.5-4b-it
    2. Set HF_TOKEN environment variable or call huggingface_hub.login()
"""

import json
import os
import re
import threading

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)


class MedGemmaVisualAnalyzer:
    """
    MedGemma 1.5 4B multimodal analyzer for trauma CT interpretation.

    Accepts the suspicious slices identified by MedSigLIPTriager, passes them
    as interleaved image+text tokens (Gemma3 native format), and returns
    structured clinical findings.

    Also provides stream_answer() for real-time Q&A via SSE.
    """

    MODEL_ID = "google/medgemma-1.5-4b-it"
    MAX_SLICES = 5  # ~256 image tokens each; 5 slices = ~1280 tokens, safe in 8K context

    def __init__(
        self,
        device: str = "auto",
        use_4bit: bool = True,
        hf_token: str = None,
    ):
        """
        Args:
            device: "auto" uses accelerate's device_map (recommended for Colab).
                    "cuda" or "cpu" for explicit placement.
            use_4bit: 4-bit BitsAndBytes quantization. Reduces VRAM from ~8GB to ~4GB.
                      Automatically disabled if CUDA is unavailable.
            hf_token: HuggingFace token. Falls back to HF_TOKEN env var.
        """
        token = hf_token or os.environ.get("HF_TOKEN")
        cuda_available = torch.cuda.is_available()

        bnb_config = None
        if use_4bit and cuda_available:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print(f"[MedGemmaVisualAnalyzer] Loading {self.MODEL_ID} with 4-bit NF4 quantization...")
        else:
            dtype = torch.bfloat16 if cuda_available else torch.float32
            print(f"[MedGemmaVisualAnalyzer] Loading {self.MODEL_ID} in {'bfloat16' if cuda_available else 'float32'}...")

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.bfloat16 if cuda_available else torch.float32,
            device_map=device if cuda_available else "cpu",
            quantization_config=bnb_config,
            token=token,
        )
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID, token=token)
        self.model.eval()

        # Track the actual device for input tensors
        self._device = next(self.model.parameters()).device
        print(f"[MedGemmaVisualAnalyzer] Ready on {self._device}.")

    def analyze(self, pil_images: list, vitals: dict = None) -> dict:
        """
        Analyze suspicious CT slices and return structured findings.

        Images are passed in order of descending MedSigLIP suspicion score,
        so MedGemma sees the most hemorrhage-likely slices first.

        Args:
            pil_images: List of PIL Images (already filtered by triage, max 5).
            vitals: Optional dict with keys hr, bp, gcs for clinical context.

        Returns:
            {
                "injury_pattern": str,
                "organs_involved": list[str],
                "bleeding_description": str,
                "severity_estimate": str,       # "mild" | "moderate" | "severe" | "none"
                "differential_diagnosis": list[str],
                "raw_response": str
            }
        """
        slices = pil_images[: self.MAX_SLICES]
        content = self._build_content(slices, vitals)
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Move to model device, keeping bfloat16 for image features
        inputs = {
            k: v.to(self._device) for k, v in inputs.items()
        }

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=800,
                do_sample=False,
                temperature=1.0,  # ignored when do_sample=False
            )

        raw = self.processor.decode(
            generation[0][input_len:], skip_special_tokens=True
        )
        return self._parse_findings(raw)

    def stream_answer(self, question: str, context: dict, pil_images: list):
        """
        Generator for SSE-based streaming Q&A about the current scan.

        The model receives the scan images + a summary of prior findings as
        context, then streams its answer token-by-token.

        Args:
            question: Clinician's follow-up question.
            context: Dict from run_pipeline() containing findings, volume, etc.
            pil_images: The suspicious slices from the session (max 3 for Q&A).

        Yields:
            str: Individual token strings. Caller wraps as "data: {token}\n\n".
        """
        slices = pil_images[:3]  # Fewer images for Q&A to keep latency low

        context_summary = (
            f"Previous AI analysis findings:\n"
            f"- Injury pattern: {context.get('injury_pattern', 'N/A')}\n"
            f"- Organs involved: {', '.join(context.get('organs_involved', []))}\n"
            f"- Bleeding: {context.get('bleeding_description', 'N/A')}\n"
            f"- Severity: {context.get('severity_estimate', 'N/A')}\n"
            f"- Hemorrhage volume: {context.get('volume_ml', 0):.1f} mL\n"
            f"- Risk level: {context.get('risk_level', 'N/A')}\n"
            f"- Differential: {', '.join(context.get('differential_diagnosis', []))}\n"
        )

        content = []
        for i, img in enumerate(slices):
            content.append({"type": "image", "image": img.convert("RGB")})
            content.append({"type": "text", "text": f"[Slice {i + 1}]"})

        content.append({"type": "text", "text": (
            f"{context_summary}\n"
            f"You are a trauma radiologist. Answer the following clinical question "
            f"concisely and accurately based on these CT slices and the analysis above.\n\n"
            f"Question: {question}"
        )})

        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
        )

        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": 500,
            "do_sample": False,
        }

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for token in streamer:
            yield token

        thread.join()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_content(self, slices: list, vitals: dict = None) -> list:
        """Build the interleaved image+text content list for MedGemma."""
        content = []

        for i, img in enumerate(slices):
            # MedGemma expects RGB images
            content.append({"type": "image", "image": img.convert("RGB")})
            content.append({"type": "text", "text": f"[CT Slice {i + 1} of {len(slices)}]"})

        vitals_text = ""
        if vitals:
            parts = []
            if vitals.get("hr"):
                parts.append(f"HR {vitals['hr']} bpm")
            if vitals.get("bp"):
                parts.append(f"BP {vitals['bp']}")
            if vitals.get("gcs"):
                parts.append(f"GCS {vitals['gcs']}")
            if parts:
                vitals_text = f"\nPatient vitals: {', '.join(parts)}."

        content.append({"type": "text", "text": (
            f"You are a trauma radiologist analyzing {len(slices)} abdominal CT "
            f"angiogram slice(s) from a trauma patient.{vitals_text}\n\n"
            "Analyze for: active hemorrhage, hemoperitoneum, solid organ injury "
            "(liver, spleen, kidneys), vascular injury, and bowel/mesenteric injury.\n\n"
            "Respond ONLY with valid JSON in this exact format:\n"
            "{\n"
            '  "injury_pattern": "<brief description of injury pattern>",\n'
            '  "organs_involved": ["<organ1>", "<organ2>"],\n'
            '  "bleeding_description": "<description of bleeding if present>",\n'
            '  "severity_estimate": "<none|mild|moderate|severe>",\n'
            '  "differential_diagnosis": ["<dx1>", "<dx2>", "<dx3>"]\n'
            "}"
        )})

        return content

    def _parse_findings(self, raw: str) -> dict:
        """
        Extract JSON from model response. Handles cases where the model
        wraps JSON in markdown code blocks or adds explanatory text.
        """
        # Try to find a JSON object in the response
        json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                # Ensure all expected keys exist with safe defaults
                parsed.setdefault("injury_pattern", "Unable to determine from provided slices")
                parsed.setdefault("organs_involved", [])
                parsed.setdefault("bleeding_description", "See raw response below")
                parsed.setdefault("severity_estimate", "unknown")
                parsed.setdefault("differential_diagnosis", [])
                parsed["raw_response"] = raw
                return parsed
            except json.JSONDecodeError:
                pass

        # Fallback: return raw text in structured form
        return {
            "injury_pattern": "Structured output unavailable — see raw response",
            "organs_involved": [],
            "bleeding_description": raw[:500] if len(raw) > 500 else raw,
            "severity_estimate": "unknown",
            "differential_diagnosis": [],
            "raw_response": raw,
        }
