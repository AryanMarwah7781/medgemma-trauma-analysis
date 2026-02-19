"""
MedSigLIP Triage Layer
======================
Uses google/medsiglip-448 (HAI-DEF medical vision-language encoder) to zero-shot
classify every CT slice for hemorrhage/injury. Only suspicious slices proceed to
the expensive MedGemma 1.5 analysis, reducing compute cost by 60-80%.

Setup:
    1. Accept model terms at https://huggingface.co/google/medsiglip-448
    2. Set HF_TOKEN environment variable or call huggingface_hub.login()
"""

import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, SiglipImageProcessor, SiglipTokenizer


class MedSigLIPTriager:
    """
    Zero-shot CT slice triage using MedSigLIP-448.

    Scores each slice against hemorrhage-specific text candidates.
    Suspicious slices (score > threshold) are forwarded to MedGemma 1.5.
    Normal-looking slices are skipped, saving significant inference time.
    """

    MODEL_ID = "google/medsiglip-448"
    IMAGE_SIZE = 448  # Required input resolution per model card

    # Clinically specific labels for abdominal trauma CT
    LABELS = [
        "CT scan showing intraabdominal hemorrhage or active bleeding",
        "CT scan with liver laceration, splenic injury, or solid organ trauma",
        "CT scan showing hemoperitoneum or free fluid in the abdomen",
        "Normal CT scan of the abdomen without hemorrhage or injury",
        "CT scan with bowel perforation or mesenteric injury",
    ]

    # Label indices that count as suspicious (index 3 = "Normal" is the only negative)
    POSITIVE_LABEL_INDICES = [0, 1, 2, 4]
    NEGATIVE_LABEL_INDEX = 3

    def __init__(
        self,
        device: str = "cpu",
        threshold: float = None,
        hf_token: str = None,
    ):
        """
        Args:
            device: "cpu" or "cuda". MedSigLIP is 850M params — loads fine on CPU,
                    which frees the GPU entirely for MedGemma 1.5.
            threshold: Suspicious score cutoff [0, 1]. Defaults to env var
                       TRIAGE_THRESHOLD or 0.25. Lower = more slices pass through.
            hf_token: HuggingFace token. Falls back to HF_TOKEN env var.
        """
        self.device = device
        self.threshold = threshold if threshold is not None else float(
            os.environ.get("TRIAGE_THRESHOLD", "0.25")
        )
        token = hf_token or os.environ.get("HF_TOKEN")

        print(f"[MedSigLIPTriager] Loading {self.MODEL_ID} on {self.device}...")
        # Load image processor and tokenizer separately — SiglipProcessor combines
        # them but hits a transformers bug where the tokenizer routing returns None.
        self.image_processor = SiglipImageProcessor.from_pretrained(self.MODEL_ID, token=token)
        self.tokenizer = SiglipTokenizer.from_pretrained(self.MODEL_ID, token=token)
        self.model = AutoModel.from_pretrained(self.MODEL_ID, token=token).to(self.device)
        self.model.eval()
        print(f"[MedSigLIPTriager] Ready. Threshold={self.threshold}")

    def score_slice(self, pil_image: Image.Image) -> dict:
        """
        Score a single CT slice against all labels.

        CT slices are grayscale — converted to RGB by replicating the channel,
        which is standard for SigLIP-family models.

        Returns:
            {
                "scores": {label_str: probability},
                "suspicious_score": float,  # sum of positive label probs
                "suspicious": bool,
                "top_label": str
            }
        """
        # MedSigLIP requires RGB at 448x448
        image = pil_image.convert("RGB").resize(
            (self.IMAGE_SIZE, self.IMAGE_SIZE), Image.BILINEAR
        )

        text_inputs = self.tokenizer(
            self.LABELS,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        image_inputs = self.image_processor(
            images=image,
            return_tensors="pt",
        ).to(self.device)
        inputs = {**text_inputs, **image_inputs}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # logits_per_image shape: (1, num_labels)
        probs = torch.softmax(outputs.logits_per_image[0], dim=0).cpu().numpy()

        scores = {label: float(probs[i]) for i, label in enumerate(self.LABELS)}
        suspicious_score = float(sum(probs[i] for i in self.POSITIVE_LABEL_INDICES))
        top_label = self.LABELS[int(np.argmax(probs))]

        return {
            "scores": scores,
            "suspicious_score": suspicious_score,
            "suspicious": suspicious_score > self.threshold,
            "top_label": top_label,
        }

    def triage_slices(self, pil_images: list) -> list:
        """
        Score all slices, annotating each result with its original index.

        Returns list of result dicts sorted by suspicious_score descending
        (most suspicious first).
        """
        results = []
        for i, img in enumerate(pil_images):
            result = self.score_slice(img)
            result["slice_index"] = i
            results.append(result)

        return sorted(results, key=lambda x: x["suspicious_score"], reverse=True)

    def get_suspicious_slices(
        self, pil_images: list, max_slices: int = 5
    ) -> tuple:
        """
        Triage all slices and return the most suspicious ones for MedGemma.

        If no slices exceed the threshold (e.g., genuinely normal scan),
        still returns the top-1 most suspicious for MedGemma to confirm.

        Args:
            pil_images: All uploaded CT slices as PIL Images.
            max_slices: Maximum number of slices to pass to MedGemma (default 5,
                        since each image ~256 tokens in Gemma3 context).

        Returns:
            (suspicious_pil_images, all_triage_results)
            - suspicious_pil_images: Ordered most-to-least suspicious, capped at max_slices
            - all_triage_results: Full results list for UI badge display
        """
        all_results = self.triage_slices(pil_images)

        # Filter to suspicious slices
        suspicious = [r for r in all_results if r["suspicious"]]

        # Always include at least top-1 so MedGemma can make a definitive call
        if not suspicious:
            suspicious = [all_results[0]]

        # Cap at max_slices, taking the most suspicious ones
        top_suspicious = suspicious[:max_slices]
        indices = [r["slice_index"] for r in top_suspicious]
        suspicious_images = [pil_images[i] for i in indices]

        return suspicious_images, all_results

    def get_triage_summary(self, all_results: list) -> dict:
        """
        Aggregate triage results for the API response and UI display.

        Returns:
            {
                "total_slices": int,
                "suspicious_count": int,
                "max_score": float,
                "mean_score": float,
                "per_slice_scores": [float, ...]  # original order
            }
        """
        # Restore original order for per-slice UI display
        by_index = sorted(all_results, key=lambda x: x["slice_index"])
        per_slice_scores = [r["suspicious_score"] for r in by_index]
        suspicious_count = sum(1 for r in all_results if r["suspicious"])

        return {
            "total_slices": len(all_results),
            "suspicious_count": suspicious_count,
            "max_score": float(max(per_slice_scores)) if per_slice_scores else 0.0,
            "mean_score": float(np.mean(per_slice_scores)) if per_slice_scores else 0.0,
            "per_slice_scores": per_slice_scores,
        }
