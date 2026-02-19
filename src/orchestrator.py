"""
Multi-Model Pipeline Orchestrator
===================================
Coordinates the full HAI-DEF trauma analysis pipeline:

  Layer 1: MedSigLIP-448 triage — zero-shot per-slice hemorrhage scoring
  Layer 2: MedGemma 1.5 visual analysis — multi-slice CT interpretation
  Layer 3: U-Net segmentation — precise voxel-level quantification
  Layer 4: MedGemma 1.5 report synthesis — EAST-aligned clinical report
  Layer 5: SSE streaming Q&A — session-aware clinician follow-up

All models are loaded once at startup and held in memory for the lifetime
of the Flask process. A lightweight session store maps UUIDs to scan
context for downstream Q&A.
"""

import os
import time
import uuid

import numpy as np
import torch
from PIL import Image

from src.triage import MedSigLIPTriager
from src.visual_analyzer import MedGemmaVisualAnalyzer
from src.pretrained_segmenter import PreTrainedSegmenter
from src.quantification import quantify_hemorrhage
from src.report_synthesizer import ReportSynthesizer


# Sessions older than this are pruned from memory
SESSION_TTL_SECONDS = 1800  # 30 minutes


class TraumaOrchestrator:
    """
    Central coordinator for the five-layer trauma analysis pipeline.

    Usage (in app.py):
        orchestrator = TraumaOrchestrator()          # load models once
        result = orchestrator.run_pipeline(paths, vitals)  # per-request
        for token in orchestrator.stream_qa(session_id, question):  # Q&A
            yield token
    """

    def __init__(
        self,
        device: str = "auto",
        use_4bit: bool = True,
        triage_threshold: float = 0.25,
        hf_token: str = None,
        lora_adapter: str = None,
    ):
        """
        Load all models. Called once at Flask startup.

        Args:
            device: Device for MedGemma ("auto" recommended for Colab).
            use_4bit: 4-bit quantization for MedGemma (requires CUDA).
            triage_threshold: MedSigLIP suspicion cutoff [0, 1].
            hf_token: HuggingFace token. Falls back to HF_TOKEN env var.
            lora_adapter: Path to PEFT LoRA adapter directory. If set, MedGemma
                          uses the fine-tuned RSNA trauma-specialist weights.
        """
        token = hf_token or os.environ.get("HF_TOKEN")
        cuda_available = torch.cuda.is_available()

        # Layer 1: MedSigLIP on CPU — 850M params is fast on CPU,
        # freeing the entire GPU for MedGemma 1.5.
        print("\n" + "=" * 60)
        print("Loading HAI-DEF pipeline models...")
        print("=" * 60)
        self.triager = MedSigLIPTriager(
            device="cpu",
            threshold=triage_threshold,
            hf_token=token,
        )

        # Layer 2 + 4: MedGemma 1.5 on GPU (or CPU fallback)
        self.visual_analyzer = MedGemmaVisualAnalyzer(
            device=device if cuda_available else "cpu",
            use_4bit=use_4bit and cuda_available,
            hf_token=token,
            lora_adapter=lora_adapter,
        )

        # Layer 3: U-Net segmentation (keeps existing functionality)
        print("[Orchestrator] Loading U-Net segmenter (ResNet34)...")
        self.segmenter = PreTrainedSegmenter(
            encoder="resnet34",
            encoder_weights="imagenet",
            device="cuda" if cuda_available else "cpu",
        )

        # Layer 4 reuses the already-loaded MedGemma instance
        self.report_synthesizer = ReportSynthesizer(self.visual_analyzer)

        # Session store: {session_id: {images, context, result, timestamp}}
        self._sessions: dict = {}

        gpu_info = torch.cuda.get_device_name(0) if cuda_available else "CPU only"
        print(f"\n[Orchestrator] All models loaded. GPU: {gpu_info}")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        image_paths: list,
        vitals: dict = None,
        patient_id: str = None,
        voxel_spacing: tuple = (0.5, 0.5, 3.0),
    ) -> dict:
        """
        Execute the full five-layer trauma analysis pipeline.

        Args:
            image_paths: List of paths to uploaded CT slice images.
            vitals: Optional dict {hr, bp, gcs} for clinical context.
            patient_id: Optional identifier for the report header.
            voxel_spacing: (x, y, z) mm spacing for volume calculation.
                           Default is typical abdominal CT protocol spacing.

        Returns:
            Complete result dict including session_id for Q&A continuity.
        """
        session_id = str(uuid.uuid4())

        # Load all slices as PIL Images (supports PNG/JPG and DICOM)
        pil_images = [self._load_image(p) for p in image_paths]

        # --- Layer 1: MedSigLIP Triage ---
        print(f"[Pipeline] Layer 1: Triaging {len(pil_images)} slice(s)...")
        suspicious_images, all_triage_results = self.triager.get_suspicious_slices(
            pil_images, max_slices=5
        )
        triage_summary = self.triager.get_triage_summary(all_triage_results)
        print(
            f"[Pipeline]   → {triage_summary['suspicious_count']}/{triage_summary['total_slices']} "
            f"suspicious (max score: {triage_summary['max_score']:.2f})"
        )

        # --- Layer 2: MedGemma 1.5 Visual Analysis ---
        print(f"[Pipeline] Layer 2: MedGemma 1.5 analyzing {len(suspicious_images)} suspicious slice(s)...")
        try:
            visual_findings = self.visual_analyzer.analyze(suspicious_images, vitals)
        except torch.cuda.OutOfMemoryError:
            print("[Pipeline] WARNING: GPU OOM during visual analysis. Falling back to single slice.")
            torch.cuda.empty_cache()
            visual_findings = self.visual_analyzer.analyze(suspicious_images[:1], vitals)

        # --- Layer 3: U-Net Segmentation + Volume Quantification ---
        print(f"[Pipeline] Layer 3: U-Net segmenting {len(pil_images)} slice(s)...")
        quant_result = self._segment_and_quantify(image_paths, voxel_spacing)

        # --- Build combined context for Layer 4 ---
        context = {
            **visual_findings,
            "volume_ml": quant_result["volume_ml"],
            "risk_level": quant_result["risk_level"],
            "num_voxels": quant_result["num_voxels"],
            "recommendation": quant_result["recommendation"],
            "vitals": vitals or {},
            "triage_summary": triage_summary,
            "patient_id": patient_id or "UNKNOWN",
        }

        # --- Layer 4: EAST-Aligned Report Synthesis ---
        print("[Pipeline] Layer 4: Synthesizing EAST-aligned report...")
        try:
            report = self.report_synthesizer.synthesize(context)
        except Exception as e:
            print(f"[Pipeline] Report synthesis error: {e}. Using template.")
            report = self.report_synthesizer._template_fallback(context)

        # --- Assemble final result ---
        result = {
            "session_id": session_id,
            "patient_id": patient_id,
            "triage": {
                "total_slices": triage_summary["total_slices"],
                "suspicious_slices": triage_summary["suspicious_count"],
                "per_slice_scores": triage_summary["per_slice_scores"],
                "max_score": triage_summary["max_score"],
                "mean_score": triage_summary["mean_score"],
            },
            "visual_findings": {
                "injury_pattern": visual_findings.get("injury_pattern"),
                "organs_involved": visual_findings.get("organs_involved", []),
                "bleeding_description": visual_findings.get("bleeding_description"),
                "severity_estimate": visual_findings.get("severity_estimate"),
                "differential_diagnosis": visual_findings.get("differential_diagnosis", []),
            },
            "quantification": {
                "volume_ml": quant_result["volume_ml"],
                "risk_level": quant_result["risk_level"],
                "num_voxels": quant_result["num_voxels"],
                "recommendation": quant_result["recommendation"],
            },
            "report": report,
            "vitals": vitals or {},
        }

        # Store session for Q&A (Layer 5)
        self._sessions[session_id] = {
            "images": suspicious_images,
            "context": context,
            "result": result,
            "timestamp": time.time(),
        }
        self._prune_old_sessions()

        print(f"[Pipeline] Complete. Session: {session_id[:8]}...")
        return result

    # ------------------------------------------------------------------
    # Layer 5: SSE Streaming Q&A
    # ------------------------------------------------------------------

    def stream_qa(self, session_id: str, question: str):
        """
        Generator for real-time Q&A about a previously analyzed scan.

        Delegates to MedGemmaVisualAnalyzer.stream_answer() which uses
        TextIteratorStreamer for token-by-token output.

        Args:
            session_id: UUID returned by run_pipeline().
            question: Clinician's follow-up question.

        Yields:
            str token strings for SSE formatting by Flask.
        """
        session = self._sessions.get(session_id)
        if not session:
            yield "Session not found or expired. Please upload a new scan."
            return

        yield from self.visual_analyzer.stream_answer(
            question=question,
            context=session["context"],
            pil_images=session["images"],
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return model load status for /health endpoint."""
        cuda = torch.cuda.is_available()
        return {
            "models_loaded": True,
            "gpu_available": cuda,
            "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
            "active_sessions": len(self._sessions),
            "medsiglip": self.triager.MODEL_ID,
            "medgemma": self.visual_analyzer.MODEL_ID,
            "lora_adapter": self.visual_analyzer.lora_adapter,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _segment_and_quantify(self, image_paths: list, voxel_spacing: tuple) -> dict:
        """
        Run U-Net segmentation on all uploaded slices, stack into a 3D
        volume, and compute hemorrhage volume in mL.
        """
        masks = []
        for path in image_paths:
            try:
                result = self.segmenter.predict(path, threshold=0.5)
                masks.append(result["mask"])
            except Exception as e:
                print(f"[Pipeline] Segmentation failed for {path}: {e}")
                # Add a blank mask so the slice count stays consistent
                masks.append(np.zeros((512, 512), dtype=np.uint8))

        if masks:
            # Stack 2D masks into a 3D volume (slices × H × W)
            combined_mask = np.stack(masks, axis=0)
        else:
            combined_mask = np.zeros((1, 512, 512), dtype=np.uint8)

        return quantify_hemorrhage(combined_mask, spacing=voxel_spacing)

    def _load_image(self, path: str) -> Image.Image:
        """
        Load a CT slice from PNG/JPG or DICOM, returning an RGB PIL Image.

        DICOM files are windowed using the embedded WindowCenter/WindowWidth
        tags (defaulting to abdominal window 50/400 if tags are absent) and
        normalised to uint8 before converting to RGB.
        """
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if ext in ("dcm", "dicom"):
            return self._dicom_to_pil(path)
        return Image.open(path).convert("RGB")

    @staticmethod
    def _dicom_to_pil(path: str) -> Image.Image:
        """Convert a DICOM file to an RGB PIL Image with CT abdominal windowing."""
        import pydicom
        import numpy as np

        ds = pydicom.dcmread(path, force=True)
        pixels = ds.pixel_array.astype(np.float32)

        # Apply RescaleSlope / RescaleIntercept to get Hounsfield Units
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        pixels = pixels * slope + intercept

        # CT window — use embedded tags, fall back to abdominal window (50/400)
        center = float(getattr(ds, "WindowCenter", 50))
        width = float(getattr(ds, "WindowWidth", 400))
        if isinstance(center, pydicom.multival.MultiValue):
            center = float(center[0])
        if isinstance(width, pydicom.multival.MultiValue):
            width = float(width[0])

        lo = center - width / 2.0
        hi = center + width / 2.0
        pixels = np.clip(pixels, lo, hi)
        pixels = ((pixels - lo) / (hi - lo) * 255).astype(np.uint8)

        # Stack grayscale into RGB
        rgb = np.stack([pixels, pixels, pixels], axis=-1)
        return Image.fromarray(rgb)

    def _load_image(self, path: str) -> Image.Image:
        """
        Load a CT slice from PNG/JPG or DICOM, returning an RGB PIL Image.

        DICOM files are windowed using the embedded WindowCenter/WindowWidth
        tags (defaulting to abdominal window 50/400 if tags are absent) and
        normalised to uint8 before converting to RGB.
        """
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if ext in ("dcm", "dicom"):
            return self._dicom_to_pil(path)
        return Image.open(path).convert("RGB")

    @staticmethod
    def _dicom_to_pil(path: str) -> Image.Image:
        """Convert a DICOM file to an RGB PIL Image with CT abdominal windowing."""
        import pydicom
        import numpy as np

        ds = pydicom.dcmread(path, force=True)
        pixels = ds.pixel_array.astype(np.float32)

        # Apply RescaleSlope / RescaleIntercept to get Hounsfield Units
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        pixels = pixels * slope + intercept

        # CT window — use embedded tags, fall back to abdominal window (50/400)
        center = float(getattr(ds, "WindowCenter", 50))
        width = float(getattr(ds, "WindowWidth", 400))
        if isinstance(center, pydicom.multival.MultiValue):
            center = float(center[0])
        if isinstance(width, pydicom.multival.MultiValue):
            width = float(width[0])

        lo = center - width / 2.0
        hi = center + width / 2.0
        pixels = np.clip(pixels, lo, hi)
        pixels = ((pixels - lo) / (hi - lo) * 255).astype(np.uint8)

        # Stack grayscale into RGB
        rgb = np.stack([pixels, pixels, pixels], axis=-1)
        return Image.fromarray(rgb)

    def _prune_old_sessions(self):
        """Remove sessions older than SESSION_TTL_SECONDS."""
        now = time.time()
        expired = [
            sid for sid, sess in self._sessions.items()
            if now - sess["timestamp"] > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            print(f"[Orchestrator] Pruned {len(expired)} expired session(s).")
