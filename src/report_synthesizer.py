"""
EAST-Aligned Report Synthesizer
=================================
Uses the already-loaded MedGemma 1.5 instance to synthesize a final structured
radiology report from all pipeline outputs. This is a text-only second pass —
no images needed — so it reuses the model already in GPU memory.

The report follows EAST (Eastern Association for the Surgery of Trauma) practice
management guidelines for abdominal hemorrhage management.

EAST References:
  - Damage Control Resuscitation (2014, updated 2024)
  - Management of Splenic Injury (2012)
  - Blunt Hepatic Trauma (2012)
  - Pelvic Fracture Hemorrhage (2011)
  https://www.east.org/education-resources/practice-management-guidelines
"""

import torch


class ReportSynthesizer:
    """
    Synthesizes the final clinical report by combining:
      - MedSigLIP triage scores
      - MedGemma visual findings
      - U-Net volume quantification
      - Patient vitals
    into a structured, EAST guideline-aligned report.

    Reuses the MedGemmaVisualAnalyzer model instance — does NOT load a second
    copy of the 4B model.
    """

    # EAST guideline-based first-line recommendations by hemorrhage risk tier
    EAST_RECOMMENDATIONS = {
        "LOW": (
            "Non-operative management (NOM) is appropriate. "
            "Serial abdominal exams every 4-6 hours. "
            "Maintain hemodynamic stability monitoring. "
            "Repeat CT in 24-48 hours if clinical concern."
        ),
        "MODERATE": (
            "Consider angiography with selective embolization (SAE) if hemodynamically stable. "
            "If hemodynamically unstable despite resuscitation, proceed to OR. "
            "Massive transfusion protocol (MTP) activation if ongoing hemorrhage. "
            "Trauma surgery consultation required."
        ),
        "HIGH": (
            "EMERGENT intervention required. "
            "If hemodynamically unstable: immediate operative exploration. "
            "If transiently stable: angioembolization as bridge or definitive therapy. "
            "Activate massive transfusion protocol (1:1:1 ratio pRBC:FFP:PLT). "
            "REBOA (Zone III) may be considered for pelvic hemorrhage. "
            "Damage control surgery principles apply."
        ),
    }

    # ATLS shock classification for contextualizing vitals
    SHOCK_CLASS = {
        (0, 750): "Class I (minimal — observe)",
        (750, 1500): "Class II (mild — IV fluids, type & screen)",
        (1500, 2000): "Class III (moderate — MTP activation, OR standby)",
        (2000, float("inf")): "Class IV (severe — immediate OR)",
    }

    def __init__(self, visual_analyzer):
        """
        Args:
            visual_analyzer: Loaded MedGemmaVisualAnalyzer instance.
                             Reuses its model and processor — no second load.
        """
        self.va = visual_analyzer

    def synthesize(self, context: dict) -> str:
        """
        Generate the final EAST-aligned clinical report.

        First attempts a MedGemma inference call with full context.
        Falls back to a high-quality deterministic template if the model
        call fails (OOM, timeout, parse error).

        Args:
            context: Combined dict from orchestrator containing:
                - injury_pattern, organs_involved, bleeding_description (from MedGemma visual)
                - severity_estimate, differential_diagnosis (from MedGemma visual)
                - volume_ml, risk_level, num_voxels (from U-Net quantification)
                - vitals: {hr, bp, gcs}
                - triage_summary: {total_slices, suspicious_count, max_score}

        Returns:
            Formatted report string.
        """
        try:
            return self._medgemma_synthesis(context)
        except Exception as e:
            print(f"[ReportSynthesizer] MedGemma synthesis failed ({e}), using template.")
            return self._template_fallback(context)

    def _medgemma_synthesis(self, ctx: dict) -> str:
        """Run a text-only MedGemma inference for report synthesis.

        The LoRA adapter was fine-tuned on image→JSON tasks. Using it for
        text-only long-form generation causes degenerate token loops.
        We disable the adapter for this call so the base model weights handle
        the report; the adapter stays active for visual analysis (Layer 2).
        """
        prompt = self._build_synthesis_prompt(ctx)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        inputs = self.va.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.va._device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        # Disable LoRA adapter for text-only synthesis — base model weights
        # generate clean English prose; adapter is image→JSON specialized.
        try:
            from peft import PeftModel
            use_disable = isinstance(self.va.model, PeftModel)
        except ImportError:
            use_disable = False

        def _generate():
            return self.va.model.generate(
                **inputs,
                max_new_tokens=800,
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
            )

        with torch.inference_mode():
            if use_disable:
                with self.va.model.disable_adapter():
                    output = _generate()
            else:
                output = _generate()

        report = self.va.processor.decode(
            output[0][input_len:], skip_special_tokens=True
        ).strip()

        # MedGemma 1.5 is a thinking model — it may prepend internal reasoning
        # before the actual report. Strip everything up to the first real section
        # header. Use regex to match headers on their own line (not in comma lists
        # like "CLINICAL INDICATION, FINDINGS, IMPRESSION...").
        import re
        header_pattern = re.compile(
            r'^(CLINICAL INDICATION|FINDINGS|IMPRESSION|RADIOLOGY|TRAUMA RADIOLOGY|CT ANGIOGRAM)',
            re.MULTILINE | re.IGNORECASE
        )
        m = header_pattern.search(report)
        if m:
            report = report[m.start():].strip()

        non_ascii = sum(1 for c in report if ord(c) > 127)
        if not report or (non_ascii / max(len(report), 1)) > 0.25:
            raise ValueError(f"Degenerate output detected ({non_ascii}/{len(report)} non-ASCII chars)")

        return report

    def _build_synthesis_prompt(self, ctx: dict) -> str:
        """Construct the synthesis prompt with all pipeline findings."""
        east_rec = self.EAST_RECOMMENDATIONS.get(
            ctx.get("risk_level", "LOW"), self.EAST_RECOMMENDATIONS["LOW"]
        )
        shock_class = self._estimate_shock_class(ctx.get("volume_ml", 0))
        vitals = ctx.get("vitals") or {}
        vitals_str = ", ".join(
            f"{k.upper()} {v}"
            for k, v in vitals.items()
            if v
        ) or "Not provided"

        organs = ", ".join(ctx.get("organs_involved", [])) or "None identified"
        differentials = "\n".join(
            f"  - {d}" for d in ctx.get("differential_diagnosis", [])
        ) or "  - None listed"

        triage = ctx.get("triage_summary", {})

        return f"""You are a trauma surgery attending. Write a formal CT radiology report using the AI analysis data below.

Do NOT include any reasoning, planning, analysis, or preamble.
Start your response IMMEDIATELY with the word "CLINICAL INDICATION" on the first line.

AI ANALYSIS DATA:
- MedSigLIP Triage: {triage.get('suspicious_count', '?')}/{triage.get('total_slices', '?')} slices suspicious (max score: {triage.get('max_score', 0):.2f})
- Injury pattern: {ctx.get('injury_pattern', 'N/A')}
- Organs involved: {organs}
- Bleeding: {ctx.get('bleeding_description', 'N/A')}
- Severity: {ctx.get('severity_estimate', 'N/A')}
- Hemorrhage volume: {ctx.get('volume_ml', 0):.1f} mL ({shock_class})
- Risk tier: {ctx.get('risk_level', 'N/A')}
- Patient vitals: {vitals_str}
- Differentials: {', '.join(ctx.get('differential_diagnosis', [])) or 'None listed'}
- EAST recommendation: {east_rec}

CLINICAL INDICATION
Abdominal CT angiogram for trauma evaluation.

FINDINGS"""

    def _template_fallback(self, ctx: dict) -> str:
        """
        High-quality deterministic report for when MedGemma inference fails.
        Still incorporates all pipeline findings and EAST guidelines.
        """
        east_rec = self.EAST_RECOMMENDATIONS.get(
            ctx.get("risk_level", "LOW"), self.EAST_RECOMMENDATIONS["LOW"]
        )
        shock_class = self._estimate_shock_class(ctx.get("volume_ml", 0))
        organs = ", ".join(ctx.get("organs_involved", [])) or "None specifically identified"
        differentials = "\n".join(
            f"  - {d}" for d in ctx.get("differential_diagnosis", [])
        ) or "  - Requires clinical correlation"
        vitals = ctx.get("vitals") or {}
        vitals_str = ", ".join(f"{k.upper()} {v}" for k, v in vitals.items() if v) or "Not documented"

        severity = ctx.get("severity_estimate", "unknown").upper()
        risk = ctx.get("risk_level", "UNKNOWN")
        volume = ctx.get("volume_ml", 0)

        return f"""RADIOLOGY CT ANGIOGRAM REPORT — AI-ASSISTED (MedGemma 1.5 + MedSigLIP)
{'=' * 60}

CLINICAL INDICATION
Trauma evaluation. AI-assisted hemorrhage detection and quantification.
Patient vitals: {vitals_str}

FINDINGS
{ctx.get('injury_pattern', 'No definitive injury pattern identified.')}

Hemorrhage assessment: {ctx.get('bleeding_description', 'See quantification below.')}
Organs involved: {organs}
AI severity estimate: {severity}
Quantified hemorrhage volume: {volume:.1f} mL ({shock_class})
Total suspicious CT voxels: {ctx.get('num_voxels', 0):,}

IMPRESSION
{severity} abdominal injury with {volume:.1f} mL estimated hemorrhage.
Risk classification: {risk} — {shock_class}.

EAST GUIDELINE RECOMMENDATION
{east_rec}

DIFFERENTIAL DIAGNOSIS
{differentials}

FOLLOW-UP PLAN
- Trauma surgery consultation: {'URGENT' if risk == 'HIGH' else 'Required'}
- Hemodynamic monitoring: Continuous
- Repeat imaging: {'Contraindicated — proceed to OR' if risk == 'HIGH' else '24-48h or sooner if clinical deterioration'}
- Labs: CBC, BMP, coagulation panel, type & screen

NOTE: This report was generated by an AI system (MedGemma 1.5 + MedSigLIP + U-Net).
All findings must be verified by a qualified radiologist before clinical action.
"""

    @staticmethod
    def _estimate_shock_class(volume_ml: float) -> str:
        """Map estimated hemorrhage volume to ATLS shock class."""
        if volume_ml < 750:
            return "ATLS Class I"
        elif volume_ml < 1500:
            return "ATLS Class II"
        elif volume_ml < 2000:
            return "ATLS Class III"
        else:
            return "ATLS Class IV"
