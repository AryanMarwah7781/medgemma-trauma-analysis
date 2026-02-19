"""
MedGemma Trauma Analysis — Flask Web Application
==================================================
Serves the multi-model HAI-DEF pipeline:
  - MedSigLIP-448 zero-shot triage
  - MedGemma 1.5 4b-it native CT volume interpretation
  - U-Net precise hemorrhage quantification
  - EAST guideline-aligned report synthesis
  - SSE streaming clinical Q&A

Running with Colab tunnel:
    USE_NGROK=true NGROK_TOKEN=your_token python app.py

HuggingFace setup (required before first run):
    1. Accept terms at https://huggingface.co/google/medgemma-1.5-4b-it
    2. Accept terms at https://huggingface.co/google/medsiglip-448
    3. export HF_TOKEN=hf_your_token_here
"""

import os
import uuid

import torch
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    stream_with_context,
)
from werkzeug.utils import secure_filename

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.orchestrator import TraumaOrchestrator

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB for multi-slice uploads
app.config["UPLOAD_FOLDER"] = "uploads"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Global orchestrator — loaded once at startup, shared across all requests
orchestrator: TraumaOrchestrator = None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_models():
    """Initialize all models. Called once before serving requests."""
    global orchestrator
    cuda = torch.cuda.is_available()
    lora_adapter = os.environ.get("LORA_ADAPTER") or None
    print(f"[app.py] CUDA available: {cuda}")
    if lora_adapter:
        print(f"[app.py] LoRA adapter: {lora_adapter}")
    orchestrator = TraumaOrchestrator(
        device="auto" if cuda else "cpu",
        use_4bit=cuda,
        triage_threshold=float(os.environ.get("TRIAGE_THRESHOLD", "0.25")),
        hf_token=os.environ.get("HF_TOKEN"),
        lora_adapter=lora_adapter,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """
    Multi-slice CT scan upload and analysis.

    Form fields:
        files:      One or more CT slice images (PNG/JPG).
        hr:         Heart rate in bpm (optional).
        bp:         Blood pressure e.g. "90/60" (optional).
        gcs:        Glasgow Coma Scale score (optional).
        patient_id: Patient identifier (optional).

    Returns:
        JSON with full pipeline result including session_id for Q&A.
    """
    if orchestrator is None:
        return jsonify({"success": False, "error": "Models not loaded yet."}), 503

    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        # Backwards-compatible: also accept single-file upload as "file"
        single = request.files.get("file")
        if single and single.filename:
            files = [single]
        else:
            return jsonify({"success": False, "error": "No files uploaded."}), 400

    # Save uploaded files
    image_paths = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            image_paths.append(filepath)

    if not image_paths:
        return jsonify({"success": False, "error": "No valid image files found."}), 400

    # Extract optional vitals
    vitals = {
        "hr": request.form.get("hr") or None,
        "bp": request.form.get("bp") or None,
        "gcs": request.form.get("gcs") or None,
    }
    vitals = {k: v for k, v in vitals.items() if v}

    patient_id = request.form.get("patient_id") or f"PT-{uuid.uuid4().hex[:6].upper()}"

    try:
        result = orchestrator.run_pipeline(
            image_paths=image_paths,
            vitals=vitals if vitals else None,
            patient_id=patient_id,
        )
        return jsonify({"success": True, "result": result})

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return jsonify({
            "success": False,
            "error": "GPU out of memory. Try uploading fewer slices or restart the server.",
        }), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/qa-stream")
def qa_stream():
    """
    SSE endpoint for real-time clinical Q&A about a previously analyzed scan.

    Query params:
        session_id: UUID from a prior /upload response.
        q:          The clinician's question.

    Returns:
        text/event-stream with token-by-token response ending with [DONE].
    """
    if orchestrator is None:
        return jsonify({"error": "Models not loaded."}), 503

    session_id = request.args.get("session_id", "")
    question = request.args.get("q", "").strip()

    if not session_id or not question:
        return jsonify({"error": "session_id and q are required."}), 400

    def generate():
        try:
            for token in orchestrator.stream_qa(session_id, question):
                yield f"data: {token}\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.route("/evaluate", methods=["POST"])
def evaluate():
    """
    Trigger ablation study evaluation against RSNA ground truth.

    JSON body:
        data_dir: Path to CT dataset directory.
        n_cases:  Number of cases to evaluate (default 10).

    Returns:
        JSON array of ablation result rows.
    """
    if orchestrator is None:
        return jsonify({"error": "Models not loaded."}), 503

    body = request.get_json(silent=True) or {}
    data_dir = body.get("data_dir", "data/huggingface")
    n_cases = int(body.get("n_cases", 10))

    try:
        from scripts.evaluate_pipeline import PipelineEvaluator
        evaluator = PipelineEvaluator(orchestrator, data_dir)
        results_df = evaluator.run_ablation(n_cases=n_cases)
        return jsonify(results_df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """Return model load status and GPU info."""
    if orchestrator is None:
        return jsonify({"status": "loading", "models_loaded": False}), 503
    return jsonify({"status": "ok", **orchestrator.get_status()})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_models()

    # Optional: expose via ngrok for Colab demo
    port = int(os.environ.get("PORT", 7860))  # 7860 = HF Spaces default; 5000 for local/Colab
    if os.environ.get("USE_NGROK", "").lower() == "true":
        try:
            from pyngrok import ngrok
            ngrok_token = os.environ.get("NGROK_TOKEN")
            if ngrok_token:
                ngrok.set_auth_token(ngrok_token)
            public_url = ngrok.connect(port)
            print(f"\n[ngrok] Public URL: {public_url}\n")
        except ImportError:
            print("[ngrok] pyngrok not installed. Run: pip install pyngrok")

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
