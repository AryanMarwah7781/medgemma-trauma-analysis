"""
Flask web app for MedGemma Hemorrhage Quantifier
Now with MedGemma LLM integration!
"""

import os
import sys
import json
import requests
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = None
OLLAMA_URL = "http://localhost:11434/api/generate"


def load_segmentation_model():
    global model
    if model is None:
        print("Loading segmentation model...")
        model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=1,
            classes=1
        )
        model.eval()
        print("Segmentation model loaded!")
    return model


def generate_medgemma_report(volume_ml, risk_level, num_voxels):
    """Use MedGemma to generate a medical report"""
    
    prompt = f"""You are MedGemma, a medical AI assistant specialized in trauma radiology.
Based on the following CT scan analysis findings, generate a professional radiology report:

Findings:
- Estimated hemorrhage volume: {volume_ml} mL
- Risk level: {risk_level}
- Number of affected voxels: {num_voxels}

Generate a structured radiology report with:
1. FINDINGS - detailed observations
2. IMPRESSION - clinical assessment
3. RECOMMENDATION - treatment advice

Be concise and clinically accurate."""

    payload = {
        "model": "amsaravi/medgemma-4b-it:q6",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '').strip()
        else:
            return None
    except Exception as e:
        print(f"MedGemma error: {e}")
        return None


def process_ct_scan(image_path):
    """Process CT scan and generate report with MedGemma"""
    seg_model = load_segmentation_model()
    
    img = Image.open(image_path).convert('L')
    img_array = np.array(img).astype(np.float32) / 255.0
    
    x = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        pred = seg_model(x)
        pred = torch.sigmoid(pred)
    
    mask = pred.cpu().numpy()[0, 0]
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    voxel_volume_ml = (0.5 * 0.5 * 3.0) / 1000
    num_bleeding_voxels = np.sum(binary_mask)
    volume_ml = num_bleeding_voxels * voxel_volume_ml
    
    if volume_ml < 10:
        risk_level = "LOW"
        recommendation = "Observation recommended"
    elif volume_ml < 50:
        risk_level = "MODERATE"
        recommendation = "Consider angioembolization"
    else:
        risk_level = "HIGH"
        recommendation = "Urgent intervention required"
    
    # Generate MedGemma report
    print("Generating MedGemma report...")
    llm_report = generate_medgemma_report(volume_ml, risk_level, num_bleeding_voxels)
    
    if not llm_report:
        # Fallback if MedGemma fails
        llm_report = f"""FINDINGS
- Active extravasation detected: {volume_ml:.2f} mL
- Risk assessment: {risk_level}
- Voxels affected: {num_bleeding_voxels}

IMPRESSION
{"Findings warrant close clinical monitoring." if risk_level == "LOW" else "Urgent consultation recommended."}

RECOMMENDATION
{recommendation}"""
    
    return {
        'volume_ml': round(volume_ml, 2),
        'num_voxels': int(num_bleeding_voxels),
        'risk_level': risk_level,
        'recommendation': recommendation,
        'llm_report': llm_report
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filename = f"{hash(file.filename)}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        result = process_ct_scan(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'result': result
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    load_segmentation_model()
    app.run(host='0.0.0.0', port=5000, debug=False)
