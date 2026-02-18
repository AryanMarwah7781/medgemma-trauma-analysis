"""
Main pipeline for MedGemma Hemorrhage Quantifier
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import CTDataLoader
from unet_model import UNet
from quantification import quantify_hemorrhage, calculate_injury_severity
from medgemma_report import generate_report, create_structured_output


def process_ct_scan(model, image_path, device='cuda'):
    """Process a single CT scan through the pipeline"""
    import torch
    from PIL import Image
    import numpy as np
    
    # Load image
    img = Image.open(image_path).convert('L')
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Add batch and channel dimensions
    x = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)
    
    # Run segmentation
    model.eval()
    with torch.no_grad():
        pred = model(x)
    
    # Convert to binary mask
    mask = (pred.cpu().numpy()[0, 0] > 0.5).astype(np.uint8)
    
    return mask


def main():
    parser = argparse.ArgumentParser(description='MedGemma Hemorrhage Quantifier')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to CT scan data')
    parser.add_argument('--patient-id', type=str, help='Specific patient ID to process')
    parser.add_argument('--model-path', type=str, default=None, help='Path to trained model weights')
    parser.add_argument('--output', type=str, default='reports/', help='Output directory for reports')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    print("=" * 50)
    print("MedGemma Hemorrhage Quantifier")
    print("=" * 50)
    
    # Load data
    print(f"\nLoading data from: {args.data_dir}")
    loader = CTDataLoader(args.data_dir)
    info = loader.get_dataset_info()
    print(f"Total slices: {info['total_slices']}")
    print(f"Total patients: {info['total_patients']}")
    
    # Load model
    print(f"\nLoading U-Net model...")
    device = args.device if args.device == 'cuda' and False else 'cpu'  # Force CPU for now
    model = UNet(in_channels=1, out_channels=1)
    model = model.to(device)
    
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading weights from: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print("No model weights provided. Using untrained model (for demo).")
    
    # Process patients
    patients = loader.get_patient_ids()
    if args.patient_id:
        patients = [args.patient_id]
    
    os.makedirs(args.output, exist_ok=True)
    
    print(f"\nProcessing {len(patients)} patient(s)...")
    
    for i, patient_id in enumerate(patients[:5]):  # Limit to 5 for demo
        print(f"\n[{i+1}/{len(patients)}] Processing patient: {patient_id}")
        
        # Get scans for this patient
        scans = loader.get_patient_scans(patient_id)
        print(f"  Found {len(scans)} scan slices")
        
        # For demo: use first scan
        if scans:
            # Simulate segmentation (in production, run through model)
            import numpy as np
            from PIL import Image
            
            img = Image.open(scans[0]).convert('L')
            img_array = np.array(img)
            
            # Create synthetic mask for demo (in production: model.predict())
            # For now, simulate finding hemorrhage
            mask = np.zeros_like(img_array, dtype=np.uint8)
            # Random "bleeding" region for demo
            mask[200:250, 200:280] = 1
            
            # Quantify hemorrhage
            result = quantify_hemorrhage(mask)
            print(f"  Volume: {result['volume_ml']} mL")
            print(f"  Risk: {result['risk_level']}")
            
            # Get injury severity (demo)
            injury = {
                'total_score': 0,
                'overall_classification': 'LOW',
                'injured_organs': []
            }
            
            # Generate report
            report = generate_report(patient_id, result, injury)
            
            # Save report
            output_file = os.path.join(args.output, f"{patient_id}_report.txt")
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"  Report saved to: {output_file}")
    
    print("\n" + "=" * 50)
    print("Processing complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
