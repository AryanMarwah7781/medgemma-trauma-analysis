"""
Data loader for RSNA Abdominal Trauma dataset
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Dict

class CTDataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.image_files = sorted(list(self.data_dir.glob("*.png")))
        
    def get_patient_ids(self) -> List[str]:
        """Get unique patient IDs from filenames"""
        patient_ids = set()
        for f in self.image_files:
            patient_id = f.stem.split('_')[0]
            patient_ids.add(patient_id)
        return sorted(list(patient_ids))
    
    def get_patient_scans(self, patient_id: str) -> List[str]:
        """Get all scan slices for a patient"""
        return sorted([str(f) for f in self.image_files if f.stem.startswith(f"{patient_id}_")])
    
    def load_scan(self, image_path: str) -> np.ndarray:
        """Load a single CT scan slice as numpy array"""
        img = Image.open(image_path).convert('L')  # Grayscale
        return np.array(img)
    
    def load_patient_volume(self, patient_id: str) -> np.ndarray:
        """Load all slices for a patient as 3D volume"""
        scans = self.get_patient_scans(patient_id)
        volume = []
        for scan_path in scans:
            volume.append(self.load_scan(scan_path))
        return np.stack(volume, axis=0)
    
    def get_dataset_info(self) -> Dict:
        """Get dataset statistics"""
        patient_ids = self.get_patient_ids()
        return {
            'total_slices': len(self.image_files),
            'total_patients': len(patient_ids),
            'data_dir': str(self.data_dir)
        }


if __name__ == "__main__":
    # Test
    DATA_DIR = "/home/aryanmarwah/.cache/kagglehub/datasets/theoviel/rsna-abdominal-trauma-detection-png-pt1/extracted"
    loader = CTDataLoader(DATA_DIR)
    info = loader.get_dataset_info()
    print(f"Total slices: {info['total_slices']}")
    print(f"Total patients: {info['total_patients']}")
    
    # Test loading one patient
    patients = loader.get_patient_ids()
    print(f"\nFirst 5 patients: {patients[:5]}")
    sample_scans = loader.get_patient_scans(patients[0])
    print(f"Scans for patient {patients[0]}: {len(sample_scans)} slices")
