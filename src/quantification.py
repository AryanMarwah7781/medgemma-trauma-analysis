"""
Volume quantification - convert segmentation to blood volume (ml)
"""

import numpy as np
from typing import Dict, Tuple


def calculate_voxel_volume(spacing: Tuple[float, float, float] = (0.5, 0.5, 3.0)) -> float:
    """
    Calculate volume of a single voxel in ml
    Default spacing: (0.5mm, 0.5mm, 3.0mm) - typical for abdominal CT
    """
    # Convert mm to cm (1 cm = 10 mm)
    spacing_cm = tuple(s / 10 for s in spacing)
    return spacing_cm[0] * spacing_cm[1] * spacing_cm[2]


def quantify_hemorrhage(segmentation_mask: np.ndarray, spacing: Tuple[float, float, float] = (0.5, 0.5, 3.0)) -> Dict:
    """
    Calculate hemorrhage volume from segmentation mask
    
    Args:
        segmentation_mask: Binary mask (H x W) or (D x H x W)
        spacing: Voxel spacing in mm (x, y, z)
    
    Returns:
        Dictionary with volume calculations
    """
    voxel_volume = calculate_voxel_volume(spacing)
    
    if segmentation_mask.ndim == 3:
        # 3D volume
        num_voxels = np.sum(segmentation_mask > 0)
        volume_ml = num_voxels * voxel_volume
    else:
        # 2D slice - assume single slice
        num_voxels = np.sum(segmentation_mask > 0)
        volume_ml = num_voxels * voxel_volume
    
    # Risk assessment based on volume
    if volume_ml < 10:
        risk_level = "LOW"
        recommendation = "Observation recommended"
    elif volume_ml < 50:
        risk_level = "MODERATE"
        recommendation = "Consider angioembolization"
    else:
        risk_level = "HIGH"
        recommendation = "Urgent intervention required"
    
    return {
        'num_voxels': int(num_voxels),
        'volume_ml': round(volume_ml, 2),
        'voxel_volume_ml': voxel_volume,
        'risk_level': risk_level,
        'recommendation': recommendation
    }


def calculate_injury_severity(organ_masks: Dict[str, np.ndarray]) -> Dict:
    """
    Calculate overall injury severity from organ segmentation
    
    Args:
        organ_masks: Dict of {'liver': mask, 'spleen': mask, 'kidney_left': mask, 'kidney_right': mask}
    
    Returns:
        Severity score and classification
    """
    injury_scores = {
        'none': 0,
        'low': 1,
        'moderate': 2,
        'high': 3
    }
    
    total_score = 0
    injured_organs = []
    
    for organ, mask in organ_masks.items():
        if mask is not None and np.sum(mask) > 0:
            # Determine injury level based on % affected
            affected_pct = (np.sum(mask) / mask.size) * 100
            
            if affected_pct < 5:
                level = 'low'
            elif affected_pct < 20:
                level = 'moderate'
            else:
                level = 'high'
            
            total_score += injury_scores[level]
            injured_organs.append(f"{organ}:{level}")
    
    # Overall classification
    if total_score == 0:
        overall = "NO INJURY"
    elif total_score < 3:
        overall = "MINOR"
    elif total_score < 6:
        overall = "MODERATE"
    else:
        overall = "SEVERE"
    
    return {
        'total_score': total_score,
        'overall_classification': overall,
        'injured_organs': injured_organs
    }


if __name__ == "__main__":
    # Test
    # Simulate a segmentation mask (100 voxels bleeding)
    test_mask = np.zeros((512, 512), dtype=np.uint8)
    test_mask[100:110, 100:110] = 1  # 100 voxels
    
    result = quantify_hemorrhage(test_mask)
    print("Hemorrhage Quantification:")
    for k, v in result.items():
        print(f"  {k}: {v}")
