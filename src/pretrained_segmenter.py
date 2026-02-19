"""
Pre-trained segmentation model for MedGemma
Uses segmentation_models_pytorch with pre-trained encoder
"""

import torch
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image
from pathlib import Path


class PreTrainedSegmenter:
    def __init__(self, encoder='resnet34', encoder_weights='imagenet', device='cpu'):
        """
        Initialize pre-trained segmentation model
        
        Args:
            encoder: Encoder backbone (resnet34, efficientnet-b0, etc.)
            encoder_weights: Pretrained weights (imagenet, etc.)
            device: Device to use (cuda/cpu)
        """
        self.device = device
        
        # Create U-Net with pretrained encoder
        self.model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=1,  # Grayscale CT
            classes=1,      # Binary segmentation (bleeding/not bleeding)
            activation=None  # Will apply sigmoid in forward
        )
        self.model = self.model.to(device)
        self.model.eval()
        
        print(f"Loaded pre-trained model: {encoder} (ImageNet weights)")
    
    def predict(self, image_path: str, threshold: float = 0.5) -> dict:
        """
        Predict segmentation for a single CT image
        
        Args:
            image_path: Path to CT image
            threshold: Binary threshold for segmentation
        
        Returns:
            Dict with mask and confidence
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('L')
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        x = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            pred = self.model(x)
            pred = torch.sigmoid(pred)
        
        # Convert to numpy
        mask = pred.cpu().numpy()[0, 0]
        binary_mask = (mask > threshold).astype(np.uint8)
        
        return {
            'mask': binary_mask,
            'probability_map': mask,
            'confidence': float(mask.max())
        }
    
    def predict_from_pil(self, pil_image: Image.Image, threshold: float = 0.5) -> dict:
        """
        Predict segmentation directly from a PIL Image, avoiding disk I/O.
        Used by the orchestrator which already has images in memory.

        Args:
            pil_image: PIL Image (any mode — converted to grayscale internally).
            threshold: Binary threshold for segmentation.

        Returns:
            Dict with mask, probability_map, and confidence.
        """
        img_array = np.array(pil_image.convert('L')).astype(np.float32) / 255.0
        x = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(x)
            pred = torch.sigmoid(pred)

        mask = pred.cpu().numpy()[0, 0]
        binary_mask = (mask > threshold).astype(np.uint8)

        return {
            'mask': binary_mask,
            'probability_map': mask,
            'confidence': float(mask.max()),
        }

    def predict_batch(self, image_paths: list, threshold: float = 0.5) -> list:
        """Predict segmentation for multiple images"""
        results = []
        for path in image_paths:
            result = self.predict(path, threshold)
            result['image_path'] = path
            results.append(result)
        return results
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to: {path}")
    
    def load_model(self, path: str):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from: {path}")


def download_and_setup():
    """Download pre-trained model and save weights"""
    print("Setting up pre-trained segmentation model...")
    
    # Create model with pretrained encoder
    model = PreTrainedSegmenter(encoder='resnet34', encoder_weights='imagenet')
    
    # Save the model weights
    save_path = 'models/pretrained_unet_resnet34.pth'
    Path(save_path).parent.mkdir(exist_ok=True)
    model.save_model(save_path)
    
    print("\nModel ready! You can now use it for inference.")
    return model


if __name__ == "__main__":
    # Test the model
    model = PreTrainedSegmenter(encoder='resnet34')
    
    # Test on a sample image
    import os
    data_dir = "/home/aryanmarwah/.cache/kagglehub/datasets/theoviel/rsna-abdominal-trauma-detection-png-pt1/extracted"
    test_images = [os.path.join(data_dir, f) for f in os.listdir(data_dir)[:3]]
    
    print(f"\nTesting on {len(test_images)} images...")
    for img_path in test_images:
        result = model.predict(img_path)
        print(f"  {os.path.basename(img_path)}: confidence={result['confidence']:.3f}, "
              f"bleeding_pixels={result['mask'].sum()}")
