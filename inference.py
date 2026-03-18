"""
Reusable Inference Script for Binary Classification Models
Can be used with any trained model from this framework
"""

import os
import torch
import numpy as np
from PIL import Image
import argparse
from typing import List, Tuple
import json

from model import SqueezeNetBinaryClassifier
from dataset import get_val_transforms
from evaluation_metrics import BinaryClassificationMetrics


class ModelInference:
    """
    Inference class for making predictions with trained models
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize inference
        
        Args:
            model_path: Path to saved model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = get_val_transforms(image_size=224)
        
        # Load model
        self.load_model(model_path)
        
    def load_model(self, model_path: str):
        """
        Load trained model from checkpoint
        
        Args:
            model_path: Path to model checkpoint
        """
        print(f"Loading model from: {model_path}")
        
        # Create model architecture
        self.model = SqueezeNetBinaryClassifier(pretrained=False, num_classes=2)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded successfully!")
            if 'val_acc' in checkpoint:
                print(f"Model validation accuracy: {checkpoint['val_acc']:.4f}")
        else:
            self.model.load_state_dict(checkpoint)
            print(f"Model loaded successfully!")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def predict_single_image(self, image_path: str) -> Tuple[int, float, np.ndarray]:
        """
        Predict class for a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (predicted_class, confidence, probabilities)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
        
        return predicted_class, confidence, probabilities
    
    def predict_batch(self, image_paths: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict classes for multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Tuple of (predicted_classes, confidences, all_probabilities)
        """
        all_predictions = []
        all_confidences = []
        all_probabilities = []
        
        for img_path in image_paths:
            pred_class, confidence, probs = self.predict_single_image(img_path)
            all_predictions.append(pred_class)
            all_confidences.append(confidence)
            all_probabilities.append(probs)
        
        return (np.array(all_predictions), 
                np.array(all_confidences), 
                np.array(all_probabilities))
    
    def predict_directory(self, directory: str, save_results: bool = True, 
                         output_dir: str = 'inference_results') -> dict:
        """
        Predict classes for all images in a directory
        
        Args:
            directory: Directory containing images
            save_results: Whether to save results to file
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing predictions and statistics
        """
        # Get all image files
        image_files = []
        for file in os.listdir(directory):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                image_files.append(os.path.join(directory, file))
        
        print(f"\nFound {len(image_files)} images in {directory}")
        
        if len(image_files) == 0:
            print("No images found!")
            return {}
        
        # Make predictions
        print("Making predictions...")
        predictions, confidences, probabilities = self.predict_batch(image_files)
        
        # Prepare results
        results = {
            'image_files': [os.path.basename(f) for f in image_files],
            'predictions': predictions.tolist(),
            'confidences': confidences.tolist(),
            'probabilities': probabilities.tolist()
        }
        
        # Calculate statistics
        unique, counts = np.unique(predictions, return_counts=True)
        stats = {
            'total_images': len(image_files),
            'class_distribution': {int(c): int(count) for c, count in zip(unique, counts)},
            'average_confidence': float(np.mean(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences))
        }
        
        results['statistics'] = stats
        
        # Print statistics
        print(f"\nPrediction Statistics:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Class distribution:")
        for class_idx, count in stats['class_distribution'].items():
            print(f"    Class {class_idx}: {count} images ({count/stats['total_images']*100:.1f}%)")
        print(f"  Average confidence: {stats['average_confidence']:.4f}")
        print(f"  Confidence range: [{stats['min_confidence']:.4f}, {stats['max_confidence']:.4f}]")
        
        # Save results
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, 'predictions.json')
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"\nResults saved to: {output_file}")
        
        return results


def main():
    """
    Main function for command-line inference
    """
    parser = argparse.ArgumentParser(description='Run inference on trained SqueezeNet model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='inference_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Create inference object
    inferencer = ModelInference(args.model, device=args.device)
    
    # Check if input is file or directory
    if os.path.isfile(args.input):
        # Single image inference
        print(f"\nRunning inference on single image: {args.input}")
        pred_class, confidence, probs = inferencer.predict_single_image(args.input)
        
        print(f"\nPrediction Results:")
        print(f"  Predicted Class: {pred_class}")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Probabilities: Class 0 = {probs[0]:.4f}, Class 1 = {probs[1]:.4f}")
        
    elif os.path.isdir(args.input):
        # Directory inference
        print(f"\nRunning inference on directory: {args.input}")
        results = inferencer.predict_directory(args.input, save_results=True, output_dir=args.output)
    
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == '__main__':
    main()







