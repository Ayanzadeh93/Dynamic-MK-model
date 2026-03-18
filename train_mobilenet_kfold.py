"""
Training Script for MobileNetV2
Binary Classification on Medical Images
Uses Train/Test split from dataset2
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import time
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

from model_mobilenet import create_mobilenet_model, get_optimizer, get_scheduler
from dataset import (load_train_test_datasets, create_dataloaders, 
                     get_class_weights, print_dataset_info)
from evaluation_metrics import BinaryClassificationMetrics, evaluate_model


class Trainer:
    """
    Trainer for MobileNetV2 with Train/Test split
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trainer with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Print device information
        print(f"\n{'='*80}")
        print(f"Device Information")
        print(f"{'='*80}")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("WARNING: CUDA not available. Training will use CPU (much slower).")
        print(f"{'='*80}")
        
        # Create results directory
        self.results_dir = config.get('results_dir', 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Save configuration
        with open(os.path.join(self.results_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"\n{'='*80}")
        print(f"Training Configuration")
        print(f"{'='*80}")
        for key, value in config.items():
            print(f"{key:30s}: {value}")
        print(f"{'='*80}\n")
        
    def train_one_epoch(self, model: nn.Module, train_loader: DataLoader, 
                       criterion, optimizer, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')
        
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, model: nn.Module, val_loader: DataLoader, 
                criterion) -> Tuple[float, float]:
        """
        Validate the model
        
        Args:
            model: PyTorch model
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train_model(self, train_paths: List[str], train_labels: List[int],
                   test_paths: List[str], test_labels: List[int]) -> Tuple[Dict, Dict]:
        """
        Train the model with Train/Test split
        
        Args:
            train_paths: Training image paths
            train_labels: Training labels
            test_paths: Test image paths
            test_labels: Test labels
            
        Returns:
            Tuple of (metrics_dict, history_dict)
        """
        print(f"\n{'='*80}")
        print(f"Starting Training")
        print(f"{'='*80}")
        print(f"{'='*80}")
        
        # Print dataset statistics for this fold
        print_dataset_info(train_labels, test_labels)
        
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            train_paths, train_labels,
            test_paths, test_labels,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            image_size=self.config['image_size']
        )
        
        # Create model
        model = create_mobilenet_model(
            pretrained=self.config['pretrained'],
            num_classes=self.config['num_classes'],
            dropout=self.config['dropout'],
            freeze_backbone=self.config.get('freeze_backbone', False),
            device=self.device
        )
        
        # Loss function with class weights
        class_weights = get_class_weights(train_labels, self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer and scheduler
        optimizer = get_optimizer(
            model,
            learning_rate=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            optimizer_type=self.config['optimizer']
        )
        
        scheduler = get_scheduler(
            optimizer,
            scheduler_type=self.config['scheduler'],
            num_epochs=self.config['num_epochs'],
            eta_min=self.config.get('eta_min', 1e-6)
        )
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.config['num_epochs']):
            # Train
            train_loss, train_acc = self.train_one_epoch(
                model, train_loader, criterion, optimizer, epoch
            )
            
            # Validate
            val_loss, val_acc = self.validate(model, val_loader, criterion)
            
            # Update scheduler
            if self.config['scheduler'].lower() == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print epoch results
            print(f'\nEpoch {epoch+1}/{self.config["num_epochs"]}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}')
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                
                # Save model checkpoint
                checkpoint_path = os.path.join(
                    self.results_dir, 
                    'best_model.pth'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, checkpoint_path)
                
                print(f'  *** New best model saved! (Val Acc: {val_acc:.4f}) ***')
            else:
                patience_counter += 1
            
            # Early stopping
            if self.config.get('early_stopping', False):
                patience = self.config.get('patience', 10)
                if patience_counter >= patience:
                    print(f'\nEarly stopping triggered after {epoch+1} epochs')
                    break
        
        print(f'\nBest validation accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}')
        
        # Load best model for evaluation
        checkpoint = torch.load(os.path.join(self.results_dir, 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate on validation set
        fold_results_dir = os.path.join(self.results_dir, '')
        os.makedirs(fold_results_dir, exist_ok=True)
        
        metrics_calculator = BinaryClassificationMetrics(save_dir=fold_results_dir)
        
        # Get predictions and metrics
        metrics, y_true, y_pred, y_pred_proba = evaluate_model(
            model, val_loader, self.device, metrics_calculator, fold_name=''
        )
        
        # Generate plots
        metrics_calculator.plot_confusion_matrix(
            y_true, y_pred, 
            title='Confusion Matrix',
            filename='confusion_matrix.png'
        )
        
        metrics_calculator.plot_roc_curve(
            y_true, y_pred_proba,
            title='ROC Curve',
            filename='roc_curve.png'
        )
        
        metrics_calculator.plot_precision_recall_curve(
            y_true, y_pred_proba,
            title='Precision-Recall Curve',
            filename='pr_curve.png'
        )
        
        metrics_calculator.plot_training_history(
            history,
            filename='training_history.png'
        )
        
        metrics_calculator.generate_classification_report(
            y_true, y_pred,
            filename='classification_report.txt'
        )
        
        # Print comprehensive metrics to console
        print(f"\n{'='*80}")
        print("FINAL EVALUATION METRICS")
        print(f"{'='*80}")
        print(f"Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision:       {metrics['precision']:.4f}")
        print(f"Recall:          {metrics['recall']:.4f}")
        print(f"F1-Score:        {metrics['f1_score']:.4f}")
        print(f"Specificity:     {metrics['specificity']:.4f}")
        print(f"Sensitivity:     {metrics['sensitivity']:.4f}")
        if 'roc_auc' in metrics:
            print(f"ROC-AUC:         {metrics['roc_auc']:.4f}")
        if 'avg_precision' in metrics:
            print(f"Avg Precision:   {metrics['avg_precision']:.4f}")
        print(f"{'='*80}\n")
        
        # Add best epoch to metrics
        metrics['best_epoch'] = best_epoch + 1
        
        return metrics, history
    
    def train(self, data_dir: str) -> Dict:
        """
        Execute training with Train/Test split
        
        Args:
            data_dir: Root directory of dataset
            
        Returns:
            List of metrics for each fold
        """
        print(f"\n{'='*80}")
        print(f"Starting Training with Train/Test Split")
        print(f"{'='*80}\n")
        
        # Load train and test datasets
        train_paths, train_labels, test_paths, test_labels = load_train_test_datasets(data_dir)
        
        print(f"Training images: {len(train_paths)}")
        print(f"Test images: {len(test_paths)}")
        print()
        
        start_time = time.time()
        
        # Train model
        metrics, history = self.train_model(
            train_paths, train_labels, test_paths, test_labels
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Save final summary
        with open(os.path.join(self.results_dir, 'training_summary.txt'), 'w') as f:
            f.write("="*80 + "\n")
            f.write("TRAINING SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Model: MobileNetV2\n")
            f.write(f"Total Training Time: {total_time/3600:.2f} hours ({total_time/60:.2f} minutes)\n")
            f.write(f"Training Images: {len(train_paths)}\n")
            f.write(f"Test Images: {len(test_paths)}\n\n")
            f.write("Final Metrics:\n")
            f.write("-"*80 + "\n")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"{key:25s}: {value:.4f}\n")
                else:
                    f.write(f"{key:25s}: {value}\n")
            f.write("="*80 + "\n")
        
        # Print final metrics summary
        print(f"\n{'='*80}")
        print("FINAL TEST SET METRICS")
        print(f"{'='*80}")
        print(f"Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision:       {metrics['precision']:.4f}")
        print(f"Recall:          {metrics['recall']:.4f}")
        print(f"F1-Score:        {metrics['f1_score']:.4f}")
        print(f"Specificity:     {metrics['specificity']:.4f}")
        print(f"Sensitivity:     {metrics['sensitivity']:.4f}")
        if 'roc_auc' in metrics:
            print(f"ROC-AUC:         {metrics['roc_auc']:.4f}")
        if 'avg_precision' in metrics:
            print(f"Avg Precision:   {metrics['avg_precision']:.4f}")
        print(f"{'='*80}")
        
        print(f"\nTotal training time: {total_time/3600:.2f} hours ({total_time/60:.2f} minutes)")
        print(f"Results saved to: {self.results_dir}")
        print(f"\n{'='*80}\n")
        
        return metrics


def main():
    """
    Main function to run training with Train/Test split
    """
    # Configuration
    config = {
        # Dataset
        'data_dir': 'dataset2',
        'num_classes': 2,
        'image_size': 224,
        
        
        
        # Model
        'pretrained': True,
        'dropout': 0.5,
        'freeze_backbone': False,  # Train all layers (backbone unfrozen)
        
        # Training
        'num_epochs': 100,
        'batch_size': 16,
        'learning_rate': 0.01,  # Higher LR for classifier-only training (frozen backbone)
        'weight_decay': 1e-4,
        'optimizer': 'adam',  # 'adam', 'adamw', or 'sgd'
        'scheduler': 'cosine',  # 'cosine', 'step', or 'plateau'
        'eta_min': 1e-6,
        
        # Early stopping (optional - set to True to enable, False to train all epochs)
        'early_stopping': False,  # Default: False (train all epochs)
        'patience': 15,  # Only used if early_stopping is True
        
        # Data loading
        'num_workers': 0,  # Set to 0 for Windows compatibility (multiprocessing issues)
        
        # Results
        'results_dir': 'final_results/mobilenet'
    }
    
    # Create trainer
    trainer = Trainer(config)
    
    # Run k-fold cross validation
    metrics = trainer.train(config['data_dir'])
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nAll results saved to: {config['results_dir']}")
    print("\nGenerated files:")
    print("  - best_model.pth: Best model checkpoint")
    print("  - confusion_matrix.png: Confusion matrix")
    print("  - roc_curve.png: ROC curve")
    print("  - pr_curve.png: Precision-Recall curve")
    print("  - training_history.png: Training history plots")
    print("  - classification_report.txt: Detailed classification report")
    print("  - training_summary.txt: Complete training summary")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()


