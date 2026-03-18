"""
Reusable Evaluation Metrics Module for Binary Classification
Can be imported and used across different models
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os


class BinaryClassificationMetrics:
    """
    Comprehensive evaluation metrics for binary classification tasks
    """
    
    def __init__(self, save_dir: str = "results"):
        """
        Initialize metrics calculator
        
        Args:
            save_dir: Directory to save plots and reports
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate all classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'specificity': self._calculate_specificity(y_true, y_pred),
            'sensitivity': recall_score(y_true, y_pred, zero_division=0),  # Same as recall
        }
        
        # Add ROC-AUC if probabilities are available
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
            except:
                metrics['roc_auc'] = 0.0
                metrics['avg_precision'] = 0.0
                
        return metrics
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (True Negative Rate)"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return specificity
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             title: str = "Confusion Matrix", 
                             filename: str = "confusion_matrix.png"):
        """
        Plot and save confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            filename: Output filename
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Class 0', 'Class 1'],
                   yticklabels=['Class 0', 'Class 1'])
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                       title: str = "ROC Curve",
                       filename: str = "roc_curve.png"):
        """
        Plot and save ROC curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Plot title
            filename: Output filename
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                    title: str = "Precision-Recall Curve",
                                    filename: str = "precision_recall_curve.png"):
        """
        Plot and save Precision-Recall curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Plot title
            filename: Output filename
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_training_history(self, history: Dict[str, List[float]], 
                             filename: str = "training_history.png"):
        """
        Plot training and validation loss/accuracy over epochs
        
        Args:
            history: Dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'
            filename: Output filename
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        if 'train_loss' in history and 'val_loss' in history:
            axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
            axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
            axes[0].set_xlabel('Epoch', fontsize=12)
            axes[0].set_ylabel('Loss', fontsize=12)
            axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
            axes[0].legend(fontsize=10)
            axes[0].grid(alpha=0.3)
        
        # Plot accuracy
        if 'train_acc' in history and 'val_acc' in history:
            axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
            axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Accuracy', fontsize=12)
            axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
            axes[1].legend(fontsize=10)
            axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_kfold_metrics(self, fold_metrics: List[Dict[str, float]], 
                          filename: str = "kfold_metrics.png"):
        """
        Plot metrics across all k-folds
        
        Args:
            fold_metrics: List of metric dictionaries for each fold
            filename: Output filename
        """
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(fold_metrics)
        
        # Select key metrics to plot
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'specificity']
        metrics_to_plot = [m for m in metrics_to_plot if m in df.columns]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics_to_plot):
            if idx < len(axes):
                values = df[metric].values
                folds = np.arange(1, len(values) + 1)
                
                axes[idx].plot(folds, values, marker='o', linewidth=2, markersize=8)
                axes[idx].axhline(y=values.mean(), color='r', linestyle='--', 
                                 label=f'Mean: {values.mean():.3f}')
                axes[idx].set_xlabel('Fold', fontsize=11)
                axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
                axes[idx].set_title(f'{metric.replace("_", " ").title()} Across Folds', 
                                   fontsize=12, fontweight='bold')
                axes[idx].legend(fontsize=9)
                axes[idx].grid(alpha=0.3)
                axes[idx].set_xticks(folds)
        
        # Hide unused subplots
        for idx in range(len(metrics_to_plot), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_metrics_boxplot(self, fold_metrics: List[Dict[str, float]], 
                            filename: str = "metrics_boxplot.png"):
        """
        Create boxplot of metrics across all folds
        
        Args:
            fold_metrics: List of metric dictionaries for each fold
            filename: Output filename
        """
        df = pd.DataFrame(fold_metrics)
        
        # Select key metrics
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'specificity']
        metrics_to_plot = [m for m in metrics_to_plot if m in df.columns]
        
        plt.figure(figsize=(12, 6))
        data_to_plot = [df[metric].values for metric in metrics_to_plot]
        
        bp = plt.boxplot(data_to_plot, labels=[m.replace('_', ' ').title() for m in metrics_to_plot],
                        patch_artist=True, showmeans=True)
        
        # Customize boxplot colors
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        plt.ylabel('Score', fontsize=12)
        plt.title('Distribution of Metrics Across All Folds', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      filename: str = "classification_report.txt"):
        """
        Generate and save detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            filename: Output filename
        """
        report = classification_report(y_true, y_pred, 
                                      target_names=['Class 0', 'Class 1'],
                                      digits=4)
        
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            f.write("Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)
            
    def save_metrics_summary(self, fold_metrics: List[Dict[str, float]], 
                           filename: str = "metrics_summary.csv"):
        """
        Save metrics summary to CSV file
        
        Args:
            fold_metrics: List of metric dictionaries for each fold
            filename: Output filename
        """
        df = pd.DataFrame(fold_metrics)
        
        # Add fold numbers
        df.insert(0, 'Fold', range(1, len(fold_metrics) + 1))
        
        # Calculate mean and std
        summary_stats = df.describe().loc[['mean', 'std']]
        
        # Save to CSV
        filepath = os.path.join(self.save_dir, filename)
        df.to_csv(filepath, index=False, float_format='%.4f')
        
        # Save summary statistics
        summary_filepath = os.path.join(self.save_dir, "metrics_statistics.csv")
        summary_stats.to_csv(summary_filepath, float_format='%.4f')
        
        return df, summary_stats
    
    def print_metrics_summary(self, fold_metrics: List[Dict[str, float]]):
        """
        Print formatted metrics summary to console
        
        Args:
            fold_metrics: List of metric dictionaries for each fold
        """
        df = pd.DataFrame(fold_metrics)
        
        print("\n" + "=" * 80)
        print("K-FOLD CROSS VALIDATION RESULTS SUMMARY")
        print("=" * 80)
        print(f"\nNumber of Folds: {len(fold_metrics)}")
        print("\nMetrics (Mean ± Std):")
        print("-" * 80)
        
        for col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            print(f"{col.replace('_', ' ').title():25s}: {mean_val:.4f} ± {std_val:.4f}")
        
        print("=" * 80 + "\n")


def evaluate_model(model, dataloader, device, metrics_calculator: BinaryClassificationMetrics,
                  fold_name: str = "test") -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on a given dataloader
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        metrics_calculator: BinaryClassificationMetrics instance
        fold_name: Name for the fold (for filename purposes)
        
    Returns:
        Tuple of (metrics_dict, y_true, y_pred, y_pred_proba)
    """
    import torch
    
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of class 1
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_pred_proba = np.array(all_probs)
    
    # Calculate metrics
    metrics = metrics_calculator.calculate_metrics(y_true, y_pred, y_pred_proba)
    
    return metrics, y_true, y_pred, y_pred_proba







