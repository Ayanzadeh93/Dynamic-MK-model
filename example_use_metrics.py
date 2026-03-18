"""
Example: How to use evaluation_metrics.py with your own models

This demonstrates how to integrate the reusable evaluation metrics
module with any binary classification model.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from evaluation_metrics import BinaryClassificationMetrics, evaluate_model


# Example 1: Using metrics with numpy arrays
def example_numpy_metrics():
    """
    Example: Calculate metrics from numpy arrays of predictions
    """
    print("\n" + "="*70)
    print("Example 1: Using metrics with numpy arrays")
    print("="*70)
    
    # Simulate some predictions (replace with your actual predictions)
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.4, 0.3, 0.85, 0.15, 0.92, 0.88])
    
    # Create metrics calculator
    metrics_calc = BinaryClassificationMetrics(save_dir='example_results')
    
    # Calculate all metrics
    metrics = metrics_calc.calculate_metrics(y_true, y_pred, y_pred_proba)
    
    # Print metrics
    print("\nMetrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    # Generate plots
    metrics_calc.plot_confusion_matrix(y_true, y_pred, title="Example Confusion Matrix")
    metrics_calc.plot_roc_curve(y_true, y_pred_proba, title="Example ROC Curve")
    metrics_calc.plot_precision_recall_curve(y_true, y_pred_proba, title="Example PR Curve")
    metrics_calc.generate_classification_report(y_true, y_pred)
    
    print("\n✓ Plots and reports saved to 'example_results/' directory")


# Example 2: Using metrics with PyTorch model and DataLoader
def example_pytorch_model():
    """
    Example: Evaluate a PyTorch model using the metrics module
    """
    print("\n" + "="*70)
    print("Example 2: Using metrics with PyTorch model")
    print("="*70)
    
    # Create a simple dummy model (replace with your actual model)
    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.fc = nn.Linear(10, 2)
        
        def forward(self, x):
            return self.fc(x)
    
    model = DummyModel()
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    # Create dummy dataloader (replace with your actual dataloader)
    from torch.utils.data import TensorDataset
    dummy_data = torch.randn(100, 10)
    dummy_labels = torch.randint(0, 2, (100,))
    dataset = TensorDataset(dummy_data, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Create metrics calculator
    metrics_calc = BinaryClassificationMetrics(save_dir='example_pytorch_results')
    
    # Evaluate model
    metrics, y_true, y_pred, y_pred_proba = evaluate_model(
        model, dataloader, device, metrics_calc, fold_name="test"
    )
    
    # Print metrics
    print("\nModel Evaluation Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    # Generate plots
    metrics_calc.plot_confusion_matrix(y_true, y_pred)
    metrics_calc.plot_roc_curve(y_true, y_pred_proba)
    
    print("\n✓ Results saved to 'example_pytorch_results/' directory")


# Example 3: Using metrics for k-fold cross validation results
def example_kfold_metrics():
    """
    Example: Visualize k-fold cross validation results
    """
    print("\n" + "="*70)
    print("Example 3: Visualizing K-Fold Cross Validation Results")
    print("="*70)
    
    # Simulate k-fold results (replace with your actual fold results)
    fold_metrics = []
    for fold in range(5):
        metrics = {
            'accuracy': 0.85 + np.random.uniform(-0.05, 0.05),
            'precision': 0.83 + np.random.uniform(-0.05, 0.05),
            'recall': 0.82 + np.random.uniform(-0.05, 0.05),
            'f1_score': 0.825 + np.random.uniform(-0.05, 0.05),
            'roc_auc': 0.90 + np.random.uniform(-0.05, 0.05),
            'specificity': 0.88 + np.random.uniform(-0.05, 0.05),
        }
        fold_metrics.append(metrics)
    
    # Create metrics calculator
    metrics_calc = BinaryClassificationMetrics(save_dir='example_kfold_results')
    
    # Generate k-fold specific plots
    metrics_calc.plot_kfold_metrics(fold_metrics)
    metrics_calc.plot_metrics_boxplot(fold_metrics)
    
    # Save metrics summary
    df, summary_stats = metrics_calc.save_metrics_summary(fold_metrics)
    
    # Print summary
    metrics_calc.print_metrics_summary(fold_metrics)
    
    print("\n✓ K-fold visualizations saved to 'example_kfold_results/' directory")


# Example 4: Custom usage - just the parts you need
def example_custom_usage():
    """
    Example: Use only specific functions you need
    """
    print("\n" + "="*70)
    print("Example 4: Custom usage - pick what you need")
    print("="*70)
    
    # Some predictions
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_pred_proba = np.random.rand(100)
    
    # Create metrics calculator
    metrics_calc = BinaryClassificationMetrics(save_dir='example_custom_results')
    
    # Option 1: Just calculate metrics (no plots)
    metrics = metrics_calc.calculate_metrics(y_true, y_pred, y_pred_proba)
    print("\nCalculated Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Option 2: Just generate specific plots you want
    metrics_calc.plot_confusion_matrix(y_true, y_pred)
    print("\n✓ Only confusion matrix saved")
    
    # Option 3: Just generate classification report
    metrics_calc.generate_classification_report(y_true, y_pred)
    print("✓ Classification report saved")


# Example 5: Integration with your own training loop
def example_training_integration():
    """
    Example: How to integrate metrics into your training loop
    """
    print("\n" + "="*70)
    print("Example 5: Integration with training loop")
    print("="*70)
    
    print("""
    # In your training script:
    
    from evaluation_metrics import BinaryClassificationMetrics
    
    # Create metrics calculator
    metrics_calc = BinaryClassificationMetrics(save_dir='my_model_results')
    
    # After each epoch or at the end of training:
    all_preds = []
    all_probs = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    metrics = metrics_calc.calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
    
    # Generate all plots
    metrics_calc.plot_confusion_matrix(all_labels, all_preds)
    metrics_calc.plot_roc_curve(all_labels, all_probs)
    metrics_calc.plot_precision_recall_curve(all_labels, all_probs)
    
    # For k-fold CV:
    fold_metrics = []
    for fold in range(k_folds):
        # ... train fold ...
        fold_metrics.append(metrics)
    
    # Visualize k-fold results
    metrics_calc.plot_kfold_metrics(fold_metrics)
    metrics_calc.save_metrics_summary(fold_metrics)
    """)


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print(" Evaluation Metrics Module - Usage Examples")
    print("="*70)
    print("\nThis demonstrates how to use evaluation_metrics.py with your own models")
    
    # Run examples
    try:
        example_numpy_metrics()
        # example_pytorch_model()  # Uncomment to run
        # example_kfold_metrics()  # Uncomment to run
        # example_custom_usage()   # Uncomment to run
        example_training_integration()
        
        print("\n" + "="*70)
        print("✓ Examples completed successfully!")
        print("="*70)
        print("\nKey takeaways:")
        print("  1. Import: from evaluation_metrics import BinaryClassificationMetrics")
        print("  2. Create: metrics_calc = BinaryClassificationMetrics(save_dir='...')")
        print("  3. Use: metrics = metrics_calc.calculate_metrics(y_true, y_pred, y_proba)")
        print("  4. Plot: metrics_calc.plot_*(...)")
        print("\nFor k-fold CV:")
        print("  - metrics_calc.plot_kfold_metrics(fold_metrics)")
        print("  - metrics_calc.save_metrics_summary(fold_metrics)")
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()







