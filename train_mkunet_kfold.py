"""
Training Script for MK-UNet 4-Model Ablation Study (M1-M4)
Binary Classification on Medical Images
Uses Train/Test split from dataset2
Trains all four variants sequentially and generates a comprehensive comparison report.
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

from model_mkunet import (
    create_mkunet_original, create_mkunet_dynamic,
    create_mkunet_freq, create_mkunet_freq_dynamic
)
from dataset import (load_train_test_datasets, create_dataloaders,
                     get_class_weights, print_dataset_info)
from evaluation_metrics import BinaryClassificationMetrics, evaluate_model


class MKUNetTrainer:
    """
    Trainer for MK-UNet Ablation Study.
    Supports M1 (Original), M2 (Dynamic), M3 (Freq), and M4 (Freq+Dynamic).
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"\n{'='*80}")
        print(f"Device Information")
        print(f"{'='*80}")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("WARNING: CUDA not available. Training will use CPU.")
        print(f"{'='*80}")

        self.results_dir = config.get('results_dir', 'results')
        os.makedirs(self.results_dir, exist_ok=True)

        with open(os.path.join(self.results_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

    def train_one_epoch(self, model, train_loader, criterion, optimizer, epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })

        return running_loss / total, correct / total

    def validate(self, model, val_loader, criterion):
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

        return running_loss / total, correct / total

    def train_model(self, model_factory, model_name, train_paths, train_labels,
                    test_paths, test_labels):
        """Train a single model variant."""
        results_dir = os.path.join(self.results_dir, model_name)
        os.makedirs(results_dir, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Training Stage: {model_name}")
        print(f"{'='*80}")

        # Instantiate model via factory
        model = model_factory(
            num_classes=self.config['num_classes'],
            dropout=self.config['dropout'],
            device=self.device
        )

        train_loader, val_loader = create_dataloaders(
            train_paths, train_labels, test_paths, test_labels,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            image_size=self.config['image_size']
        )

        class_weights = get_class_weights(train_labels, self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config['num_epochs'], eta_min=1e-6
        )

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0.0
        best_epoch = 0
        start_time = time.time()

        for epoch in range(self.config['num_epochs']):
            train_loss, train_acc = self.train_one_epoch(
                model, train_loader, criterion, optimizer, epoch
            )
            val_loss, val_acc = self.validate(model, val_loader, criterion)
            scheduler.step()

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                }, os.path.join(results_dir, 'best_model.pth'))
                print(f'  [Epoch {epoch+1}] New best model! Val Acc: {val_acc:.4f}')

        elapsed = time.time() - start_time

        # Evaluation on best model
        checkpoint = torch.load(os.path.join(results_dir, 'best_model.pth'),
                                map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        metrics_calculator = BinaryClassificationMetrics(save_dir=results_dir)
        metrics, y_true, y_pred, y_pred_proba = evaluate_model(
            model, val_loader, self.device, metrics_calculator
        )

        # Plotting
        metrics_calculator.plot_confusion_matrix(y_true, y_pred, filename='confusion_matrix.png')
        metrics_calculator.plot_roc_curve(y_true, y_pred_proba, filename='roc_curve.png')
        metrics_calculator.plot_training_history(history, filename='training_history.png')
        metrics_calculator.generate_classification_report(y_true, y_pred, filename='report.txt')

        metrics['best_epoch'] = best_epoch + 1
        metrics['training_time_min'] = elapsed / 60
        metrics['params'] = sum(p.numel() for p in model.parameters())

        # Cleanup
        del model
        torch.cuda.empty_cache()

        return metrics

    def run_ablation_study(self, data_dir):
        """Train M1-M4 and compare them. Smarter version to skip completed ones."""
        train_paths, train_labels, test_paths, test_labels = load_train_test_datasets(data_dir)
        print_dataset_info(train_labels, test_labels)

        factories = [
            (create_mkunet_original,      'M1_Original'),
            (create_mkunet_dynamic,       'M2_Dynamic'),
            (create_mkunet_freq,          'M3_Freq'),
            (create_mkunet_freq_dynamic,  'M4_FreqDynamic'),
        ]

        # Load existing results if they exist to allow resumption
        results_json = os.path.join(self.results_dir, 'ablation_results.json')
        all_results = {}
        if os.path.exists(results_json):
            try:
                with open(results_json, 'r') as f:
                    all_results = json.load(f)
                print(f"Loaded existing results for: {list(all_results.keys())}")
            except Exception as e:
                print(f"Could not load existing results: {e}")

        for factory_fn, name in factories:
            if name in all_results:
                print(f"\nSkipping {name} (already exists).")
                continue
                
            metrics = self.train_model(factory_fn, name, train_paths, train_labels, test_paths, test_labels)
            all_results[name] = metrics
            # Save intermediate results
            self._save_comparison(all_results)

        return all_results

    def _save_comparison(self, results):
        """Save comparison table and JSON."""
        filepath = os.path.join(self.results_dir, 'ablation_comparison.txt')
        with open(filepath, 'w') as f:
            f.write("="*100 + "\n")
            f.write("  MK-UNet ABLATION STUDY RESULTS\n")
            f.write("="*100 + "\n\n")

            metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'params', 'training_time_min']
            
            header = f"{'Model':20s}"
            for k in metric_keys:
                header += f" | {k:>10s}"
            f.write(header + "\n" + "-"*100 + "\n")

            m1_results = results.get('M1_Original', {})

            for name, metrics in results.items():
                row = f"{name:20s}"
                for k in metric_keys:
                    val = metrics.get(k, 0)
                    if k == 'params':
                        row += f" | {val:>10,}"
                    else:
                        row += f" | {val:>10.4f}"
                f.write(row + "\n")
                
                # Delta vs M1
                if name != 'M1_Original':
                    d_row = f"{'':20s}"
                    for k in metric_keys:
                        if k == 'params':
                            d = metrics.get(k, 0) - m1_results.get(k, 0)
                            d_row += f" | {f'+{d:,}':>10s}"
                        else:
                            d = metrics.get(k, 0) - m1_results.get(k, 0)
                            sign = '+' if d >= 0 else ''
                            d_row += f" | {f'{sign}{d:.4f}':>10s}"
                    f.write(d_row + " (vs M1)\n")
            
            f.write("\n" + "="*100 + "\n")

        with open(os.path.join(self.results_dir, 'ablation_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nFinal comparison saved to: {filepath}")


def main():
    config = {
        'data_dir': 'dataset2',
        'num_classes': 2,
        'image_size': 224,
        'num_epochs': 100,
        'batch_size': 16,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'dropout': 0.5,
        'num_workers': 0,
        'results_dir': 'final_results/mkunet_ablation_study'
    }

    trainer = MKUNetTrainer(config)
    trainer.run_ablation_study(config['data_dir'])


if __name__ == '__main__':
    main()
