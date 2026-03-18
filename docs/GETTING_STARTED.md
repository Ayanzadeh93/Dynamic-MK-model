# Getting Started Guide

## Quick Start (3 Steps)

### Step 1: Install Requirements

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch 2.1.0
- TorchVision 0.16.0
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn
- Pillow (for image loading)
- tqdm (for progress bars)

### Step 2: Verify Setup

```bash
python test_setup.py
```

This will check:
- All packages are installed correctly
- CUDA availability (if you have a GPU)
- Dataset is properly structured
- Model can be created and run

### Step 3: Start Training

```bash
python train_kfold.py
```

Or use the quick start script:

```bash
python quick_start.py
```

That's it! The training will run automatically with 20-fold cross validation.

---

## What Happens During Training

1. **Data Loading**: Loads all images from `dataset/0/` and `dataset/1/`
2. **K-Fold Split**: Divides data into 20 stratified folds
3. **For Each Fold**:
   - Train SqueezeNet with ImageNet pretrained weights
   - Apply early stopping based on validation accuracy
   - Save best model checkpoint
   - Generate evaluation metrics and plots
4. **Final Summary**: Aggregates results from all folds

## Training Configuration

You can modify settings in `train_kfold.py`:

```python
config = {
    'num_folds': 20,          # Number of folds (default: 20)
    'num_epochs': 50,         # Max epochs per fold (default: 50)
    'batch_size': 16,         # Batch size (default: 16)
    'learning_rate': 0.001,   # Initial learning rate (default: 0.001)
    'early_stopping': True,   # Enable early stopping (default: True)
    'patience': 15,           # Early stopping patience (default: 15)
}
```

### Common Adjustments

**If training is too slow:**
- Reduce `num_folds` to 10 or 5 for faster experiments
- Reduce `num_epochs` to 30
- Increase `batch_size` to 32 (if you have enough GPU memory)

**If you get out of memory errors:**
- Reduce `batch_size` to 8 or 4
- Set `freeze_backbone: True` to train only the classifier

**For better accuracy:**
- Increase `num_epochs` to 100
- Try different learning rates: 0.0001 or 0.01
- Adjust `dropout` rate (0.3 to 0.7)

## Expected Output

### Console Output
You'll see:
- Configuration summary
- Dataset statistics for each fold
- Training progress with loss/accuracy
- Validation results after each epoch
- Best model checkpoints being saved
- Final summary of all folds

### Generated Files

```
results_squeezenet_kfold20/
├── config.json                          # Configuration used
├── fold_metrics.csv                     # Metrics for each fold
├── metrics_statistics.csv               # Mean ± Std of metrics
├── training_summary.txt                 # Complete summary
├── kfold_metrics_comparison.png         # Metrics across folds
├── metrics_boxplot.png                  # Distribution of metrics
├── best_model_fold_1.pth               # Best model for fold 1
├── best_model_fold_2.pth               # Best model for fold 2
├── ...                                  # (up to fold 20)
└── fold_1/                              # Detailed results for fold 1
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── pr_curve.png
    ├── training_history.png
    └── classification_report.txt
```

## Running Inference

After training, use the best model to make predictions:

**Single image:**
```bash
python inference.py --model results_squeezenet_kfold20/best_model_fold_1.pth --input path/to/image.tif
```

**Directory of images:**
```bash
python inference.py --model results_squeezenet_kfold20/best_model_fold_1.pth --input path/to/images/
```

Results will be saved to `inference_results/predictions.json`

## Training Time Estimates

Based on your dataset (905 images):

- **With GPU (CUDA)**: ~2-4 hours for 20 folds
- **With CPU**: ~10-20 hours for 20 folds

Each fold takes approximately:
- GPU: 6-12 minutes
- CPU: 30-60 minutes

## Monitoring Progress

The training shows:
- Real-time progress bars for each epoch
- Training and validation loss/accuracy
- Learning rate updates
- Best model saves

You can stop training anytime with `Ctrl+C` - best models are already saved.

## Troubleshooting

### Import Errors
```
ModuleNotFoundError: No module named 'torch'
```
**Solution:** Install requirements: `pip install -r requirements.txt`

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch_size in config (try 8 or 4)

### Slow Training
**Solution:** 
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Reduce num_folds for testing (e.g., 5 instead of 20)
- Use smaller num_epochs (e.g., 30 instead of 50)

### No Improvement in Training
**Solution:**
- Check if learning_rate is too high/low
- Try unfreezing backbone: set `freeze_backbone: False`
- Increase num_epochs

## Using with Other Models

To use this framework with a different model:

1. Modify `model.py` to include your architecture
2. Update the model creation in `train_kfold.py`
3. Everything else (data loading, evaluation, plots) stays the same

See `example_use_metrics.py` for examples of using the evaluation module with other models.

## GPU vs CPU

**GPU (Recommended):**
- Much faster training (5-10x speedup)
- Can use larger batch sizes
- Essential for experimentation

**CPU (Works but slower):**
- Training will take longer
- Use smaller batch sizes
- Good for testing setup

## Next Steps

After training completes:

1. **Review Results**: Check `training_summary.txt` for overall performance
2. **Analyze Metrics**: Look at `fold_metrics.csv` to see variation across folds
3. **View Plots**: Open the PNG files to visualize performance
4. **Run Inference**: Use trained models to make predictions on new images
5. **Iterate**: Adjust hyperparameters and retrain if needed

## Support

- Check `README.md` for detailed documentation
- Run `python example_use_metrics.py` to see metric usage examples
- All code is well-commented for easy understanding

---

**Happy Training!** 🚀




