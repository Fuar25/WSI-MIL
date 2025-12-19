# Refactored MIL Training Project with TRIDENT Integration

This project has been refactored to support Object-Oriented Design, K-Fold Cross-Validation, and integration with the TRIDENT library for visualization.

## Project Structure

- `main.py`: The main entry point. Supports training, cross-validation, and visualization modes.
- `core/trainer.py`: Contains the `Trainer` class handling training loops, validation, and CV.
- `core/base.py`: Abstract base classes for Datasets and Models.
- `data_loader/dataset.py`: `WSIFeatureDataset` inheriting from `BaseDataset`.
- `models/mil_models.py`: `ABMIL` model inheriting from `BaseModel`.
- `config/config.py`: Configuration file.

## How to Use

### 1. Configuration
Modify `config/config.py` to set your data paths and hyperparameters.

### 2. Training (Cross-Validation)
To run K-Fold Cross-Validation (default 5 folds):
```bash
python main.py --mode cv --folds 5
```

### 3. Visualization
To visualize attention heatmaps using TRIDENT:
```bash
python main.py --mode vis --model_path experiments/model_fold_1_best.pth --vis_sample 0
```
**Note**: Visualization requires WSI files and coordinate files. You may need to modify `core/trainer.py` or provide paths if your dataset doesn't include them directly.

## TRIDENT Integration
This project integrates with the `TRIDENT` library for heatmap visualization. Ensure `TRIDENT` is installed in your environment.

## Extending
- **Datasets**: Inherit from `BaseDataset` in `core/base.py`.
- **Models**: Inherit from `BaseModel` in `core/base.py`.
