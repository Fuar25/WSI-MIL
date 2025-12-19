from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class RunTimeConfig:
    # ==================== Dataset Configuration ====================
    dataset_type: str = 'h5'  # 'npy' or 'h5'
    
    # For 'npy' dataset (CSV-based)
    root_dir: str = './data/ctranspath'  # Root directory for NPY files
    csv_path: str = './data/labels.csv'
    label_col: str = 'GT'
    sample_id_col: str = 'file_name'
    feature_suffix: str = '_ctranspath.npy'
    
    # For 'h5' dataset (directory-based, each directory is a class)
    data_paths: Optional[Dict[str, str]] = field(default_factory=lambda: {
        'positive': '/mnt/gml/GML/Experiments/Experiment3/MALT/10x_256px_0px_overlap/slide_features_chief',
        'negative': '/mnt/gml/GML/Experiments/Experiment3/Reactive/10x_256px_0px_overlap/slide_features_chief'
    })
    feature_key: str = 'features'
    binary_mode: bool = True

    # ==================== Model Configuration ====================
    model_name: str = 'abmil'  # 'abmil' or 'linear_probe'
    input_dim: int = 1536
    hidden_dim: int = 512
    num_classes: int = 1  # Will be auto-set for H5 dataset
    dropout: float = 0.2
    n_heads: int = 1
    gated: bool = True
    
    # ==================== Training Configuration ====================
    # General
    epochs: int = 30
    batch_size: int = 1
    learning_rate: float = 0.0004
    weight_decay: float = 1e-5
    seed: int = 42
    device: str = 'cuda'  # Use 'cpu' if CUDA compatibility issues
    patience: int = 5  # Number of epochs to wait for improvement before early stopping
    
    # Cross-Validation specific
    k_folds: int = 5  # Number of folds for cross-validation
    val_ratio: float = 0.1  # Validation split ratio within each fold (train:val = 9:1)

    # Full dataset training specific
    best_epochs = 9

    # ==================== Logging & Saving ====================
    save_dir: str = './experiments'
    save_best_only: bool = True  # Only save best models based on validation loss

runtime_config = RunTimeConfig()

@dataclass
class VisualizationConfig:
    """Configuration for ABMIL Visualizer pipeline"""
    device: str = 'cuda'
    abmil_weights_path: str = './experiments/model_deployment.pth'
    job_dir: str = './visualization'

    # Segmentation config
    segmentation_model_name: str = "grandqc"

    # Patch extraction config
    target_mag: int = 20
    patch_size: int = 512
    overlap: int = 256

    # Feature extraction config
    patch_encoder_name: str = "uni_v2"

    # Visualization config
    vis_level: int = 1
    num_top_patches: int = 10
    normalize_heatmap: bool = True

    # ==================== Model Configuration ====================
    input_dim: int = 1536
    hidden_dim: int = 512
    num_classes: int = 1  # Will be auto-set for H5 dataset
    dropout: float = 0.2
    n_heads: int = 1
    gated: bool = True


visualization_config = VisualizationConfig()