from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class DatasetConfig:
    """Dataset configuration parameters"""
    dataset_type: str = 'h5'  # 'npy' or 'h5'
    
    # For 'npy' dataset (CSV-based)
    root_dir: str = './data/ctranspath'
    csv_path: str = './data/labels.csv'
    label_col: str = 'GT'
    sample_id_col: str = 'file_name'
    feature_suffix: str = '_ctranspath.npy'
    
    # For 'h5' dataset (directory-based, each directory is a class)
    data_paths: Optional[Dict[str, str]] = field(default_factory=lambda: {
        'positive': '/mnt/6T/GML/Experiments/Experiment3/MALT/10x_256px_0px_overlap/slide_features_chief',
        'negative': '/mnt/6T/GML/Experiments/Experiment3/Reactive/10x_256px_0px_overlap/slide_features_chief'
    })
    feature_key: str = 'features'
    binary_mode: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration parameters"""
    model_name: str = 'abmil'  # 'abmil', 'linear_probe', or MIL-Lab model string
    
    # MIL-Lab integration
    use_mil_lab: bool = True
    mil_lab_model_name: str = 'abmil'  # Specific model name for MIL-Lab builder (e.g. 'clam_sb', 'transmil')
    pretrained: bool = False
    checkpoint_path: str = ''
    
    # Model architecture parameters (ignored if loading specific pretrained model string)
    input_dim: int = 1536
    hidden_dim: int = 512
    num_classes: int = 1  # Will be auto-set for H5 dataset
    dropout: float = 0.2
    n_heads: int = 1
    gated: bool = True


@dataclass
class TrainingConfig:
    """Training process configuration parameters"""
    # General training
    epochs: int = 30
    batch_size: int = 1
    learning_rate: float = 0.0005
    weight_decay: float = 1e-5
    seed: int = 42
    device: str = 'cuda'
    patience: int = 5  # Early stopping patience
    
    # Cross-validation
    k_folds: int = 5
    val_ratio: float = 0.1
    
    # Full dataset training
    best_epochs: int = 9


@dataclass
class LoggingConfig:
    """Logging and saving configuration parameters"""
    save_dir: str = './experiments'
    save_best_only: bool = True
    log_test_results: bool = True
    test_results_csv: str = 'test_results.csv'


@dataclass
class RunTimeConfig:
    """Main runtime configuration containing all sub-configurations"""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


runtime_config = RunTimeConfig()


@dataclass
class SegmentationConfig:
    """Tissue segmentation configuration"""
    segmentation_model_name: str = "grandqc"


@dataclass
class PatchExtractionConfig:
    """Patch extraction configuration"""
    target_mag: int = 20
    patch_size: int = 512
    overlap: int = 256


@dataclass
class FeatureExtractionConfig:
    """Feature extraction configuration"""
    patch_encoder_name: str = "uni_v2"


@dataclass
class VisualizationDisplayConfig:
    """Visualization display configuration"""
    vis_level: int = 1
    num_top_patches: int = 10
    normalize_heatmap: bool = True


@dataclass
class VisualizationConfig:
    """Configuration for ABMIL Visualizer pipeline"""
    # General
    device: str = 'cuda'
    abmil_weights_path: str = './experiments/model_deployment.pth'
    job_dir: str = './visualization'
    
    # Sub-configurations
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    patch_extraction: PatchExtractionConfig = field(default_factory=PatchExtractionConfig)
    feature_extraction: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)
    display: VisualizationDisplayConfig = field(default_factory=VisualizationDisplayConfig)
    
    # Model configuration (reuse from runtime)
    model: ModelConfig = field(default_factory=ModelConfig)


visualization_config = VisualizationConfig()