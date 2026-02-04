from dataclasses import dataclass, field
from typing import Dict, Optional

from .defaults import (
    BINARY_THRESHOLD,
    DEFAULT_DATASET_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_VOTING_STRATEGY,
    PATIENT_ID_PATTERN,
)


@dataclass
class DatasetConfig:
    """Dataset configuration parameters."""

    dataset_name: str = DEFAULT_DATASET_NAME
    data_paths: Optional[Dict[str, str]] = field(
        default_factory=lambda: {
            'positive': '/mnt/6T/GML/Experiments/Experiment3/MALT/10x_256px_0px_overlap/slide_features_chief',
            'negative': '/mnt/6T/GML/Experiments/Experiment3/Reactive/10x_256px_0px_overlap/slide_features_chief',
        }
    )
    feature_key: str = 'features'
    coords_key: str = 'coords'
    patient_id_pattern: str = PATIENT_ID_PATTERN
    binary_mode: Optional[bool] = None


@dataclass
class ModelConfig:
    """Model architecture configuration parameters."""

    model_name: str = DEFAULT_MODEL_NAME
    input_dim: int = 1536
    hidden_dim: int = 512
    attention_dim: Optional[int] = None
    num_classes: int = 1
    dropout: float = 0.2
    gated: bool = True
    n_heads: int = 1
    encoder_dropout: float = 0.2
    classifier_dropout: float = 0.2


@dataclass
class TrainingConfig:
    """Training process configuration parameters."""

    epochs: int = 30
    batch_size: int = 1
    learning_rate: float = 0.0005
    weight_decay: float = 1e-5
    seed: int = 42
    device: str = 'cuda'
    patience: int = 5
    k_folds: int = 5
    val_ratio: float = 0.1
    voting_strategy: str = DEFAULT_VOTING_STRATEGY
    voting_config: Dict[str, float] = field(
        default_factory=lambda: {'threshold': BINARY_THRESHOLD}
    )
    best_epochs: int = 9


@dataclass
class LoggingConfig:
    """Logging and saving configuration parameters"""
    save_dir: str = './experiments'
    save_best_only: bool = True
    log_test_results: bool = True
    test_results_csv: str = 'test_results.csv'
    save_features: bool = False  # Save features for fusion (train + test sets per fold)


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