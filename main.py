import argparse
import sys

from config.config import runtime_config, visualization_config
from data_loader.dataset import WSIFeatureDataset, H5FeatureDataset
from models.mil_models import ABMIL
from models.linear_probe import LinearProbe
from core.trainer import Trainer
from core.abmil_visualizer import ABMIL_Visualizer

def main():
    parser = argparse.ArgumentParser(description="WSI Training with TRIDENT Integration")
    parser.add_argument('--mode', type=str, default='cv',
                        help='Mode: train (full dataset for deployment), cv (cross-validation), vis (visualization)')
    parser.add_argument('--folds', type=int, default=None, help='Number of folds for cross-validation (overrides runtime_config)')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model for visualization')
    parser.add_argument('--wsi_path', type=str, default=None, help='Path to WSI file for visualization')
    
    args = parser.parse_args()

    # Initialize Dataset
    if runtime_config.dataset_type == 'h5':
        if runtime_config.data_paths is None:
            print("Error: runtime_config.data_paths must be provided for H5 dataset.")
            return
        dataset = H5FeatureDataset(
            data_paths=runtime_config.data_paths,
            feature_key=runtime_config.feature_key,
            binary_mode=runtime_config.binary_mode
        )
        
        # Auto-configure num_classes based on dataset
        # For binary classification (2 classes), use num_classes=1
        # For multi-class (>2 classes), use num_classes=number of classes
        num_dataset_classes = len(dataset.classes)
        if num_dataset_classes == 2:
            runtime_config.num_classes = 1  # Binary classification
            print(f"Auto-configured num_classes to 1 (binary classification) for {num_dataset_classes} classes.")
        else:
            # runtime_config.num_classes = num_dataset_classes
            # print(f"Auto-configured num_classes to {runtime_config.num_classes} (multi-class).")
            raise ValueError(f"{num_dataset_classes} classes has not been supported!")

    elif runtime_config.dataset_type == 'h5':
        dataset = WSIFeatureDataset(
            root_dir=runtime_config.root_dir,
            csv_path=runtime_config.csv_path,
            label_col=runtime_config.label_col,
            sample_id_col=runtime_config.sample_id_col,
            feature_suffix=runtime_config.feature_suffix
        )
    else:
        raise TypeError(f"{runtime_config.dataset_type} type has not been supported!")
    
    if len(dataset) == 0:
        print("Error: No data found.")
        return

    # Initialize Trainer
    if runtime_config.model_name == 'abmil':
        model_class = ABMIL
    elif runtime_config.model_name == 'linear_probe':
        model_class = LinearProbe
    else:
        raise ValueError(f"Model {runtime_config.model_name} not supported.")

    trainer = Trainer(model_class, dataset, runtime_config)
    
    if args.mode == 'train':
        trainer.train_full_dataset()
        
    elif args.mode == 'cv':
        k_folds = args.folds if args.folds is not None else runtime_config.k_folds
        trainer.cross_validate(k_folds=k_folds)

    elif args.mode == 'vis':
        if args.model_path is None:
            print("Please provide --model_path for visualization.")
            return
        
        if args.wsi_path is None:
            print("Please provide --wsi_path for visualization.")
            return

        # Update config with model path
        visualization_config.abmil_weights_path = args.model_path
        
        # Initialize Visualizer
        visualizer = ABMIL_Visualizer(visualization_config, args.wsi_path)
        
        # Run Pipeline
        visualizer.run_pipeline()
        
    else:
        raise ValueError(f"{args.mode} mode has not been supported!")

if __name__ == "__main__":
    runtime_config.model_name = 'linear_probe'
    runtime_config.input_dim = 768
    runtime_config.save_dir = "/mnt/gml/GML/Experiments/Experiment3/weights"
    runtime_config.best_epochs = 20

    sys.argv = [
        "main",
        "--mode", "train",
        '--folds', "5"
    ]
    
    main()
