import argparse
import sys

from config.config import runtime_config, visualization_config
from data_loader.dataset import WSIFeatureDataset, H5FeatureDataset
from models.mil_models import MILModel
from models.linear_probe import LinearProbe
from core.trainer import Trainer
from core.abmil_visualizer import ABMIL_Visualizer

def main():
    parser = argparse.ArgumentParser(description="WSI Training with TRIDENT Integration")
    parser.add_argument('--mode', type=str, default='cv',
                        help='Mode: train (full dataset for deployment), cv (cross-validation), vis (visualization), search (hyperparameter search)')
    parser.add_argument('--folds', type=int, default=None, help='Number of folds for cross-validation (overrides runtime_config)')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of trials for hyperparameter search')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model for visualization')
    parser.add_argument('--wsi_path', type=str, default=None, help='Path to WSI file for visualization')
    
    args = parser.parse_args()

    # Initialize Dataset
    if runtime_config.dataset.dataset_type == 'h5':
        if runtime_config.dataset.data_paths is None:
            print("Error: runtime_config.dataset.data_paths must be provided for H5 dataset.")
            return
        dataset = H5FeatureDataset(
            data_paths=runtime_config.dataset.data_paths,
            feature_key=runtime_config.dataset.feature_key,
            binary_mode=runtime_config.dataset.binary_mode
        )
        
        # Auto-configure num_classes based on dataset
        # For binary classification (2 classes), use num_classes=1
        # For multi-class (>2 classes), use num_classes=number of classes
        num_dataset_classes = len(dataset.classes)
        if num_dataset_classes == 2:
            runtime_config.model.num_classes = 1  # Binary classification
            print(f"Auto-configured num_classes to 1 (binary classification) for {num_dataset_classes} classes.")
        else:
            # runtime_config.model.num_classes = num_dataset_classes
            # print(f"Auto-configured num_classes to {runtime_config.model.num_classes} (multi-class).")
            raise ValueError(f"{num_dataset_classes} classes has not been supported!")

    elif runtime_config.dataset.dataset_type == 'npy':
        dataset = WSIFeatureDataset(
            root_dir=runtime_config.dataset.root_dir,
            csv_path=runtime_config.dataset.csv_path,
            label_col=runtime_config.dataset.label_col,
            sample_id_col=runtime_config.dataset.sample_id_col,
            feature_suffix=runtime_config.dataset.feature_suffix
        )
    else:
        raise TypeError(f"{runtime_config.dataset.dataset_type} type has not been supported!")
    
    if len(dataset) == 0:
        print("Error: No data found.")
        return

    # Initialize Trainer
    if runtime_config.model.model_name == 'linear_probe':
        model_class = LinearProbe
    else:
        # Assume it's a MIL-Lab model
        # Update mil_lab_model_name if model_name is a specific MIL-Lab model string
        # If model_name is generic 'abmil', we keep mil_lab_model_name as 'abmil' (or whatever is in config)
        # If model_name is specific (e.g. 'clam_sb', 'abmil.base...'), we update mil_lab_model_name
        if runtime_config.model.model_name != 'abmil': 
             runtime_config.model.mil_lab_model_name = runtime_config.model.model_name
        
        model_class = MILModel

    trainer = Trainer(model_class, dataset, runtime_config)
    
    if args.mode == 'train':
        trainer.train_full_dataset()
        
    elif args.mode == 'cv':
        k_folds = args.folds if args.folds is not None else runtime_config.training.k_folds
        trainer.cross_validate(k_folds=k_folds)

    elif args.mode == 'search':
        try:
            from core.hpo import HyperParameterOptimizer
        except ImportError:
            print("Error: 'optuna' is required for hyperparameter search. Please install it via 'pip install optuna'.")
            return

        optimizer = HyperParameterOptimizer(model_class, dataset, runtime_config)
        optimizer.optimize(n_trials=args.n_trials)

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
    
    runtime_config.dataset.data_paths = {
        'positive': '/mnt/5T/Tiff/Experiments/Experiment1/Ki-67/MALT/10x_256px_0px_overlap/features_uni_v2',
        'negative': '/mnt/5T/Tiff/Experiments/Experiment1/Ki-67/Reactive/10x_256px_0px_overlap/features_uni_v2'
    }

    runtime_config.logging.save_dir = "/mnt/5T/Tiff/Experiments/Experiment1/results/Ki-67"
    runtime_config.training.device = "cuda:2"
    runtime_config.training.seed = 42
    runtime_config.training.voting_strategy = "majority"  # 使用硬投票聚合到patient级

    runtime_config.model.model_name = 'abmil'
    runtime_config.model.input_dim = 1536
    runtime_config.model.hidden_dim = 512
    runtime_config.model.n_heads = 4
    runtime_config.model.dropout = 0.2
    runtime_config.model.gated = True

    
    sys.argv = [
        "main",
        "--mode", "cv",
        "--folds", "5",
        '--n_trials', "100"
    ]
    
    main()
