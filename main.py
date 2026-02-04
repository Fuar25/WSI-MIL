import argparse
import sys

from config.config import runtime_config, visualization_config
from core.registry import create_dataset, create_model
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

    # Initialize Dataset via factory
    dataset_cfg = runtime_config.dataset
    if dataset_cfg.data_paths is None:
        print("Error: runtime_config.dataset.data_paths must be provided for dataset initialization.")
        return

    dataset = create_dataset(
        dataset_cfg.dataset_name,
        data_paths=dataset_cfg.data_paths,
        feature_key=dataset_cfg.feature_key,
        coords_key=dataset_cfg.coords_key,
        patient_id_pattern=dataset_cfg.patient_id_pattern,
        binary_mode=dataset_cfg.binary_mode,
    )

    num_dataset_classes = len(dataset.classes)
    if num_dataset_classes == 2:
        runtime_config.model.num_classes = 1
        print("Auto-configured num_classes to 1 (binary classification).")
    else:
        runtime_config.model.num_classes = num_dataset_classes
        print(f"Auto-configured num_classes to {num_dataset_classes} (multi-class).")
    
    if len(dataset) == 0:
        print("Error: No data found.")
        return

    def model_builder():
        model_cfg = runtime_config.model
        return create_model(
            model_cfg.model_name,
            input_dim=model_cfg.input_dim,
            hidden_dim=model_cfg.hidden_dim,
            num_classes=model_cfg.num_classes,
            dropout=model_cfg.dropout,
            attention_dim=model_cfg.attention_dim,
            gated=model_cfg.gated,
            encoder_dropout=model_cfg.encoder_dropout,
            classifier_dropout=model_cfg.classifier_dropout,
        )

    trainer = Trainer(model_builder, dataset, runtime_config)
    
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

        optimizer = HyperParameterOptimizer(model_builder, dataset, runtime_config)
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

    runtime_config.model.model_name = 'mil'
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
