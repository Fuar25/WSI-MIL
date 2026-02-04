import optuna
import numpy as np
import copy
import torch
import os
import csv
from core.trainer import Trainer

class HyperParameterOptimizer:
    def __init__(self, model_builder, dataset, base_config):
        self.model_builder = model_builder
        self.dataset = dataset
        self.base_config = base_config

    def objective(self, trial):
        # Create a copy of the config for this trial
        config = copy.deepcopy(self.base_config)
        
        # Enforce deterministic behavior for fair comparison
        # Reset seeds at the start of each trial so that the same params yield the same result
        seed = config.training.seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
        # Define search space
        config.training.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        config.training.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        config.model.dropout = trial.suggest_float("dropout", 0, 0.5)
        config.model.hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 768, 1024])
        config.model.n_heads = trial.suggest_categorical("n_heads", [1, 4, 8])
        
        # Update save directory for this trial to avoid overwriting
        config.logging.save_dir = f"{self.base_config.logging.save_dir}/trial_{trial.number}"
        config.logging.log_test_results = False # Disable detailed logging for speed/storage
        
        # Initialize Trainer
        trainer = Trainer(self.model_builder, self.dataset, config)
        
        # Run Cross-Validation (using fewer folds for speed if needed, but keeping robust)
        # We can use the average Test AUC as the metric to maximize
        fold_results = trainer.cross_validate(k_folds=config.training.k_folds)
        
        # Calculate average Test AUC
        avg_test_auc = np.mean([r['test_auc'] for r in fold_results])
        avg_test_acc = np.mean([r['test_acc'] for r in fold_results])

        # Log results to CSV
        log_path = os.path.join(self.base_config.logging.save_dir, "hpo_log.csv")
        os.makedirs(self.base_config.logging.save_dir, exist_ok=True)

        # Prepare log data
        log_data = {
            "trial": trial.number,
            "avg_test_auc": avg_test_auc,
            "avg_test_acc": avg_test_acc,
        }
        log_data.update(trial.params)

        # Write to CSV
        file_exists = os.path.isfile(log_path)
        with open(log_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_data)
        
        return avg_test_auc

    def optimize(self, n_trials=20, study_name="abmil_optimization"):
        print(f"Starting Bayesian Optimization with {n_trials} trials...")
        
        study = optuna.create_study(direction="maximize", study_name=study_name)
        study.optimize(self.objective, n_trials=n_trials)
        
        print("\nOptimization Complete!")
        print("Best trial:")
        trial = study.best_trial
        
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
            
        return study.best_trial
