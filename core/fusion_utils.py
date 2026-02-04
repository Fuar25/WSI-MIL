"""Shared utilities for multi-stain fusion strategies used by voting.py."""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config.config import runtime_config
from core.registry import create_dataset, create_model
from core.trainer import Trainer

# ANSI color helpers for terminal clarity
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"


def build_dataset():
    dataset_cfg = runtime_config.dataset
    dataset = create_dataset(
        dataset_cfg.dataset_name,
        data_paths=dataset_cfg.data_paths,
        feature_key=dataset_cfg.feature_key,
        coords_key=dataset_cfg.coords_key,
        patient_id_pattern=dataset_cfg.patient_id_pattern,
        binary_mode=dataset_cfg.binary_mode,
    )
    num_dataset_classes = len(dataset.classes)
    runtime_config.model.num_classes = 1 if num_dataset_classes == 2 else num_dataset_classes
    return dataset


def build_model_builder():
    model_cfg = runtime_config.model

    def _builder():
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

    return _builder


def run_cv_for_stain(stain_cfg: Dict[str, Any], k_folds: int, save_features: bool = False):
    """
    Run cross-validation for a single stain.
    
    Args:
        stain_cfg: Configuration dict containing 'name', 'data_paths', 'save_dir', 'weight'
        k_folds: Number of folds for cross-validation
        save_features: If True, save features for fusion
        
    Returns:
        fold_results from trainer.cross_validate()
    """
    runtime_config.dataset.data_paths = stain_cfg["data_paths"]
    runtime_config.logging.save_dir = stain_cfg["save_dir"]
    runtime_config.logging.save_features = save_features

    print(f"\n{CYAN}================ Running stain: {stain_cfg['name']} ================{RESET}")
    dataset = build_dataset()
    if len(dataset) == 0:
        raise RuntimeError(f"No data found for stain {stain_cfg['name']}")

    model_builder = build_model_builder()
    trainer = Trainer(model_builder, dataset, runtime_config)
    return trainer.cross_validate(k_folds=k_folds)


def get_common_patients(stain_runs: List[Dict], k_folds: int, split: str = 'test') -> Dict[int, set]:
    """
    Get the set of patient IDs that appear across all stains for each fold.
    
    Args:
        stain_runs: List of stain configuration dicts
        k_folds: Number of folds
        split: 'train' or 'test'
        
    Returns:
        Dict mapping fold number to set of common patient IDs
    """
    import torch
    
    common_patients = {}
    
    for fold in range(1, k_folds + 1):
        fold_patients = None
        
        for run in stain_runs:
            features_path = os.path.join(run["save_dir"], f'fold_{fold}_features.pt')
            if not os.path.exists(features_path):
                print(f"{YELLOW}Warning: Missing features for stain {run['name']} fold {fold}{RESET}")
                continue
                
            data = torch.load(features_path, weights_only=False)
            patient_ids = set(data[split]['patient_ids'])
            
            if fold_patients is None:
                fold_patients = patient_ids
            else:
                fold_patients = fold_patients.intersection(patient_ids)
        
        common_patients[fold] = fold_patients if fold_patients else set()
    
    return common_patients


def load_and_align_features(
    stain_runs: List[Dict], 
    fold: int, 
    split: str = 'test',
    use_all_patients: bool = True
) -> Dict[str, Any]:
    """
    Load features from all stains for a given fold and align by patient ID.
    Handles missing modalities by zero-padding.
    
    Args:
        stain_runs: List of stain configuration dicts
        fold: Fold number (1-indexed)
        split: 'train' or 'test'
        use_all_patients: If True, use union of all patients (with zero-padding for missing).
                          If False, use intersection (only patients with all stains).
                          
    Returns:
        Dict containing:
            'patient_ids': List of patient IDs
            'features': np.ndarray of shape (n_patients, total_feature_dim)
            'labels': np.ndarray of shape (n_patients,)
            'stain_dims': Dict mapping stain name to feature dimension
            'available_stains': Dict mapping patient_id to list of available stains
    """
    import torch
    
    # First pass: collect all data and determine dimensions
    stain_data = {}
    stain_dims = {}
    all_patient_ids = set()
    patient_labels = {}
    
    for run in stain_runs:
        features_path = os.path.join(run["save_dir"], f'fold_{fold}_features.pt')
        if not os.path.exists(features_path):
            print(f"{YELLOW}Warning: Missing features for stain {run['name']} fold {fold}{RESET}")
            continue
            
        data = torch.load(features_path, weights_only=False)
        split_data = data[split]
        
        stain_name = run["name"]
        stain_dims[stain_name] = split_data['features'].shape[1]
        
        # Create patient -> feature mapping
        stain_data[stain_name] = {}
        for i, pid in enumerate(split_data['patient_ids']):
            stain_data[stain_name][pid] = split_data['features'][i]
            all_patient_ids.add(pid)
            # Store label (should be consistent across stains)
            if pid not in patient_labels:
                patient_labels[pid] = split_data['labels'][i]
    
    if not stain_data:
        raise RuntimeError(f"No feature data loaded for fold {fold}")
    
    # Determine which patients to include
    if use_all_patients:
        # Union: include all patients, zero-pad missing stains
        final_patient_ids = sorted(all_patient_ids)
    else:
        # Intersection: only patients with all stains
        final_patient_ids = sorted(all_patient_ids)
        for stain_name, data in stain_data.items():
            final_patient_ids = [pid for pid in final_patient_ids if pid in data]
    
    # Total feature dimension (sum of all stain dimensions)
    total_dim = sum(stain_dims.values())
    stain_order = list(stain_dims.keys())
    
    # Second pass: build aligned feature matrix with zero-padding
    n_patients = len(final_patient_ids)
    features = np.zeros((n_patients, total_dim), dtype=np.float32)
    labels = np.zeros(n_patients, dtype=np.int64)
    available_stains = {}
    
    for i, pid in enumerate(final_patient_ids):
        available_stains[pid] = []
        offset = 0
        
        for stain_name in stain_order:
            dim = stain_dims[stain_name]
            
            if stain_name in stain_data and pid in stain_data[stain_name]:
                features[i, offset:offset+dim] = stain_data[stain_name][pid]
                available_stains[pid].append(stain_name)
            # else: remains zero (zero-padding)
            
            offset += dim
        
        labels[i] = patient_labels.get(pid, 0)
    
    return {
        'patient_ids': final_patient_ids,
        'features': features,
        'labels': labels,
        'stain_dims': stain_dims,
        'stain_order': stain_order,
        'available_stains': available_stains
    }
