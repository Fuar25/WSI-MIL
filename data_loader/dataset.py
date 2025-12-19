
import os
import torch
import numpy as np
import pandas as pd
import h5py
from typing import Tuple, Dict, Any, Optional, List
from core.base import BaseDataset

class WSIFeatureDataset(BaseDataset):
    """
    A dataset class for loading WSI features from NPY files, inheriting from BaseDataset.
    """
    def __init__(self, 
                 root_dir: str, 
                 csv_path: str, 
                 label_col: str = 'GT',
                 sample_id_col: str = 'file_name',
                 feature_suffix: str = '_ctranspath.npy'):
        """
        Args:
            root_dir: Directory containing the feature .npy files.
            csv_path: Path to the CSV file with labels.
            label_col: Column name for the label in the CSV.
            sample_id_col: Column name for the sample ID in the CSV.
            feature_suffix: Suffix of the feature files (e.g., '_ctranspath.npy').
        """
        self.root_dir = root_dir
        self.feature_suffix = feature_suffix
        
        # Load metadata
        self.metadata_df = pd.read_csv(csv_path)
        
        self.data = []
        self._scan_files(label_col, sample_id_col)
        
        print(f"Dataset loaded: {len(self.data)} samples found.")

    def _scan_files(self, label_col, sample_id_col):
        """
        Scans the root directory and matches files with the CSV.
        """
        # Create a lookup dictionary for labels
        # Ensure sample_id_col is treated as string for matching
        self.metadata_df[sample_id_col] = self.metadata_df[sample_id_col].astype(str)
        label_map = dict(zip(self.metadata_df[sample_id_col], self.metadata_df[label_col]))
        
        if not os.path.exists(self.root_dir):
            print(f"Warning: Directory {self.root_dir} does not exist.")
            return

        # Walk through the directory to find feature files
        for root, _, files in os.walk(self.root_dir):
            for filename in files:
                if filename.endswith(self.feature_suffix):
                    # Extract sample ID from filename
                    # e.g., "sample1_ctranspath.npy" -> "sample1"
                    sample_id = filename.replace(self.feature_suffix, '')
                    
                    if sample_id in label_map:
                        file_path = os.path.join(root, filename)
                        label = label_map[sample_id]
                        self.data.append({
                            'sample_id': sample_id,
                            'feature_path': file_path,
                            'label': label
                        })

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        
        # Load features
        features = np.load(item['feature_path'])
        features = torch.from_numpy(features).float()
        
        # Handle label
        label = torch.tensor(item['label']).float() # Assuming binary/regression. Change to long() for multi-class
        
        return features, label

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]
    
    def get_coords(self, idx: int) -> Optional[np.ndarray]:
        """
        Returns coordinates for the given index if available.
        For .npy files, this might not be available unless stored separately.
        """
        # Placeholder: If you have coordinate files, load them here.
        # For now, return None as .npy files in this codebase don't seem to have coords.
        return None

class H5FeatureDataset(BaseDataset):
    """
    A dataset class for loading WSI features from H5 files, supporting multiple class directories.
    """
    def __init__(self, 
                 data_paths: Dict[str, str],
                 feature_key: str = 'features',
                 binary_mode: bool = None):
        """
        Args:
            data_paths: Dictionary mapping class names to directory paths.
                        Example: {'Normal': '/path/to/normal', 'Tumor': '/path/to/tumor'}
            feature_key: Key in the H5 file to access features.
            binary_mode: If True, treat as binary classification (labels as float).
                        If False, treat as multi-class (labels as long).
                        If None, auto-detect based on number of classes.
        """
        self.data_paths = data_paths
        self.feature_key = feature_key
        self.classes = sorted(list(data_paths.keys()))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Auto-detect binary mode if not specified
        if binary_mode is None:
            self.binary_mode = len(self.classes) == 2
        else:
            self.binary_mode = binary_mode
        
        self.data = []
        self._scan_files()
        
        print(f"H5 Dataset loaded: {len(self.data)} samples found across {len(self.classes)} classes.")
        print(f"Class mapping: {self.class_to_idx}")
        print(f"Mode: {'Binary' if self.binary_mode else 'Multi-class'}")

    def _scan_files(self):
        for cls_name, dir_path in self.data_paths.items():
            if not os.path.exists(dir_path):
                print(f"Warning: Directory {dir_path} for class {cls_name} does not exist.")
                continue
                
            label = self.class_to_idx[cls_name]
            
            for root, _, files in os.walk(dir_path):
                for filename in files:
                    if filename.endswith('.h5'):
                        file_path = os.path.join(root, filename)
                        self.data.append({
                            'sample_id': filename.replace('.h5', ''),
                            'feature_path': file_path,
                            'label': label,
                            'class_name': cls_name
                        })

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        
        with h5py.File(item['feature_path'], 'r') as f:
            features = np.array(f[self.feature_key])
        
        features = torch.from_numpy(features).float()
        
        # Return appropriate label type based on mode
        if self.binary_mode:
            # Binary classification: return as float (0.0 or 1.0)
            label = torch.tensor(item['label']).float()
        else:
            # Multi-class: return as long (integer class index)
            label = torch.tensor(item['label']).long()
        
        return features, label

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]
        
    def get_coords(self, idx: int) -> Optional[np.ndarray]:
        item = self.data[idx]
        with h5py.File(item['feature_path'], 'r') as f:
            if 'coords' in f:
                return np.array(f['coords'])
        return None

